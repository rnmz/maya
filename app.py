import gc
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np
import requests
import webview

try:
    import torch
    from transformers import (
        AutoModelForMultimodalLM,
        AutoModelForTokenClassification,
        AutoProcessor,
    )
    TORCH_AVAILABLE = True
    IMPORT_ERROR = ""
except ImportError as e:
    TORCH_AVAILABLE = False
    IMPORT_ERROR = str(e)


ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

ALIGNER_LANGUAGES = {
    "Chinese", "English", "Cantonese", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
}

NO_SPACE_LANGUAGES = {"Chinese", "Cantonese", "Japanese"}

CHUNK_SECONDS = 30
SNAP_SECONDS = 3.0 
MAX_SEG_SECONDS = 6.0
MAX_SEG_CHARS = 84
GAP_BREAK = 0.8 
SENTENCE_END = ".!?…。！？"


def format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    secs, millis = divmod(total_ms, 1000)
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def plan_chunks(samples: np.ndarray, rate: int) -> list:
    total = len(samples)
    target = CHUNK_SECONDS * rate
    if total <= target * 1.2:
        return [(0, total)]

    frame = int(rate * 0.05)
    snap = int(SNAP_SECONDS * rate)
    bounds = [0]
    while total - bounds[-1] > target * 1.2:
        center = bounds[-1] + target
        lo = max(bounds[-1] + frame, center - snap)
        hi = min(total - frame, center + snap)
        part = samples[lo:hi].astype(np.float32)
        n_frames = len(part) // frame
        if n_frames == 0:
            cut = center
        else:
            energy = (part[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1)
            cut = lo + int(np.argmin(energy)) * frame + frame // 2
        bounds.append(cut)
    bounds.append(total)
    return list(zip(bounds[:-1], bounds[1:]))


def group_words(words: list, no_space: bool) -> list:
    joiner = "" if no_space else " "
    result = []
    current = []

    def flush():
        if current:
            text = joiner.join(w[0] for w in current).strip()
            if text:
                result.append((current[0][1], current[-1][2], text))
            current.clear()

    for word in words:
        if current:
            gap = word[1] - current[-1][2]
            length = word[2] - current[0][1]
            chars = len(joiner.join(w[0] for w in current)) + len(word[0])
            if gap > GAP_BREAK or length > MAX_SEG_SECONDS or chars > MAX_SEG_CHARS:
                flush()
        current.append(word)
        if word[0].rstrip()[-1:] in SENTENCE_END and (word[2] - current[0][1]) >= 1.0:
            flush()
    flush()
    return result


class AutoSubAPI:

    def __init__(self):
        self._window = None
        self._asr_processor = None
        self._asr_model = None
        self._aligner_processor = None
        self._aligner_model = None
        self._ollama_base = "http://localhost:11434"
        self._is_processing = False

    def _set_window(self, window):
        self._window = window

    def _emit_status(self, step: str, progress: int, message: str, stats: dict = None):
        """Sends progress and metrics to the UI."""
        if self._window:
            payload = {
                "step": step,
                "progress": progress,
                "message": message,
                "stats": stats or {},
            }
            self._window.evaluate_js(f"window.updatePipelineStatus({json.dumps(payload)})")

    def get_system_info(self):
        """GPU status and list of Ollama models for the start screen."""
        gpu_name = "CPU Mode (CUDA not found)"
        vram_gb = 0
        cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)

        models = []
        try:
            resp = requests.get(f"{self._ollama_base}/api/tags", timeout=2)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass

        return {
            "gpu": gpu_name,
            "vram": vram_gb,
            "cuda": cuda,
            "ollama_models": models if models else ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"],
        }

    def select_video_file(self):
        file_types = ("Video Files (*.mp4;*.mkv;*.mov;*.avi;*.webm)", "All files (*.*)")
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN, allow_multiple=False, file_types=file_types
        )
        if result and len(result) > 0:
            filepath = result[0]
            size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)
            return {"path": filepath, "name": os.path.basename(filepath), "size_mb": size_mb}
        return None

    def start_pipeline(self, video_path: str, ollama_model: str, target_lang: str, custom_prompt: str):
        if self._is_processing:
            return {"status": "busy"}
        self._is_processing = True
        thread = threading.Thread(
            target=self._run_pipeline_worker,
            args=(video_path, ollama_model, target_lang, custom_prompt),
            daemon=True,
        )
        thread.start()
        return {"status": "started"}

    def retranslate_single_row(self, index: int, source_text: str, model: str, target_lang: str, custom_prompt: str):
        """⚡ button in the table row: retranslates a specific phrase."""
        t0 = time.time()
        translated = self._translate_text_ollama(source_text, model, target_lang, custom_prompt)
        latency_ms = int((time.time() - t0) * 1000)
        return {"index": index, "translation": translated, "latency_ms": latency_ms}

    def export_srt_file(self, segments: list):
        file_types = ("SubRip Subtitle (*.srt)", "All files (*.*)")
        save_path = self._window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename="translated_subtitles.srt",
            file_types=file_types,
        )
        if not save_path:
            return {"saved": False}

        if isinstance(save_path, (list, tuple)):
            save_path = save_path[0]

        srt_lines = []
        for idx, seg in enumerate(segments, start=1):
            start_tc = seg.get("start", "00:00:00,000")
            end_tc = seg.get("end", "00:00:03,000")
            text = seg.get("translation") or seg.get("source") or ""
            srt_lines.append(f"{idx}\n{start_tc} --> {end_tc}\n{text.strip()}\n")

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        return {"saved": True, "path": save_path}

    def _run_pipeline_worker(self, video_path: str, ollama_model: str, target_lang: str, custom_prompt: str):
        start_time = time.time()
        try:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("FFmpeg not found in PATH.")

            with tempfile.TemporaryDirectory(prefix="autosub_") as tmp_dir:
                self._emit_status("audio", 10, "Extracting audio track (16kHz Mono WAV)...")
                wav_path = os.path.join(tmp_dir, "audio.wav")
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    wav_path,
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0:
                    err_tail = proc.stderr.decode("utf-8", errors="replace")[-400:]
                    raise RuntimeError(f"FFmpeg exited with an error: {err_tail}")

                with wave.open(wav_path, "rb") as wf:
                    rate = wf.getframerate()
                    samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                duration = round(len(samples) / float(rate), 1)

                self._emit_status("asr", 25, f"Loading {ASR_MODEL_ID}...", {"duration": duration})
                segments = self._transcribe(samples, rate, tmp_dir)

            if not segments:
                raise RuntimeError("No speech found in the video.")

            assert self._window is not None
            self._window.evaluate_js(f"window.loadSubtitles({json.dumps(segments)})")

            total_segs = len(segments)
            for idx, seg in enumerate(segments):
                pct = 55 + int(((idx + 1) / max(1, total_segs)) * 40)
                self._emit_status(
                    "translate",
                    pct,
                    f"Translating via {ollama_model} ({idx + 1}/{total_segs})...",
                    {"segments": total_segs},
                )
                seg["translation"] = self._translate_text_ollama(
                    seg["source"], ollama_model, target_lang, custom_prompt
                )
                self._window.evaluate_js(f"window.updateSubtitleRow({idx}, {json.dumps(seg)})")

            elapsed = round(time.time() - start_time, 1)
            self._emit_status(
                "done", 100,
                f"Done! Processed {total_segs} segments in {elapsed} sec.",
                {"segments": total_segs, "elapsed": elapsed},
            )

        except Exception as e:
            self._emit_status("error", 0, f"Pipeline error: {e}")
        finally:
            self._is_processing = False


    def _load_asr(self, device: str, dtype):
        if self._asr_model is None:
            self._asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
            self._asr_model = (
                AutoModelForMultimodalLM.from_pretrained(ASR_MODEL_ID, dtype=dtype).to(device).eval()
            )

    def _load_aligner(self, device: str, dtype):
        if self._aligner_model is None:
            self._aligner_processor = AutoProcessor.from_pretrained(ALIGNER_MODEL_ID)
            self._aligner_model = (
                AutoModelForTokenClassification.from_pretrained(ALIGNER_MODEL_ID, dtype=dtype)
                .to(device)
                .eval()
            )

    def _release_models(self):
        self._asr_processor = None
        self._asr_model = None
        self._aligner_processor = None
        self._aligner_model = None
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _transcribe(self, samples: np.ndarray, rate: int, tmp_dir: str) -> list:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "Failed to import torch/transformers (requires transformers>=5.13.0): "
                f"{IMPORT_ERROR}"
            )

        use_cuda = torch.cuda.is_available()
        device = "cuda:0" if use_cuda else "cpu"
        if use_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self._load_asr(device, dtype)

        chunks = plan_chunks(samples, rate)
        total_chunks = len(chunks)
        segments = []

        for i, (a, b) in enumerate(chunks):
            pct = 30 + int((i / total_chunks) * 25)
            self._emit_status(
                "asr", pct, f"Transcribing speech ({i + 1}/{total_chunks})..."
            )

            chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
            with wave.open(chunk_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(samples[a:b].tobytes())

            text, language = self._recognize_chunk(chunk_path)
            if not text:
                continue

            offset = a / rate
            chunk_end = b / rate
            no_space = language in NO_SPACE_LANGUAGES

            phrases = []
            if language in ALIGNER_LANGUAGES:
                try:
                    self._load_aligner(device, dtype)
                    words = self._align_chunk(chunk_path, text, language)
                    words = [(w, s + offset, e + offset) for (w, s, e) in words]
                    phrases = group_words(words, no_space)
                except Exception:
                    phrases = []

            if not phrases:
                phrases = [(offset, chunk_end, text)]

            for start_s, end_s, phrase in phrases:
                end_s = max(end_s, start_s + 0.2)
                segments.append({
                    "id": len(segments) + 1,
                    "start": format_srt_time(start_s),
                    "end": format_srt_time(end_s),
                    "duration": round(end_s - start_s, 1),
                    "source": phrase,
                    "translation": "...",
                })

        return segments

    def _recognize_chunk(self, wav_path: str):
        proc = self._asr_processor
        model = self._asr_model
        inputs = proc.apply_transcription_request(audio=wav_path).to(model.device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = proc.decode(generated_ids, return_format="parsed")[0]
        text = (parsed.get("transcription") or "").strip()
        language = parsed.get("language") or ""
        return text, language

    def _align_chunk(self, wav_path: str, transcript: str, language: str) -> list:
        """Audio chunk + text -> [(word, start_sec, end_sec), ...] within the chunk."""
        proc = self._aligner_processor
        model = self._aligner_model
        inputs, word_lists = proc.prepare_forced_aligner_inputs(
            audio=wav_path, transcript=transcript, language=language
        )
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            outputs = model(**inputs)
        items = proc.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=model.config.timestamp_token_id,
        )[0]
        return [(it["text"], float(it["start_time"]), float(it["end_time"])) for it in items]


    def _translate_text_ollama(self, text: str, model: str, target_lang: str, custom_prompt: str) -> str:
        if not text.strip():
            return ""

        sys_instruction = (
            f"You are a professional subtitle translator. Translate the user's subtitle line into {target_lang}. "
            f"Output ONLY the translated text without quotes, explanations, or notes. "
            f"{custom_prompt}"
        ).strip()

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_instruction},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }

        try:
            resp = requests.post(f"{self._ollama_base}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return content.strip()
        except Exception as e:
            return f"[Ollama Error: {e}]"


if __name__ == "__main__":
    api = AutoSubAPI()
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

    window = webview.create_window(
        title="Maya subtitle app",
        url=html_path,
        js_api=api,
        width=1540,
        height=920,
        min_size=(1200, 760),
        background_color="#F4F5F7",
    )
    api._set_window(window)
    webview.start(debug=False)