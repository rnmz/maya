# Maya - AI Subtitle Translator

Maya is a small desktop application that creates subtitles from video and translates them with locally running AI models.

It combines automatic speech recognition, word-level timestamp alignment, and translation in one simple workflow. The goal is to make subtitle creation easier without relying on a hosted translation service.

## Features

- Extracts audio from video files using FFmpeg.
- Recognizes speech with **Qwen3-ASR**.
- Generates subtitle timing with **Qwen3 Forced Aligner** when the detected language is supported.
- Translates subtitle segments using a local **Ollama** model.
- Lets you review and retranslate individual subtitle lines.
- Exports subtitles to the standard `.srt` format.
- Reuses cached model files and, while the application is running, can keep loaded recognition models in memory to reduce repeated loading.

## Requirements

- Python 3.10 or newer
- FFmpeg installed and available in `PATH`
- Ollama installed and running locally
- A compatible Python installation for PyTorch and Transformers
- Enough RAM and, optionally, GPU VRAM to run the selected recognition models

GPU acceleration is used when CUDA is available. Otherwise, speech recognition runs on the CPU and may be considerably slower.

## Models

Maya uses these Hugging Face models for speech processing:

- `Qwen/Qwen3-ASR-1.7B-hf` — speech recognition
- `Qwen/Qwen3-ForcedAligner-0.6B-hf` — word-level timestamp alignment

Translation is performed by a model available in your local Ollama installation. You can select the model in the application.

The first run may download the speech-processing models. Later runs should use the local Hugging Face cache unless the cache is removed, a different environment is used, or the model files are unavailable.

## Installation

### 1. Install system dependencies

Install **FFmpeg** and make sure the `ffmpeg` command works from your terminal.

Install **Ollama**, start it, and download a translation model. For example:

```bash
ollama pull qwen2.5:7b
```

You can use another model supported by your Ollama installation.

### 2. Install Python dependencies

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The speech-processing code requires a Transformers version that provides the Qwen3 ASR and forced-alignment APIs used by the application. Check the model documentation and your PyTorch/CUDA compatibility before choosing a build.

### 3. Run Maya

Place `index.html` next to the Python application file, then run:

```bash
python app.py
```

If your Python file has a different name, use that filename instead. The application opens in a desktop window.

## Usage

1. Start Ollama and confirm that your chosen translation model is available.
2. Open Maya and select a video file.
3. Choose an Ollama model and the target subtitle language.
4. Optionally enter a custom translation instruction.
5. Start processing and wait for recognition and translation to finish.
6. Review the subtitle lines and retranslate individual lines if needed.
7. Export the result as an `.srt` file.

## Notes

- Processing time depends on video length, hardware, and model size.
- The forced aligner is used only for languages supported by the application. If alignment fails or is unavailable, the application falls back to a broader timestamp for the recognized chunk.
- Ollama must be reachable at `http://localhost:11434` unless the application configuration is changed.
- Model weights can require substantial disk space, RAM, and VRAM.
- Keep your model cache intact to avoid unnecessary downloads.

## License
**MIT**