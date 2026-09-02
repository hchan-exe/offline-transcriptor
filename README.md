# Offline Transcriptor

Offline-first audio and video transcription for Windows. Converts local recordings to GitHub-Flavored Markdown transcripts with `[MM:SS]` timestamps using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on CPU.

## Supported formats

`.mp4`, `.mkv`, `.avi`, `.mov`, `.mp3`, `.wav`, `.m4a`

## Features

- CLI and Tkinter GUI (drag-and-drop)
- CPU-optimised defaults (`int8`, VAD filtering)
- English, Mandarin, and Cantonese presets
- Transcripts saved as Markdown with timestamps

## Requirements

- Python 3.10+
- Windows 10/11
- ~2 GB disk space for the default `small` Whisper model (downloaded on first run)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**GUI:** Double-click `transcribe.bat`, or:

```bash
python transcribe.py --gui
```

**CLI:**

```bash
python transcribe.py path\to\recording.mp4
python transcribe.py path\to\folder --recursive
```

Output is written to `Audio Transcript_Output/` (created automatically).

## License

MIT
