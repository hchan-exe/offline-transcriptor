#!/usr/bin/env python3
"""
Offline-first audio/video transcription pipeline for Windows 11.

Converts local recordings (.mp4, .mkv, .avi, .mov, .mp3, .wav, .m4a) into
GitHub-Flavored Markdown transcripts with [MM:SS] timestamps using
faster-whisper on CPU (int8).

Supports CLI and a Tkinter GUI (--gui / no args via transcribe.bat).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".m4a"}

# Tuned for CPU laptops (e.g. Intel U-series + Iris Xe, no NVIDIA CUDA).
# Models: small / medium / turbo only (no distil).
DEFAULT_MODEL = "small"
FALLBACK_MODEL = "medium"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 1  # best CPU speed/quality tradeoff for meetings
VAD_FILTER = True  # skip silence — large win on 1:1s / webinars
# Leave a couple of logical cores for Windows UI so the machine stays responsive
# and the U-series chip is less likely to thermal-throttle.
_CPU_COUNT = os.cpu_count() or 4
CPU_THREADS = max(2, min(8, _CPU_COUNT - 2 if _CPU_COUNT > 4 else _CPU_COUNT))

MODEL_CHOICES = (
    "small",   # Fast (recommended on CPU)
    "medium",  # Balanced quality / speed
    "turbo",   # Highest quality, slowest on CPU
)

# All Markdown transcripts are written here (created on first save).
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "Audio Transcript_Output"

# Display label -> Whisper language code (None = auto-detect).
# Cantonese: standard Whisper has no dedicated "yue" code on turbo/medium,
# so we force zh + a Cantonese initial_prompt to bias decoding.
LANGUAGE_PRESETS: dict[str, Optional[str]] = {
    "English": "en",
    "Chinese (Mandarin)": "zh",
    "Chinese (Cantonese)": "zh",
    "Auto-Detect": None,
}

CANTONESE_LABEL = "Chinese (Cantonese)"
CANTONESE_INITIAL_PROMPT = (
    "呢個係粵語會議錄音。以下用繁體中文記錄粵語內容。"
)

DEFAULT_LANGUAGE_LABEL = "English"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transcribe")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TranscriptResult:
    media_path: Path
    success: bool
    markdown: str = ""
    out_path: Optional[Path] = None
    detected_lang: str = "unknown"
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Convert seconds to [MM:SS] (supports hours as MM beyond 59)."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    return f"[{minutes:02d}:{secs:02d}]"


def language_code_for_label(label: str) -> Optional[str]:
    """Map a UI/CLI language label to a Whisper language code."""
    if label in LANGUAGE_PRESETS:
        return LANGUAGE_PRESETS[label]
    normalized = label.strip().lower()
    if normalized in {"", "auto", "detect", "none", "auto-detect"}:
        return None
    if normalized in {"en", "english"}:
        return "en"
    if normalized in {"zh", "mandarin", "chinese", "cmn"}:
        return "zh"
    if normalized in {"yue", "cantonese", "zh-yue", "zh_hk"}:
        return "zh"  # Whisper code; Cantonese bias via initial_prompt
    return normalized


def initial_prompt_for_label(label: str) -> Optional[str]:
    """Optional Whisper initial_prompt for language-specific bias."""
    if label == CANTONESE_LABEL or label.strip().lower() in {
        "yue",
        "cantonese",
        "zh-yue",
        "zh_hk",
    }:
        return CANTONESE_INITIAL_PROMPT
    return None


def language_label(lang: Optional[str], display: Optional[str] = None) -> str:
    if display:
        return display
    if lang is None:
        return "auto"
    return lang


def collect_media_files(path: Path) -> list[Path]:
    """Return sorted list of supported media files from a file or folder tree."""
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path.resolve()]
        log.warning("Unsupported file type: %s", path)
        return []

    if path.is_dir():
        files: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
            files.extend(path.rglob(f"*{ext.upper()}"))
        unique = sorted({f.resolve() for f in files if f.is_file()})
        return unique

    log.error("Path does not exist: %s", path)
    return []


def transcript_path(media_path: Path) -> Path:
    """Return the Markdown output path for a media file inside OUTPUT_DIR."""
    return OUTPUT_DIR / f"{media_path.stem}.md"


def ensure_output_dir() -> Path:
    """Create the transcript output folder if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def should_skip(media_path: Path) -> bool:
    """Skip if a corresponding .md transcript already exists in OUTPUT_DIR."""
    md_path = transcript_path(media_path)
    if md_path.exists():
        log.info("Skipping (transcript exists): %s", md_path.name)
        return True
    return False


def prompt_language() -> tuple[Optional[str], str]:
    """
    Interactive language menu.

    Returns:
        (whisper_code, display_label)
    """
    labels = list(LANGUAGE_PRESETS.keys())
    print("\nSelect Audio Language Mode:")
    for i, label in enumerate(labels, start=1):
        default = " (Default)" if label == DEFAULT_LANGUAGE_LABEL else ""
        print(f"{i}. {label}{default}")
    print(f"{len(labels) + 1}. Enter Custom Language Code (e.g. ja, es, fr)")
    print()

    while True:
        choice = input(f"Choice [1-{len(labels) + 1}]: ").strip() or "1"
        try:
            idx = int(choice)
        except ValueError:
            print("  Invalid choice. Enter a number.")
            continue

        if 1 <= idx <= len(labels):
            label = labels[idx - 1]
            return LANGUAGE_PRESETS[label], label

        if idx == len(labels) + 1:
            code = input("Enter ISO language code: ").strip().lower()
            if code:
                return code, code
            print("  Language code cannot be empty. Try again.")
            continue

        print(f"  Invalid choice. Enter 1–{len(labels) + 1}.")


def load_model(model_name: Optional[str] = None):
    """
    Load WhisperModel on CPU (int8).

    Tries the requested model, then DEFAULT_MODEL, then FALLBACK_MODEL.
    """
    from faster_whisper import WhisperModel

    requested = (model_name or DEFAULT_MODEL).strip().lower()
    candidates: list[str] = []
    for name in (requested, DEFAULT_MODEL, FALLBACK_MODEL):
        if name and name not in candidates:
            candidates.append(name)

    last_error: Optional[Exception] = None
    for name in candidates:
        try:
            log.info(
                "Loading WhisperModel('%s', device=%s, compute_type=%s, cpu_threads=%d)...",
                name,
                DEVICE,
                COMPUTE_TYPE,
                CPU_THREADS,
            )
            model = WhisperModel(
                name,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS,
            )
            log.info("Model '%s' loaded successfully.", name)
            return model, name
        except Exception as exc:  # noqa: BLE001 — fallback path
            last_error = exc
            log.warning("Failed to load model '%s': %s", name, exc)

    raise RuntimeError(
        f"Unable to load any of: {', '.join(candidates)}. "
        "Check that faster-whisper is installed and the model can be downloaded."
    ) from last_error


ProgressCallback = Callable[[float, float, int, str], None]
# args: current_sec, duration_sec, segment_index, latest_text


def transcribe_file(
    model,
    media_path: Path,
    selected_lang: Optional[str] = None,
    *,
    display_label: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> TranscriptResult:
    """
    Transcribe a single media file and write Markdown to OUTPUT_DIR.

    progress_cb(current_sec, duration_sec, segment_index, latest_text) is called
    as each segment is produced so UIs can show live progress.
    """
    log.info(
        "Transcribing: %s (lang=%s)",
        media_path.name,
        language_label(selected_lang, display_label),
    )
    try:
        kwargs: dict = {
            "language": selected_lang,
            "beam_size": BEAM_SIZE,
            "vad_filter": VAD_FILTER,
            "vad_parameters": dict(
                min_silence_duration_ms=500,  # treat short pauses as speech continuity
                speech_pad_ms=200,
            ),
            "condition_on_previous_text": False,  # faster + less repetition loops
            "word_timestamps": False,  # segment-level only — much cheaper
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        segments_gen, info = model.transcribe(str(media_path), **kwargs)
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        detected = getattr(info, "language", "unknown") or "unknown"
        log.info(
            "Audio duration %s | detected=%s | vad=%s beam=%d",
            format_timestamp(duration).strip("[]") if duration else "?",
            detected,
            VAD_FILTER,
            BEAM_SIZE,
        )

        segments = []
        for i, seg in enumerate(segments_gen, start=1):
            segments.append(seg)
            if progress_cb is not None:
                text = (seg.text or "").strip()
                progress_cb(float(seg.end or 0.0), duration, i, text)
            if i == 1 or i % 10 == 0:
                log.info(
                    "  … segment %d @ %s",
                    i,
                    format_timestamp(float(seg.end or 0.0)),
                )

        out_path, markdown = write_markdown(
            media_path,
            segments,
            selected_lang,
            detected,
            display_label=display_label,
        )
        return TranscriptResult(
            media_path=media_path,
            success=True,
            markdown=markdown,
            out_path=out_path,
            detected_lang=detected,
        )
    except Exception as exc:  # noqa: BLE001 — per-file isolation
        log.error("Failed to transcribe %s: %s", media_path.name, exc)
        return TranscriptResult(
            media_path=media_path,
            success=False,
            error=str(exc),
        )


def transcribe_file_by_label(
    model,
    media_path: Path,
    language_label_name: str,
    *,
    progress_cb: Optional[ProgressCallback] = None,
) -> TranscriptResult:
    """Convenience: resolve a display label then transcribe."""
    code = language_code_for_label(language_label_name)
    prompt = initial_prompt_for_label(language_label_name)
    return transcribe_file(
        model,
        media_path,
        code,
        display_label=language_label_name,
        initial_prompt=prompt,
        progress_cb=progress_cb,
    )


def build_markdown(
    media_path: Path,
    segments: Iterable,
    selected_lang: Optional[str],
    detected_lang: str,
    display_label: Optional[str] = None,
) -> tuple[str, int]:
    """Build GFM transcript text (does not write to disk). Returns (markdown, segment_count)."""
    mode_label = language_label(selected_lang, display_label)

    lines = [
        f"# Audio Transcript: {media_path.stem}",
        f"**Language Mode:** {mode_label} (Detected: {detected_lang})",
        "",
        "---",
        "",
    ]

    segment_count = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(f"**{format_timestamp(seg.start)}**: {text}")
        lines.append("")
        segment_count += 1

    if segment_count == 0:
        lines.append("*No speech detected.*")
        lines.append("")

    return "\n".join(lines), segment_count


def write_markdown(
    media_path: Path,
    segments: Iterable,
    selected_lang: Optional[str],
    detected_lang: str,
    display_label: Optional[str] = None,
) -> tuple[Path, str]:
    """Write GFM transcript into OUTPUT_DIR. Returns (path, markdown)."""
    markdown, segment_count = build_markdown(
        media_path, segments, selected_lang, detected_lang, display_label
    )
    ensure_output_dir()
    out_path = transcript_path(media_path)
    out_path.write_text(markdown, encoding="utf-8")
    log.info("Wrote %d segment(s) -> %s", segment_count, out_path)
    return out_path, markdown


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe local audio/video files to Markdown using faster-whisper. "
            "Accepts file and/or folder paths (folders scanned recursively)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python transcribe.py --gui\n"
            "  python transcribe.py --gui meeting1.mp4 meeting2.mkv\n"
            "  python transcribe.py recording.mp4\n"
            "  python transcribe.py C:\\Recordings --lang en\n"
            "  python transcribe.py webinar.mkv --lang \"Chinese (Mandarin)\"\n"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[],
        help="Media file(s) and/or folder path(s).",
    )
    parser.add_argument(
        "--lang",
        "-l",
        default=None,
        metavar="CODE",
        help=(
            "Language: auto | en | zh | English | 'Chinese (Mandarin)' | "
            "'Chinese (Cantonese)' | custom ISO code. "
            "If omitted in CLI mode, an interactive menu is shown."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Tkinter preview interface.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        choices=list(MODEL_CHOICES),
        help=f"Whisper model size (default: {DEFAULT_MODEL}). smaller = faster on CPU.",
    )
    return parser.parse_args(argv)


def gather_from_paths(raw_paths: list[str]) -> list[Path]:
    """Collect supported media from one or more file/folder arguments."""
    found: list[Path] = []
    for raw in raw_paths:
        root = Path(raw.strip().strip('"'))
        if not root.exists():
            log.warning("Path not found (skipped): %s", root)
            continue
        found.extend(collect_media_files(root))
    # Preserve order while deduplicating
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def resolve_language(cli_lang: Optional[str]) -> tuple[Optional[str], str]:
    """Map CLI --lang to (whisper_code, display_label), or prompt."""
    if cli_lang is None:
        return prompt_language()

    raw = cli_lang.strip()
    # Exact preset match (case-sensitive labels) or case-insensitive
    for label in LANGUAGE_PRESETS:
        if raw.lower() == label.lower():
            return LANGUAGE_PRESETS[label], label

    normalized = raw.lower()
    if normalized in {"", "auto", "detect", "none"}:
        return None, "Auto-Detect"
    if normalized in {"en", "english"}:
        return "en", "English"
    if normalized in {"zh", "mandarin", "cmn"}:
        return "zh", "Chinese (Mandarin)"
    if normalized in {"yue", "cantonese", "zh-yue", "zh_hk"}:
        return "zh", CANTONESE_LABEL
    return normalized, normalized


def run_cli(args: argparse.Namespace) -> int:
    raw_paths = list(args.paths)
    if not raw_paths:
        typed = input("Enter media file or folder path: ").strip().strip('"')
        if typed:
            raw_paths = [typed]
    if not raw_paths:
        log.error("No path provided.")
        return 1

    media_files = gather_from_paths(raw_paths)
    if not media_files:
        log.error("No supported media files found.")
        log.info("Supported extensions: %s", ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        return 1

    pending = [f for f in media_files if not should_skip(f)]
    skipped = len(media_files) - len(pending)
    log.info(
        "Found %d media file(s); %d to process, %d skipped.",
        len(media_files),
        len(pending),
        skipped,
    )
    if not pending:
        log.info("Nothing to do.")
        return 0

    selected_lang, display = resolve_language(args.lang)
    prompt = initial_prompt_for_label(display)
    log.info("Language mode: %s", display)

    try:
        model, model_name = load_model(getattr(args, "model", None))
    except Exception as exc:  # noqa: BLE001
        log.error("Model load failed: %s", exc)
        return 1

    log.info(
        "Using model '%s' on %s (%s, threads=%d, beam=%d, vad=%s).",
        model_name,
        DEVICE,
        COMPUTE_TYPE,
        CPU_THREADS,
        BEAM_SIZE,
        VAD_FILTER,
    )

    ok = 0
    fail = 0
    for i, media in enumerate(pending, start=1):
        log.info("--- [%d/%d] %s ---", i, len(pending), media.name)
        result = transcribe_file(
            model,
            media,
            selected_lang,
            display_label=display,
            initial_prompt=prompt,
        )
        if result.success:
            ok += 1
        else:
            fail += 1

    log.info("Done. Succeeded: %d | Failed: %d | Skipped: %d", ok, fail, skipped)
    return 0 if fail == 0 else 2


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # GUI when --gui, or when launched with no media paths (preview-first flow).
    if args.gui or not args.paths:
        from gui import run_gui

        initial = gather_from_paths(args.paths) if args.paths else None
        run_gui(initial_paths=initial)
        return 0

    return run_cli(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
