"""
Transcribe audio using faster-whisper (CPU mode).

Usage:
    python pipeline/transcribe.py <job_id> [--model medium] [--language ar]

Reads audio.wav from the job directory and produces transcript_original.srt.

NOTE: First run downloads the Whisper model (~3 GB for large-v3, ~1.5 GB for medium).
      This is cached in ~/.cache/huggingface/ for subsequent runs.
"""

import sys
import json
import logging
import time
import argparse
from pathlib import Path

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JOBS_DIR, DEFAULT_WHISPER_MODEL, WHISPER_CPU_THREADS

logger = logging.getLogger("dawah.transcribe")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_duration(seconds: float) -> str:
    """Format seconds as M:SS for display."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def write_srt(segments: list, output_path: Path) -> None:
    """Write segments to an SRT file (UTF-8 with BOM for Romanian diacritics)."""
    with open(output_path, "w", encoding="utf-8-sig") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def load_job(job_id: str) -> dict:
    """Load job_info.json for the given job ID."""
    job_dir = JOBS_DIR / job_id
    info_path = job_dir / "job_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Job not found: {job_id} (no job_info.json in {job_dir})")
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_job_status(job_id: str, status: str, **extra):
    """Update job_info.json with new status and optional fields."""
    job_dir = JOBS_DIR / job_id
    info_path = job_dir / "job_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    info["status"] = status
    info.update(extra)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def run_transcription(job_id: str, model_size: str = None, language: str = None,
                      progress_cb=None) -> str:
    """
    Transcribe audio for the given job.

    Args:
        job_id: The job identifier
        model_size: Whisper model ("large-v3" or "medium")
        language: Source language ("ar", "en", or None for auto-detect)
        progress_cb: Optional callback(step, progress_pct, message) for SSE progress

    Returns:
        Path to the output SRT file.
    """
    if model_size is None:
        model_size = DEFAULT_WHISPER_MODEL

    # Load job info
    job_info = load_job(job_id)
    job_dir = JOBS_DIR / job_id
    audio_path = job_dir / "audio.wav"
    srt_path = job_dir / "transcript_original.srt"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = job_info.get("video_duration", 0)
    duration_min = duration / 60 if duration else 0

    logger.info("Job: %s | Audio: %s | Duration: %.1f min | Model: %s | Language: %s",
                job_id, audio_path, duration_min, model_size, language or "auto")

    # Update job status
    update_job_status(job_id, "transcribing", whisper_model=model_size)

    # Load faster-whisper model
    if progress_cb:
        progress_cb("transcribe", 0, "Loading Whisper model...")
    logger.info("Loading Whisper model '%s' (CPU, int8)...", model_size)
    load_start = time.time()

    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=WHISPER_CPU_THREADS,
    )

    load_time = time.time() - load_start
    logger.info("Model loaded in %.1fs", load_time)

    if progress_cb:
        progress_cb("transcribe", 5, f"Whisper model loaded in {load_time:.0f}s. Transcribing...")

    # Run transcription
    trans_start = time.time()

    # Language parameter: None means auto-detect
    lang_param = language if language and language != "auto" else None

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=lang_param,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    detected_lang = info.language
    lang_prob = info.language_probability
    logger.info("Detected language: %s (confidence: %.1f%%)", detected_lang, lang_prob * 100)

    # Collect segments with progress reporting
    segments = []
    last_progress_time = 0  # throttle progress updates to max 1 per second
    for seg in segments_iter:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        })

        # Report progress via callback (throttled)
        now = time.time()
        if duration > 0 and (now - last_progress_time) >= 1.0:
            last_progress_time = now
            pct = min(seg.end / duration, 1.0)
            progress_int = 5 + int(pct * 90)  # 5-95% range
            elapsed_str = _format_duration(seg.end)
            total_str = _format_duration(duration)
            msg = f"Transcribing... {elapsed_str} / {total_str}"

            if progress_cb:
                progress_cb("transcribe", progress_int, msg)

    trans_time = time.time() - trans_start
    logger.info("Transcription complete in %.1fs (%d segments)", trans_time, len(segments))

    if progress_cb:
        progress_cb("transcribe", 98, f"Writing SRT ({len(segments)} segments)...")

    # Write SRT file
    write_srt(segments, srt_path)

    # Update job info
    update_job_status(
        job_id, "transcribed",
        detected_language=detected_lang,
        language_confidence=lang_prob,
        segment_count=len(segments),
        transcription_time=trans_time,
        srt_original_path=str(srt_path),
    )

    if progress_cb:
        progress_cb("transcribe", 100, f"Transcription complete! {len(segments)} segments in {trans_time:.0f}s")

    logger.info("Done! %d segments written to %s", len(segments), srt_path)
    return str(srt_path)


# ── Standalone entry point ───────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[Transcribe] %(message)s")

    parser = argparse.ArgumentParser(description="Transcribe audio with faster-whisper")
    parser.add_argument("job_id", help="Job ID to transcribe")
    parser.add_argument("--model", default=None, choices=["large-v3", "medium"],
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--language", default=None, choices=["ar", "en", "auto"],
                        help="Source language (default: auto-detect)")
    args = parser.parse_args()

    print("=" * 60)
    print("DAWAH-TRANSLATE -- Audio Transcription")
    print("=" * 60)

    def _cli_progress(step, pct, msg):
        print(f"  [{step}] {pct}% | {msg}")

    try:
        srt_path = run_transcription(args.job_id, args.model, args.language,
                                     progress_cb=_cli_progress)
        print()
        print(f"Next step: python pipeline/translate.py {args.job_id}")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
