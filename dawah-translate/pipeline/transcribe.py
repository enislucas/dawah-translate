"""
Transcribe audio using faster-whisper (CPU mode).

Usage:
    python pipeline/transcribe.py <job_id> [--model medium] [--language ar]

Reads audio.wav from the job directory and produces transcript_original.srt.

NOTE: First run downloads the Whisper model (~3 GB for large-v3, ~1.5 GB for medium).
      This is cached in ~/.cache/huggingface/ for subsequent runs.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JOBS_DIR, DEFAULT_WHISPER_MODEL, WHISPER_CPU_THREADS


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


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


def run_transcription(job_id: str, model_size: str = None, language: str = None) -> str:
    """
    Transcribe audio for the given job.

    Args:
        job_id: The job identifier
        model_size: Whisper model ("large-v3" or "medium")
        language: Source language ("ar", "en", or None for auto-detect)

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

    print(f"Job: {job_id}")
    print(f"Audio: {audio_path}")
    print(f"Duration: {duration_min:.1f} minutes")
    print(f"Model: {model_size}")
    print(f"Language: {language or 'auto-detect'}")
    print()

    # Update job status
    update_job_status(job_id, "transcribing", whisper_model=model_size)

    # Load faster-whisper model
    print(f"Loading Whisper model '{model_size}' (CPU, int8)...")
    print("  (First run downloads the model — this may take a few minutes)")
    load_start = time.time()

    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=WHISPER_CPU_THREADS,
    )

    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s")
    print()

    # Run transcription
    print("Transcribing...")
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
    print(f"  Detected language: {detected_lang} (confidence: {lang_prob:.1%})")

    # Collect segments with progress reporting
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        })

        # Print progress
        if duration > 0:
            progress = min(seg.end / duration * 100, 100)
            elapsed = time.time() - trans_start
            if seg.end > 0:
                est_total = elapsed / (seg.end / duration) if seg.end < duration else elapsed
                est_remaining = max(est_total - elapsed, 0)
                print(
                    f"\r  Progress: {progress:.1f}% | "
                    f"Elapsed: {elapsed:.0f}s | "
                    f"Remaining: ~{est_remaining:.0f}s | "
                    f"Segments: {len(segments)}",
                    end="", flush=True,
                )

    trans_time = time.time() - trans_start
    print(f"\n  Transcription complete in {trans_time:.1f}s ({len(segments)} segments)")
    print()

    # Write SRT file
    print(f"Writing SRT to {srt_path} ...")
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

    print(f"Done! {len(segments)} segments written.")
    return str(srt_path)


# ── Standalone entry point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio with faster-whisper")
    parser.add_argument("job_id", help="Job ID to transcribe")
    parser.add_argument("--model", default=None, choices=["large-v3", "medium"],
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--language", default=None, choices=["ar", "en", "auto"],
                        help="Source language (default: auto-detect)")
    args = parser.parse_args()

    print("=" * 60)
    print("DAWAH-TRANSLATE — Audio Transcription")
    print("=" * 60)

    try:
        srt_path = run_transcription(args.job_id, args.model, args.language)
        print()
        print(f"Next step: python pipeline/translate.py {args.job_id}")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
