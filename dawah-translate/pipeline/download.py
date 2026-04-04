"""
Download YouTube videos and extract audio for transcription.

Usage:
    python pipeline/download.py <youtube_url>

Creates a job directory in jobs/ with:
    - video.mp4 (max 1080p)
    - audio.wav (16kHz mono PCM16 for Whisper)
    - job_info.json (metadata)
"""

import sys
import os
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JOBS_DIR


def create_job_id() -> str:
    """Generate a short unique job ID using timestamp + random suffix."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{timestamp}_{suffix}"


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: ffprobe failed — {result.stderr.strip()}")
        return 0.0
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))


def download_video(url: str, job_dir: Path) -> dict:
    """
    Download video from YouTube using yt-dlp.

    Returns dict with video metadata (title, duration, etc.).
    """
    video_path = job_dir / "video.mp4"

    print(f"  Downloading video to {video_path} ...")

    # yt-dlp options: max 1080p, force MP4 container
    import yt_dlp

    ydl_opts = {
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "outtmpl": str(video_path),
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "progress_hooks": [_progress_hook],
        "extractor_args": {"youtube": {"remote_components": ["ejs:github"]}},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # yt-dlp may add extra extension — find the actual file
    if not video_path.exists():
        # Check for common variations
        for f in job_dir.glob("video.*"):
            if f.suffix in (".mp4", ".mkv", ".webm"):
                video_path = f
                break

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found after download in {job_dir}")

    # Get accurate duration from the downloaded file
    duration = get_video_duration(video_path)

    return {
        "title": info.get("title", "Unknown"),
        "uploader": info.get("uploader", "Unknown"),
        "duration": duration,
        "url": url,
        "video_path": str(video_path),
    }


def _progress_hook(d):
    """Print download progress."""
    if d["status"] == "downloading":
        pct = d.get("_percent_str", "?%")
        speed = d.get("_speed_str", "?")
        eta = d.get("_eta_str", "?")
        print(f"\r  Download: {pct} at {speed}, ETA: {eta}   ", end="", flush=True)
    elif d["status"] == "finished":
        print(f"\n  Download complete. Merging formats...")


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """
    Extract audio from video as WAV (16kHz, mono, PCM 16-bit).

    This format is optimal for Whisper transcription.
    """
    print(f"  Extracting audio to {audio_path} ...")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16 kHz sample rate
        "-ac", "1",               # Mono
        str(audio_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")

    size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"  Audio extracted: {size_mb:.1f} MB")


def save_job_info(job_dir: Path, job_id: str, metadata: dict, **extra) -> dict:
    """Save job metadata to job_info.json."""
    job_info = {
        "job_id": job_id,
        "status": "downloaded",
        "created_at": datetime.now().isoformat(),
        "video_title": metadata.get("title", "Unknown"),
        "video_uploader": metadata.get("uploader", "Unknown"),
        "video_duration": metadata.get("duration", 0),
        "source_url": metadata.get("url", ""),
        "video_path": str(job_dir / "video.mp4"),
        "audio_path": str(job_dir / "audio.wav"),
        **extra,
    }

    info_path = job_dir / "job_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    return job_info


def run_download(url: str, source_language: str = "auto") -> dict:
    """
    Full download pipeline: create job → download video → extract audio.

    Returns the job_info dict.
    """
    # Create job directory
    job_id = create_job_id()
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"Job ID: {job_id}")
    print(f"Job directory: {job_dir}")
    print()

    # Step 1: Download video
    print("[1/2] Downloading video...")
    metadata = download_video(url, job_dir)
    duration_min = metadata["duration"] / 60
    print(f"  Title: {metadata['title']}")
    print(f"  Duration: {duration_min:.1f} minutes")
    print()

    # Step 2: Extract audio
    print("[2/2] Extracting audio for transcription...")
    video_path = job_dir / "video.mp4"
    audio_path = job_dir / "audio.wav"
    extract_audio(video_path, audio_path)
    print()

    # Save job info
    job_info = save_job_info(
        job_dir, job_id, metadata,
        source_language=source_language,
    )

    print(f"Done! Job '{job_id}' is ready for transcription.")
    print(f"  Video: {job_info['video_path']}")
    print(f"  Audio: {job_info['audio_path']}")

    return job_info


# ── Standalone entry point ────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/download.py <youtube_url> [language]")
        print("  language: ar, en, or auto (default: auto)")
        sys.exit(1)

    url = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "auto"

    print("=" * 60)
    print("DAWAH-TRANSLATE — Video Download")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Language: {lang}")
    print()

    try:
        info = run_download(url, source_language=lang)
        print()
        print("Next step: python pipeline/transcribe.py", info["job_id"])
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
