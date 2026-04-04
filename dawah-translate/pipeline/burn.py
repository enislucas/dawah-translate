"""
Burn subtitles into video using FFmpeg with ASS format.

Usage:
    python pipeline/burn.py <job_id>

Reads transcript_romanian.srt, converts to ASS, burns into video.mp4 → final.mp4.
"""

import sys
import json
import subprocess
import re
import time
from pathlib import Path

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JOBS_DIR, FONTS_DIR, FFMPEG_PRESET, FFMPEG_CRF


def ensure_font():
    """Check that Noto Sans Bold font is available in fonts/."""
    font_path = FONTS_DIR / "NotoSans-Bold.ttf"
    if font_path.exists():
        return font_path

    print("  Font not found. Attempting to download Noto Sans Bold...")
    try:
        import urllib.request
        url = "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans-Bold.ttf"
        urllib.request.urlretrieve(url, str(font_path))
        print(f"  Downloaded to {font_path}")
        return font_path
    except Exception as e:
        print(f"  Auto-download failed: {e}")
        print(f"  Please manually download Noto Sans Bold from Google Fonts")
        print(f"  and place NotoSans-Bold.ttf in: {FONTS_DIR}")
        raise FileNotFoundError(f"Font file not found: {font_path}")


def run_burn(job_id: str) -> str:
    """
    Burn Romanian subtitles into the video.

    Returns path to final.mp4.
    """
    from pipeline.subtitle import srt_to_ass

    job_dir = JOBS_DIR / job_id
    info_path = job_dir / "job_info.json"

    # Load job info
    with open(info_path, "r", encoding="utf-8") as f:
        job_info = json.load(f)

    video_path = job_dir / "video.mp4"
    srt_path = job_dir / "transcript_romanian.srt"
    ass_path = job_dir / "subtitles.ass"
    final_path = job_dir / "final.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"Romanian SRT not found: {srt_path}")

    print(f"Job: {job_id}")
    print(f"Video: {video_path}")
    print()

    # Ensure font is available
    print("[1/3] Checking font...")
    ensure_font()
    print("  Font ready")
    print()

    # Convert SRT → ASS
    print("[2/3] Converting SRT to ASS format...")
    srt_to_ass(srt_path, ass_path, FONTS_DIR)
    print(f"  ASS file: {ass_path}")
    print()

    # Burn subtitles with FFmpeg
    print("[3/3] Burning subtitles into video...")
    print(f"  Output: {final_path}")
    print(f"  Preset: {FFMPEG_PRESET}, CRF: {FFMPEG_CRF}")
    print()

    # On Windows, FFmpeg filter path parsing is fragile with colons and long paths.
    # Strategy: copy the ASS + font into the job dir, cd there, use relative paths.
    import shutil
    import os

    # Copy font into job dir for relative path access
    font_src = FONTS_DIR / "NotoSans-Bold.ttf"
    font_dst = job_dir / "NotoSans-Bold.ttf"
    if font_src.exists() and not font_dst.exists():
        shutil.copy2(font_src, font_dst)

    # Build command using relative paths by running from the job directory
    # This avoids all Windows path escaping issues
    cmd = [
        "ffmpeg", "-y",
        "-i", "video.mp4",
        "-vf", "subtitles=subtitles.ass",
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", str(FFMPEG_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        "final.mp4",
    ]

    start_time = time.time()

    # Get video duration for progress reporting
    duration = job_info.get("video_duration", 0)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=str(job_dir),  # Run from job dir so relative paths work
    )

    # Parse FFmpeg progress from stderr
    # FFmpeg outputs lines like: frame= 1234 fps= 30 ... time=00:01:23.45 ...
    stderr_output = []
    for line in process.stderr:
        stderr_output.append(line)
        time_match = re.search(r'time=(\d+):(\d+):(\d+)\.(\d+)', line)
        if time_match and duration > 0:
            h, m, s, cs = time_match.groups()
            current = int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100
            progress = min(current / duration * 100, 100)
            elapsed = time.time() - start_time
            if current > 0:
                est_total = elapsed / (current / duration)
                remaining = max(est_total - elapsed, 0)
                print(
                    f"\r  Progress: {progress:.1f}% | "
                    f"Elapsed: {elapsed:.0f}s | "
                    f"Remaining: ~{remaining:.0f}s",
                    end="", flush=True,
                )

    process.wait()
    total_time = time.time() - start_time

    if process.returncode != 0:
        error_text = ''.join(stderr_output[-20:])  # Last 20 lines
        raise RuntimeError(f"FFmpeg failed (exit code {process.returncode}):\n{error_text}")

    print(f"\n  Burn complete in {total_time:.1f}s")

    # Verify output
    if not final_path.exists():
        raise RuntimeError("FFmpeg completed but output file not found")

    final_size = final_path.stat().st_size / (1024 * 1024)
    print(f"  Final video: {final_size:.1f} MB")

    # Update job info
    job_info["status"] = "complete"
    job_info["final_path"] = str(final_path)
    job_info["final_size_mb"] = round(final_size, 1)
    job_info["burn_time"] = round(total_time, 1)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Final video: {final_path}")
    return str(final_path)


# ── Standalone entry point ────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/burn.py <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]

    print("=" * 60)
    print("DAWAH-TRANSLATE — Subtitle Burning")
    print("=" * 60)

    try:
        final = run_burn(job_id)
        print(f"\nPlay the final video: {final}")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
