"""
Export translated subtitles in multiple formats.

Supported formats:
    - SRT (SubRip, standard)
    - VTT (WebVTT, for web players)
    - TXT (plain text transcript)
    - Bilingual SRT (source + translation side-by-side)
    - ASS (Advanced SubStation Alpha, styled)

Usage:
    from pipeline.export import export_job
    paths = export_job(job_id, formats=["srt", "vtt", "txt", "bilingual"])
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JOBS_DIR
from pipeline.subtitle import parse_srt, _format_timestamp, srt_to_ass


def _format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export_srt(segments: list, output_path: Path) -> Path:
    """Export as standard SRT."""
    with open(output_path, "w", encoding="utf-8-sig") as f:
        for i, seg in enumerate(segments, 1):
            start = _format_timestamp(seg["start"])
            end = _format_timestamp(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")
    return output_path


def export_vtt(segments: list, output_path: Path) -> Path:
    """Export as WebVTT."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            start = _format_vtt_timestamp(seg["start"])
            end = _format_vtt_timestamp(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")
    return output_path


def export_txt(segments: list, output_path: Path, include_timestamps: bool = False) -> Path:
    """Export as plain text transcript."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            if include_timestamps:
                minutes = int(seg["start"] // 60)
                secs = int(seg["start"] % 60)
                f.write(f"[{minutes}:{secs:02d}] ")
            f.write(seg["text"].strip().replace("\n", " ") + "\n")
    return output_path


def export_bilingual(source_segments: list, translated_segments: list,
                     output_path: Path) -> Path:
    """Export bilingual SRT with source on top line and translation below."""
    trans_lookup = {seg["index"]: seg["text"] for seg in translated_segments}

    with open(output_path, "w", encoding="utf-8-sig") as f:
        for i, seg in enumerate(source_segments, 1):
            start = _format_timestamp(seg["start"])
            end = _format_timestamp(seg["end"])
            source_text = seg["text"].strip()
            trans_text = trans_lookup.get(seg["index"], "").strip()
            # Source on first line(s), translation below with a blank line
            f.write(f"{i}\n{start} --> {end}\n{source_text}\n{trans_text}\n\n")
    return output_path


def export_job(job_id: str, formats: list = None) -> dict:
    """
    Export a job's subtitles in the requested formats.

    Args:
        job_id: Job identifier
        formats: List of format strings. Default: ["srt", "vtt", "txt", "bilingual"]

    Returns:
        Dict mapping format name to output file path.
    """
    if formats is None:
        formats = ["srt", "vtt", "txt", "bilingual"]

    job_dir = JOBS_DIR / job_id
    export_dir = job_dir / "exports"
    export_dir.mkdir(exist_ok=True)

    # Load translated SRT
    ro_srt = job_dir / "transcript_romanian.srt"
    if not ro_srt.exists():
        raise FileNotFoundError(f"Romanian SRT not found: {ro_srt}")
    translated = parse_srt(ro_srt)

    # Load original SRT (for bilingual)
    orig_srt = job_dir / "transcript_original.srt"
    original = parse_srt(orig_srt) if orig_srt.exists() else []

    # Load job info for filename
    info_path = job_dir / "job_info.json"
    title = "subtitles"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        raw_title = info.get("video_title", "subtitles")
        title = "".join(c for c in raw_title if c.isalnum() or c in " -_").strip()[:60]

    results = {}

    if "srt" in formats:
        path = export_srt(translated, export_dir / f"{title}_RO.srt")
        results["srt"] = str(path)

    if "vtt" in formats:
        path = export_vtt(translated, export_dir / f"{title}_RO.vtt")
        results["vtt"] = str(path)

    if "txt" in formats:
        path = export_txt(translated, export_dir / f"{title}_RO.txt", include_timestamps=True)
        results["txt"] = str(path)

    if "bilingual" in formats and original:
        path = export_bilingual(original, translated, export_dir / f"{title}_bilingual.srt")
        results["bilingual"] = str(path)

    if "ass" in formats:
        ass_path = export_dir / f"{title}_RO.ass"
        srt_to_ass(ro_srt, ass_path)
        results["ass"] = str(ass_path)

    return results
