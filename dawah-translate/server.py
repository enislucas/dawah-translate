"""
DAWAH-TRANSLATE — FastAPI Web Server

Serves the job submission form, review UI, and API endpoints.
Runs the download→transcribe→translate pipeline as background tasks.

Usage:
    python server.py
    → Open http://localhost:8000
"""

import json
import asyncio
import traceback
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import JOBS_DIR, STATIC_DIR, GLOSSARY_PATH, JOB_MAX_AGE_DAYS

app = FastAPI(title="Dawah-Translate", version="1.0")


# ── Models ────────────────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    url: str
    source_language: str = "auto"
    whisper_model: str = "large-v3"
    translation_engine: str = "local"  # "local" (NLLB-200) or "api" (Claude)
    claude_model: str = "sonnet"       # Only used if translation_engine == "api"


class SaveSrtRequest(BaseModel):
    segments: list[dict]


class RetranslateRequest(BaseModel):
    segment_index: int
    source_language: str = "auto"
    translation_engine: str = "local"
    claude_model: str = "sonnet"


# ── Helpers ───────────────────────────────────────────────────────────

def read_job_info(job_id: str) -> dict | None:
    """Read job_info.json for the given job. Returns None if not found."""
    info_path = JOBS_DIR / job_id / "job_info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_job_info(job_id: str, **updates):
    """Update fields in job_info.json."""
    info_path = JOBS_DIR / job_id / "job_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    info.update(updates)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def cleanup_old_jobs():
    """Delete job directories older than JOB_MAX_AGE_DAYS."""
    if not JOBS_DIR.exists():
        return
    cutoff = datetime.now() - timedelta(days=JOB_MAX_AGE_DAYS)
    removed = 0
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        info_path = job_dir / "job_info.json"
        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                created = datetime.fromisoformat(info.get("created_at", "2099-01-01"))
                if created < cutoff:
                    import shutil
                    shutil.rmtree(job_dir, ignore_errors=True)
                    removed += 1
            except Exception:
                pass
    if removed:
        print(f"Cleaned up {removed} old job(s)")


# ── Background pipeline ──────────────────────────────────────────────

def run_pipeline(job_id: str, url: str, source_language: str,
                 whisper_model: str, translation_engine: str = "local",
                 claude_model: str = "sonnet"):
    """
    Run the full pipeline: download → transcribe → translate.
    This runs in a background thread.
    """
    try:
        # Step 1: Download
        update_job_info(job_id, status="downloading", progress="Downloading video...")
        from pipeline.download import download_video, extract_audio, save_job_info as save_dl_info

        job_dir = JOBS_DIR / job_id
        metadata = download_video(url, job_dir)

        video_path = job_dir / "video.mp4"
        audio_path = job_dir / "audio.wav"
        extract_audio(video_path, audio_path)

        update_job_info(
            job_id,
            status="downloaded",
            video_title=metadata.get("title", "Unknown"),
            video_uploader=metadata.get("uploader", "Unknown"),
            video_duration=metadata.get("duration", 0),
            source_url=url,
            video_path=str(video_path),
            audio_path=str(audio_path),
            progress="Video downloaded, starting transcription...",
        )

        # Step 2: Transcribe
        update_job_info(job_id, status="transcribing", progress="Transcribing audio...")
        from pipeline.transcribe import run_transcription
        run_transcription(job_id, model_size=whisper_model, language=source_language)
        update_job_info(job_id, progress="Transcription complete, starting translation...")

        # Step 3: Translate
        if translation_engine == "api":
            update_job_info(job_id, status="translating", progress="Translating subtitles (Claude API)...")
            from pipeline.translate import run_translation
            run_translation(job_id, claude_model=claude_model)
        else:
            update_job_info(job_id, status="translating", progress="Translating subtitles (NLLB-200 local)...")
            from pipeline.translate_local import run_translation_local
            run_translation_local(job_id, language=source_language)

        update_job_info(job_id, status="ready_for_review", progress="Ready for review!")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Pipeline error for job {job_id}: {e}\n{tb}")
        update_job_info(
            job_id,
            status="error",
            error=str(e),
            error_traceback=tb,
            progress=f"Error: {e}",
        )


# ── Pages ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/review/{job_id}", response_class=HTMLResponse)
async def review_page(job_id: str):
    return FileResponse(str(STATIC_DIR / "review.html"))


@app.get("/glossary", response_class=HTMLResponse)
async def glossary_page():
    return FileResponse(str(STATIC_DIR / "glossary.html"))


@app.get("/about", response_class=HTMLResponse)
async def about_page():
    return FileResponse(str(STATIC_DIR / "about.html"))


@app.get("/architecture", response_class=HTMLResponse)
async def architecture_page():
    return FileResponse(str(STATIC_DIR / "architecture.html"))


# ── API Endpoints ─────────────────────────────────────────────────────

@app.post("/api/submit")
async def submit_job(req: SubmitRequest):
    """Submit a new translation job. Starts the pipeline in the background."""
    from pipeline.download import create_job_id

    job_id = create_job_id()
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create initial job_info.json
    job_info = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "source_url": req.url,
        "source_language": req.source_language,
        "whisper_model": req.whisper_model,
        "translation_engine": req.translation_engine,
        "claude_model": req.claude_model,
        "progress": "Job queued...",
    }
    with open(job_dir / "job_info.json", "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    # Start pipeline in background thread
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, run_pipeline, job_id, req.url,
        req.source_language, req.whisper_model, req.translation_engine, req.claude_model,
    )

    return {"job_id": job_id, "status": "queued"}


@app.post("/api/retry/{job_id}")
async def retry_job(job_id: str):
    """Retry a failed job from where it left off."""
    info = read_job_info(job_id)
    if not info:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if info.get("status") != "error":
        return JSONResponse({"error": "Job is not in error state"}, status_code=400)

    job_dir = JOBS_DIR / job_id
    has_video = (job_dir / "video.mp4").exists()
    has_audio = (job_dir / "audio.wav").exists()
    has_original_srt = (job_dir / "transcript_original.srt").exists()
    has_romanian_srt = (job_dir / "transcript_romanian.srt").exists()

    update_job_info(job_id, status="retrying", error="", error_traceback="",
                    progress="Retrying from last successful step...")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, _run_retry, job_id, info,
        has_video, has_audio, has_original_srt, has_romanian_srt,
    )

    return {"job_id": job_id, "status": "retrying"}


def _run_retry(job_id, info, has_video, has_audio, has_original_srt, has_romanian_srt):
    """Retry pipeline from the last successful step."""
    try:
        claude_model = info.get("claude_model", "sonnet")
        translation_engine = info.get("translation_engine", "local")
        source_language = info.get("source_language", "auto")
        whisper_model = info.get("whisper_model", "large-v3")
        url = info.get("source_url", "")

        if not has_video or not has_audio:
            # Need to re-download
            run_pipeline(job_id, url, source_language, whisper_model, translation_engine, claude_model)
            return

        if not has_original_srt:
            # Need to transcribe
            update_job_info(job_id, status="transcribing", progress="Transcribing audio...")
            from pipeline.transcribe import run_transcription
            run_transcription(job_id, model_size=whisper_model, language=source_language)

        if not has_romanian_srt:
            # Need to translate
            if translation_engine == "api":
                update_job_info(job_id, status="translating", progress="Translating (Claude API)...")
                from pipeline.translate import run_translation
                run_translation(job_id, claude_model=claude_model)
            else:
                update_job_info(job_id, status="translating", progress="Translating (NLLB-200 local)...")
                from pipeline.translate_local import run_translation_local
                run_translation_local(job_id, language=source_language)

        update_job_info(job_id, status="ready_for_review", progress="Ready for review!")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Retry error for job {job_id}: {e}\n{tb}")
        update_job_info(job_id, status="error", error=str(e), error_traceback=tb,
                        progress=f"Error: {e}")


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    """Get current job status and progress."""
    info = read_job_info(job_id)
    if not info:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return info


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs sorted by creation time (newest first)."""
    jobs = []
    if JOBS_DIR.exists():
        for job_dir in sorted(JOBS_DIR.iterdir(), reverse=True):
            if job_dir.is_dir():
                info = read_job_info(job_dir.name)
                if info:
                    jobs.append({
                        "job_id": info.get("job_id"),
                        "status": info.get("status"),
                        "video_title": info.get("video_title", ""),
                        "created_at": info.get("created_at", ""),
                        "progress": info.get("progress", ""),
                    })
    return jobs


@app.get("/api/job/{job_id}/video")
async def stream_video(job_id: str, request: Request):
    """Stream the video file with range request support."""
    video_path = JOBS_DIR / job_id / "video.mp4"
    if not video_path.exists():
        return JSONResponse({"error": "Video not found"}, status_code=404)

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse range: bytes=start-end
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1
        content_length = end - start + 1

        def iter_file():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Type": "video/mp4",
            },
        )

    return FileResponse(str(video_path), media_type="video/mp4")


@app.get("/api/job/{job_id}/srt")
async def get_srt(job_id: str):
    """Get the Romanian SRT as a JSON array of segments."""
    from pipeline.subtitle import parse_srt, check_reading_speed

    # Try Romanian first, fall back to original
    srt_path = JOBS_DIR / job_id / "transcript_romanian.srt"
    if not srt_path.exists():
        srt_path = JOBS_DIR / job_id / "transcript_original.srt"
    if not srt_path.exists():
        return JSONResponse({"error": "SRT not found"}, status_code=404)

    segments = parse_srt(srt_path)

    # Also load original for side-by-side view
    original_path = JOBS_DIR / job_id / "transcript_original.srt"
    original_segments = {}
    if original_path.exists():
        for seg in parse_srt(original_path):
            original_segments[seg["index"]] = seg["text"]

    # Enrich with reading speed and original text
    result = []
    for seg in segments:
        speed = check_reading_speed(seg)
        result.append({
            "index": seg["index"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "original_text": original_segments.get(seg["index"], ""),
            "reading_speed": round(speed, 1),
            "too_fast": speed > 21,
        })

    return result


@app.post("/api/job/{job_id}/srt")
async def save_srt(job_id: str, req: SaveSrtRequest):
    """Save edited segments back to the Romanian SRT file."""
    from pipeline.subtitle import write_srt, normalize_romanian

    srt_path = JOBS_DIR / job_id / "transcript_romanian.srt"

    # Normalize Romanian diacritics in all segments
    segments = []
    for seg in req.segments:
        segments.append({
            "index": seg["index"],
            "start": seg["start"],
            "end": seg["end"],
            "text": normalize_romanian(seg["text"]),
        })

    write_srt(segments, srt_path)
    return {"status": "saved", "segments": len(segments)}


@app.post("/api/job/{job_id}/retranslate")
async def retranslate_segment(job_id: str, req: RetranslateRequest):
    """Re-translate a single segment (local or API)."""
    from pipeline.subtitle import parse_srt, check_reading_speed

    srt_path = JOBS_DIR / job_id / "transcript_original.srt"
    if not srt_path.exists():
        return JSONResponse({"error": "Original SRT not found"}, status_code=404)

    segments = parse_srt(srt_path)

    # Find the segment by index
    segment = None
    for seg in segments:
        if seg["index"] == req.segment_index:
            segment = seg
            break

    if not segment:
        return JSONResponse({"error": f"Segment {req.segment_index} not found"}, status_code=404)

    try:
        if req.translation_engine == "api":
            from pipeline.translate import translate_single_segment
            translated_text = await asyncio.to_thread(
                translate_single_segment, segment, req.source_language
            )
        else:
            from pipeline.translate_local import translate_single_segment_local
            translated_text = await asyncio.to_thread(
                translate_single_segment_local, segment, req.source_language
            )
        speed = len(translated_text.replace('\n', '')) / max(segment["end"] - segment["start"], 0.1)
        return {
            "index": segment["index"],
            "text": translated_text,
            "reading_speed": round(speed, 1),
            "too_fast": speed > 21,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/job/{job_id}/burn")
async def burn_subtitles(job_id: str):
    """Trigger FFmpeg subtitle burning."""
    info = read_job_info(job_id)
    if not info:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    update_job_info(job_id, status="burning", progress="Burning subtitles into video...")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_burn, job_id)

    return {"status": "burning"}


def _run_burn(job_id: str):
    """Run FFmpeg burn in background thread."""
    try:
        from pipeline.burn import run_burn
        run_burn(job_id)
        update_job_info(job_id, status="complete", progress="Done! Video ready for download.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Burn error for job {job_id}: {e}\n{tb}")
        update_job_info(job_id, status="error", error=str(e), progress=f"Burn error: {e}")


@app.get("/api/glossary")
async def get_glossary():
    """Get the glossary as JSON."""
    if not GLOSSARY_PATH.exists():
        return {}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/glossary")
async def save_glossary(request: Request):
    """Save the full glossary JSON."""
    data = await request.json()
    with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return {"status": "saved", "terms": len(data)}


@app.get("/api/job/{job_id}/download")
async def download_final(job_id: str):
    """Download the final burned MP4."""
    final_path = JOBS_DIR / job_id / "final.mp4"
    if not final_path.exists():
        return JSONResponse({"error": "Final video not ready"}, status_code=404)

    info = read_job_info(job_id)
    title = info.get("video_title", "video") if info else "video"
    # Sanitize filename
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:80]
    filename = f"{safe_title}_RO.mp4"

    return FileResponse(str(final_path), media_type="video/mp4", filename=filename)


# ── Lifespan ──────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    print("Dawah-Translate server starting...")
    cleanup_old_jobs()
    print("Server ready at http://localhost:8000")
    yield

app.router.lifespan_context = lifespan

# ── Static files (mounted last so explicit routes take priority) ─────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
