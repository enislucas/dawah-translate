"""
DAWAH-TRANSLATE -- FastAPI Web Server (V3)

Serves the job submission form, review UI, and API endpoints.
Runs the download->transcribe->translate pipeline as background tasks.
SSE progress streaming for all long-running operations.

Usage:
    python server.py
    -> Open http://localhost:8000
"""

import json
import asyncio
import logging
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import JOBS_DIR, STATIC_DIR, GLOSSARY_PATH, JOB_MAX_AGE_DAYS

logger = logging.getLogger("dawah.server")

app = FastAPI(title="Dawah-Translate V3", version="3.0")


# ── Models ────────────────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    """Initial submission — only download/transcribe params.
    Model choice and subtitle mode are picked later on the config screen."""
    url: str
    source_language: str = "auto"
    whisper_model: str = "large-v3"


class StartTranslationRequest(BaseModel):
    """Sent from the config screen after the user picks model + mode."""
    claude_model: str = "sonnet"
    subtitle_mode: str = "black_bar"  # "black_bar" or "overlay"


class SaveSrtRequest(BaseModel):
    segments: list[dict]


class RetranslateRequest(BaseModel):
    segment_index: int
    source_language: str = "auto"
    claude_model: str = "sonnet"


class UpdateSegmentRequest(BaseModel):
    """Update a single segment's text and/or timestamps."""
    index: int
    text: str | None = None
    start: float | None = None
    end: float | None = None


class FeedbackRequest(BaseModel):
    """Per-segment feedback from the reviewer."""
    segment_index: int
    rating: str  # "good", "bad", "note"
    note: str = ""


class TitleUpdateRequest(BaseModel):
    """Update the Romanian title for final export."""
    title_romanian: str


# ── Progress tracking (SSE) ──────────────────────────────────────────
# Per-job progress state, updated by pipeline threads, read by SSE endpoint.

_progress_store: dict[str, dict] = {}
_progress_lock = threading.Lock()


def _set_progress(job_id: str, step: str, progress: int, message: str):
    """Thread-safe progress update. Called from pipeline background threads."""
    with _progress_lock:
        _progress_store[job_id] = {
            "step": step,
            "progress": progress,
            "message": message,
        }


def _get_progress(job_id: str) -> dict:
    """Thread-safe progress read."""
    with _progress_lock:
        return _progress_store.get(job_id, {"step": "unknown", "progress": 0, "message": ""})


def _make_progress_callback(job_id: str):
    """Create a progress_cb(step, pct, message) bound to a specific job."""
    def cb(step: str, progress: int, message: str):
        _set_progress(job_id, step, progress, message)
        # Also update job_info.json progress field for polling fallback
        try:
            update_job_info(job_id, progress=message)
        except Exception:
            pass
    return cb


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
        logger.info("Cleaned up %d old job(s)", removed)


# ── Background pipeline ──────────────────────────────────────────────

def run_pipeline(job_id: str, url: str, source_language: str,
                 whisper_model: str):
    """
    Run the FIRST half of the pipeline: download → transcribe.
    Then PAUSES (status = awaiting_config) so the user can pick model +
    subtitle mode on the config screen. The second half runs when the
    user POSTs to /api/job/{job_id}/start-translation.
    """
    progress_cb = _make_progress_callback(job_id)

    try:
        # Step 1: Download
        progress_cb("download", 0, "Downloading video...")
        update_job_info(job_id, status="downloading")
        from pipeline.download import download_video, extract_audio

        job_dir = JOBS_DIR / job_id
        metadata = download_video(url, job_dir)

        video_path = job_dir / "video.mp4"
        audio_path = job_dir / "audio.wav"

        progress_cb("download", 80, "Extracting audio...")
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
        )
        progress_cb("download", 100, "Download complete.")

        # Step 2: Transcribe
        update_job_info(job_id, status="transcribing")
        from pipeline.transcribe import run_transcription
        run_transcription(job_id, model_size=whisper_model, language=source_language,
                          progress_cb=progress_cb)

        # PAUSE — wait for user config (model choice + subtitle mode)
        update_job_info(
            job_id,
            status="awaiting_config",
            progress="Transcription complete. Awaiting translation config...",
        )
        progress_cb("config", 100, "Ready for config — choose model and subtitle mode")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Pipeline error for job %s: %s\n%s", job_id, e, tb)
        update_job_info(
            job_id,
            status="error",
            error=str(e),
            error_traceback=tb,
            progress=f"Error: {e}",
        )
        _set_progress(job_id, "error", 0, f"Error: {e}")


def run_translation_phase(job_id: str, claude_model: str):
    """
    Run the SECOND half of the pipeline: translate → validate.
    Triggered by POST /api/job/{job_id}/start-translation after the user
    confirms model choice on the config screen.
    """
    progress_cb = _make_progress_callback(job_id)

    try:
        update_job_info(job_id, status="translating", claude_model=claude_model)
        from pipeline.translate import run_translation
        run_translation(job_id, claude_model=claude_model, progress_cb=progress_cb)

        # Quality validation
        progress_cb("validate", 0, "Running quality checks...")
        update_job_info(job_id, status="validating")
        from pipeline.quality import validate_job
        quality_report = validate_job(job_id)
        update_job_info(
            job_id,
            status="ready_for_review",
            progress="Ready for review!",
            quality_score=quality_report["average_score"],
            quality_critical=quality_report["critical_count"],
            quality_warnings=quality_report["warning_count"],
        )
        progress_cb("complete", 100, "Ready for review!")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Translation-phase error for job %s: %s\n%s", job_id, e, tb)
        update_job_info(
            job_id,
            status="error",
            error=str(e),
            error_traceback=tb,
            progress=f"Error: {e}",
        )
        _set_progress(job_id, "error", 0, f"Error: {e}")


# ── Pages ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/review/{job_id}", response_class=HTMLResponse)
async def review_page(job_id: str):
    return FileResponse(str(STATIC_DIR / "review.html"))


@app.get("/config/{job_id}", response_class=HTMLResponse)
async def config_page(job_id: str):
    return FileResponse(str(STATIC_DIR / "config.html"))


@app.get("/glossary", response_class=HTMLResponse)
async def glossary_page():
    return FileResponse(str(STATIC_DIR / "glossary.html"))


@app.get("/about", response_class=HTMLResponse)
async def about_page():
    return FileResponse(str(STATIC_DIR / "about.html"))


@app.get("/architecture", response_class=HTMLResponse)
async def architecture_page():
    return FileResponse(str(STATIC_DIR / "architecture.html"))


# ── API Endpoints ────────────────────────────────────────────────────

@app.post("/api/submit")
async def submit_job(req: SubmitRequest):
    """Submit a new translation job. Starts the pipeline in the background."""
    from pipeline.download import create_job_id

    job_id = create_job_id()
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create initial job_info.json — model + subtitle_mode set later
    job_info = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "source_url": req.url,
        "source_language": req.source_language,
        "whisper_model": req.whisper_model,
        "translation_engine": "cloud",  # V3: Claude API only
        "progress": "Job queued...",
    }
    with open(job_dir / "job_info.json", "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    # Start pipeline in background thread (download + transcribe only)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, run_pipeline, job_id, req.url,
        req.source_language, req.whisper_model,
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/job/{job_id}/estimate")
async def get_estimate(job_id: str):
    """Cost estimate for a transcribed job. Used by the config screen."""
    from pipeline.estimate import estimate_job
    try:
        return await asyncio.to_thread(estimate_job, job_id)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/job/{job_id}/start-translation")
async def start_translation(job_id: str, req: StartTranslationRequest):
    """Resume the pipeline after the user picks model + subtitle mode."""
    info = read_job_info(job_id)
    if not info:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if info.get("status") != "awaiting_config":
        return JSONResponse(
            {"error": f"Job is not awaiting config (status={info.get('status')})"},
            status_code=400,
        )

    update_job_info(
        job_id,
        claude_model=req.claude_model,
        subtitle_mode=req.subtitle_mode,
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_translation_phase, job_id, req.claude_model)

    return {"job_id": job_id, "status": "translating"}


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
    progress_cb = _make_progress_callback(job_id)

    try:
        claude_model = info.get("claude_model", "sonnet")
        source_language = info.get("source_language", "auto")
        whisper_model = info.get("whisper_model", "large-v3")
        url = info.get("source_url", "")

        if not has_video or not has_audio:
            # Restart from download — this will end at awaiting_config
            run_pipeline(job_id, url, source_language, whisper_model)
            # If model was already chosen, also resume translation phase
            if claude_model:
                run_translation_phase(job_id, claude_model)
            return

        if not has_original_srt:
            update_job_info(job_id, status="transcribing")
            from pipeline.transcribe import run_transcription
            run_transcription(job_id, model_size=whisper_model, language=source_language,
                              progress_cb=progress_cb)

        if not has_romanian_srt:
            update_job_info(job_id, status="translating")
            from pipeline.translate import run_translation
            run_translation(job_id, claude_model=claude_model, progress_cb=progress_cb)

        # Quality validation
        progress_cb("validate", 0, "Running quality checks...")
        update_job_info(job_id, status="validating")
        from pipeline.quality import validate_job
        quality_report = validate_job(job_id)
        update_job_info(
            job_id,
            status="ready_for_review",
            progress="Ready for review!",
            quality_score=quality_report["average_score"],
            quality_critical=quality_report["critical_count"],
            quality_warnings=quality_report["warning_count"],
        )
        progress_cb("complete", 100, "Ready for review!")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Retry error for job %s: %s\n%s", job_id, e, tb)
        update_job_info(job_id, status="error", error=str(e), error_traceback=tb,
                        progress=f"Error: {e}")
        _set_progress(job_id, "error", 0, f"Error: {e}")


# ── SSE Progress Endpoint ────────────────────────────────────────────

@app.get("/api/jobs/{job_id}/progress")
async def job_progress_sse(job_id: str):
    """
    Server-Sent Events stream for real-time progress updates.
    Sends progress events every 500ms until the job completes or errors.
    """
    async def event_generator():
        last_msg = ""
        while True:
            prog = _get_progress(job_id)
            info = read_job_info(job_id)
            status = info.get("status", "unknown") if info else "unknown"

            # Build SSE event
            event_data = json.dumps({
                "step": prog["step"],
                "progress": prog["progress"],
                "message": prog["message"],
                "status": status,
            })

            # Only send if something changed
            if event_data != last_msg:
                yield f"data: {event_data}\n\n"
                last_msg = event_data

            # Stop streaming on terminal states
            if status in ("ready_for_review", "complete", "error"):
                # Send one final event
                yield f"data: {json.dumps({'step': status, 'progress': 100 if status != 'error' else 0, 'message': info.get('progress', ''), 'status': status})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
                        "quality_score": info.get("quality_score"),
                        "quality_critical": info.get("quality_critical", 0),
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

    # Load quality report if available
    quality_lookup = {}
    report_path = JOBS_DIR / job_id / "quality_report.json"
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            for seg_report in report.get("segments", []):
                quality_lookup[seg_report["index"]] = seg_report
        except Exception:
            pass

    # Load manually_edited list, feedback, and transcription errors
    info = read_job_info(job_id) or {}
    edited_set = set(info.get("manually_edited", []))
    feedback_map = {f["segment_index"]: f for f in info.get("feedback", [])}

    # Build transcription error lookup from Step 1 summary
    summary = info.get("document_summary", {})
    transcription_errors = {}
    for err in summary.get("transcription_errors", []):
        idx = err.get("segment_index")
        if idx is not None:
            transcription_errors[idx] = {
                "issue": err.get("issue", ""),
                "likely_correct": err.get("likely_correct", ""),
            }

    # Enrich with reading speed, original text, and quality flags
    result = []
    for seg in segments:
        speed = check_reading_speed(seg)
        entry = {
            "index": seg["index"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "original_text": original_segments.get(seg["index"], ""),
            "reading_speed": round(speed, 1),
            "too_fast": speed > 21,
            "manually_edited": seg["index"] in edited_set,
        }
        fb = feedback_map.get(seg["index"])
        if fb:
            entry["feedback_rating"] = fb["rating"]
            entry["feedback_note"] = fb.get("note", "")

        te = transcription_errors.get(seg["index"])
        if te:
            entry["transcription_error"] = te

        q = quality_lookup.get(seg["index"])
        if q:
            entry["quality_score"] = q["score"]
            entry["quality_flags"] = q["flags"]
            entry["has_critical"] = q["has_critical"]
            entry["has_warning"] = q["has_warning"]

        result.append(entry)

    return result


@app.post("/api/job/{job_id}/srt")
async def save_srt(job_id: str, req: SaveSrtRequest):
    """Save edited segments back to the Romanian SRT file."""
    from pipeline.subtitle import write_srt, normalize_romanian

    srt_path = JOBS_DIR / job_id / "transcript_romanian.srt"

    segments = []
    for seg in req.segments:
        segments.append({
            "index": seg["index"],
            "start": seg["start"],
            "end": seg["end"],
            "text": normalize_romanian(seg["text"]),
        })

    # Track which segments were manually edited
    info = read_job_info(job_id) or {}
    edited = set(info.get("manually_edited", []))
    for seg in req.segments:
        if seg.get("manually_edited"):
            edited.add(seg["index"])
    if edited:
        update_job_info(job_id, manually_edited=sorted(edited))

    write_srt(segments, srt_path)
    return {"status": "saved", "segments": len(segments)}


@app.post("/api/job/{job_id}/retranslate")
async def retranslate_segment(job_id: str, req: RetranslateRequest):
    """Re-translate a single segment via Claude API with full context.

    Sends the target segment along with 10 previously-translated segments
    (for flow context), 10 upcoming original segments (for lookahead), the
    Step 1 document summary, the glossary, and the set of glossary terms
    that have already been introduced earlier in the video.
    """
    from pipeline.subtitle import parse_srt
    from pipeline.translate import translate_single_segment, load_glossary

    job_dir = JOBS_DIR / job_id
    srt_original = job_dir / "transcript_original.srt"
    srt_romanian = job_dir / "transcript_romanian.srt"

    if not srt_original.exists():
        return JSONResponse({"error": "Original SRT not found"}, status_code=404)

    original_segs = parse_srt(srt_original)
    translated_segs = parse_srt(srt_romanian) if srt_romanian.exists() else []

    # Locate target by index, falling back to position-in-list lookup
    target_idx = req.segment_index
    target_pos = None
    for i, seg in enumerate(original_segs):
        if seg["index"] == target_idx:
            target_pos = i
            break

    if target_pos is None:
        return JSONResponse({"error": f"Segment {target_idx} not found"}, status_code=404)

    segment = original_segs[target_pos]

    # Gather 10 segments before and after for context
    CONTEXT_WINDOW = 10
    prev_original = original_segs[max(0, target_pos - CONTEXT_WINDOW):target_pos]
    next_original = original_segs[target_pos + 1:target_pos + 1 + CONTEXT_WINDOW]

    # Find matching translated segments by index
    trans_by_idx = {s["index"]: s for s in translated_segs}
    prev_translated = [trans_by_idx[o["index"]] for o in prev_original
                       if o["index"] in trans_by_idx]
    # Trim prev_original to match prev_translated length (1-to-1 pairing)
    prev_original = prev_original[-len(prev_translated):] if prev_translated else []

    # Load summary from job_info.json (saved during Step 1)
    info = read_job_info(job_id) or {}
    summary = info.get("document_summary") or {
        "speaker": "Unknown", "topic": "Islamic lecture",
        "content_summary": "", "detected_terms": {}, "transcription_errors": [],
    }

    # Scan all earlier translated segments to detect which glossary terms
    # have already been introduced (so we don't repeat the parenthetical)
    glossary = load_glossary()
    introduced_terms = set()
    earlier_translated = [s for s in translated_segs
                          if s["index"] < target_idx]
    for s in earlier_translated:
        text_lower = s["text"].lower()
        for arabic, ginfo in glossary.items():
            trans = ginfo.get("transliteration", "")
            if trans and trans.lower() in text_lower:
                introduced_terms.add(trans)

    try:
        translated_text = await asyncio.to_thread(
            translate_single_segment,
            segment, req.source_language, req.claude_model,
            prev_original, prev_translated, next_original,
            summary, introduced_terms,
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


@app.post("/api/job/{job_id}/update-segment")
async def update_segment(job_id: str, req: UpdateSegmentRequest):
    """Update a single segment's text and/or timestamps, mark as manually edited."""
    from pipeline.subtitle import parse_srt, write_srt, normalize_romanian

    srt_path = JOBS_DIR / job_id / "transcript_romanian.srt"
    if not srt_path.exists():
        return JSONResponse({"error": "Romanian SRT not found"}, status_code=404)

    segments = parse_srt(srt_path)
    target = None
    for seg in segments:
        if seg["index"] == req.index:
            target = seg
            break

    if target is None:
        return JSONResponse({"error": f"Segment {req.index} not found"}, status_code=404)

    if req.text is not None:
        target["text"] = normalize_romanian(req.text)
    if req.start is not None:
        target["start"] = req.start
    if req.end is not None:
        target["end"] = req.end

    write_srt(segments, srt_path)

    # Track manually edited segments in job_info
    info = read_job_info(job_id) or {}
    edited = set(info.get("manually_edited", []))
    edited.add(req.index)
    update_job_info(job_id, manually_edited=sorted(edited))

    duration = target["end"] - target["start"]
    chars = target["text"].replace("\n", "").replace("\\N", "").replace("\r", "").strip()
    speed = round(len(chars) / max(duration, 0.1), 1)

    return {
        "index": target["index"],
        "start": target["start"],
        "end": target["end"],
        "text": target["text"],
        "reading_speed": speed,
        "too_fast": speed > 21,
    }


@app.post("/api/job/{job_id}/feedback")
async def submit_feedback(job_id: str, req: FeedbackRequest):
    """Store per-segment reviewer feedback in job_info."""
    info = read_job_info(job_id)
    if info is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    feedback_list = info.get("feedback", [])
    # Replace any existing feedback for this segment
    feedback_list = [f for f in feedback_list if f.get("segment_index") != req.segment_index]
    feedback_list.append({
        "segment_index": req.segment_index,
        "rating": req.rating,
        "note": req.note,
        "timestamp": datetime.now().isoformat(),
    })
    update_job_info(job_id, feedback=feedback_list)

    return {"status": "saved", "total_feedback": len(feedback_list)}


@app.post("/api/job/{job_id}/title")
async def update_title(job_id: str, req: TitleUpdateRequest):
    """Update the Romanian title used for the final export filename."""
    info = read_job_info(job_id)
    if info is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    update_job_info(job_id, title_romanian=req.title_romanian.strip())
    return {"status": "saved", "title_romanian": req.title_romanian.strip()}


@app.post("/api/job/{job_id}/burn")
async def burn_subtitles(job_id: str):
    """Trigger FFmpeg subtitle burning."""
    info = read_job_info(job_id)
    if not info:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    update_job_info(job_id, status="burning", progress="Burning subtitles into video...")
    _set_progress(job_id, "burn", 0, "Burning subtitles into video...")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_burn, job_id)

    return {"status": "burning"}


def _run_burn(job_id: str):
    """Run FFmpeg burn in background thread."""
    try:
        from pipeline.burn import run_burn
        run_burn(job_id)
        update_job_info(job_id, status="complete", progress="Done! Video ready for download.")
        _set_progress(job_id, "complete", 100, "Done! Video ready for download.")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Burn error for job %s: %s\n%s", job_id, e, tb)
        update_job_info(job_id, status="error", error=str(e), progress=f"Burn error: {e}")
        _set_progress(job_id, "error", 0, f"Burn error: {e}")


@app.get("/api/job/{job_id}/quality")
async def get_quality_report(job_id: str):
    """Get quality report for a job. Generates it if not cached."""
    report_path = JOBS_DIR / job_id / "quality_report.json"

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ro_srt = JOBS_DIR / job_id / "transcript_romanian.srt"
    if not ro_srt.exists():
        return JSONResponse({"error": "Romanian SRT not found"}, status_code=404)

    from pipeline.quality import validate_job
    report = validate_job(job_id)
    return report


@app.post("/api/job/{job_id}/quality/refresh")
async def refresh_quality_report(job_id: str):
    """Re-run quality validation (after edits)."""
    ro_srt = JOBS_DIR / job_id / "transcript_romanian.srt"
    if not ro_srt.exists():
        return JSONResponse({"error": "Romanian SRT not found"}, status_code=404)

    from pipeline.quality import validate_job
    report = validate_job(job_id)
    return report


@app.get("/api/job/{job_id}/export/{fmt}")
async def export_subtitles(job_id: str, fmt: str):
    """Export subtitles in the given format (srt, vtt, txt, bilingual, ass)."""
    valid_formats = {"srt", "vtt", "txt", "bilingual", "ass"}
    if fmt not in valid_formats:
        return JSONResponse({"error": f"Invalid format. Choose from: {valid_formats}"}, status_code=400)

    from pipeline.export import export_job
    try:
        results = export_job(job_id, formats=[fmt])
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)

    if fmt not in results:
        return JSONResponse({"error": "Export failed"}, status_code=500)

    filepath = results[fmt]
    media_types = {
        "srt": "application/x-subrip",
        "vtt": "text/vtt",
        "txt": "text/plain",
        "bilingual": "application/x-subrip",
        "ass": "text/x-ssa",
    }
    return FileResponse(filepath, media_type=media_types.get(fmt, "application/octet-stream"),
                        filename=Path(filepath).name)


@app.get("/api/job/{job_id}/exports")
async def list_exports(job_id: str):
    """List available export formats for a job."""
    ro_srt = JOBS_DIR / job_id / "transcript_romanian.srt"
    orig_srt = JOBS_DIR / job_id / "transcript_original.srt"

    formats = []
    if ro_srt.exists():
        formats.extend(["srt", "vtt", "txt", "ass"])
        if orig_srt.exists():
            formats.append("bilingual")

    return {"formats": formats}


@app.post("/api/job/{job_id}/save-to-memory")
async def save_to_translation_memory(job_id: str):
    """Save all reviewed segments from a job to translation memory."""
    info = read_job_info(job_id)
    if not info:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    from pipeline.memory import TranslationMemory
    tm = TranslationMemory()
    source_lang = info.get("source_language", "ar")
    count = tm.store_job_corrections(job_id, source_lang)

    return {"status": "saved", "entries": count}


@app.get("/api/translation-memory/stats")
async def translation_memory_stats():
    """Get translation memory statistics."""
    from pipeline.memory import TranslationMemory
    tm = TranslationMemory()
    return tm.get_stats()


@app.get("/api/translation-memory/lookup")
async def translation_memory_lookup(text: str, lang: str = "ar"):
    """Look up a translation from memory."""
    from pipeline.memory import TranslationMemory
    tm = TranslationMemory()
    matches = tm.lookup(text, lang)
    return {"matches": matches}


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
    # Prefer Romanian title if available (from Step 1 summary pass)
    title = (info.get("title_romanian") or info.get("video_title", "video")) if info else "video"
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:80]
    filename = f"{safe_title}_RO.mp4"

    return FileResponse(str(final_path), media_type="video/mp4", filename=filename)


# ── Lifespan ─────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    logger.info("Dawah-Translate V3 server starting...")
    cleanup_old_jobs()
    logger.info("Server ready at http://localhost:8000")
    yield

app.router.lifespan_context = lifespan

# ── Static files (mounted last so explicit routes take priority) ─────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
