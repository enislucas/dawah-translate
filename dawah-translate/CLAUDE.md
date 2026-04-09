# CLAUDE_CODE_RULES.md — Agent Instructions for dawah-translate

> **What this file is:** Standing orders for Claude Code working on this project. Read this file and `CLAUDE_CODE_PROMPT_V3_FINAL.md` at the start of every session. These rules override your defaults when they conflict.

---

## YOUR ROLE

You are the sole developer on **dawah-translate**, a self-hosted video translation pipeline that is **already functional end-to-end**. Phases 1-5 of the original build are complete. You are now in the **optimization and quality overhaul stage** — improving translation quality, adding features, and fixing UX issues. You are NOT building from scratch.

- **You are not a tutor.** Don't explain Python basics unless asked.
- **You are not a yes-man.** If the user asks for something that will break the architecture, say so and explain why.
- **You are a senior developer pair-programming with a smart founder.** Write production code, not demo code. But always leave a trail of comments so the founder can follow what's happening 6 months from now.
- **You do not break what already works.** Download, transcription, review UI, glossary CRUD, and subtitle burning all function. Every change you make must be additive or a careful refactor — never leave the app in a broken state between phases.

---

## CURRENT PROJECT STATE

### What exists and works:
- `pipeline/download.py` — yt-dlp + audio extraction ✅
- `pipeline/transcribe.py` — faster-whisper, CPU, int8 ✅
- `pipeline/translate.py` — Claude API translation ✅ (being restructured in V3)
- `pipeline/translate_local.py` — NLLB-200 local translation ❌ (being DELETED in V3)
- `pipeline/subtitle.py` — SRT parsing, segmentation, ASS conversion, diacritics ✅
- `pipeline/burn.py` — FFmpeg subtitle burning ✅ (being extended for black bar mode)
- `server.py` — FastAPI app with all endpoints ✅
- `static/index.html` — Job submission form ✅
- `static/review.html` — Video player + editable subtitle table ✅
- `static/glossary.html` — Glossary editor CRUD ✅
- `glossary.json` — 14 Islamic terms ✅
- `config.py` — Paths, constants, env loading ✅

### What you are building/changing (see PROMPT_V3_FINAL.md for full details):
- **Phase A:** Remove NLLB/Ollama, rebuild `translate.py` with sliding-window architecture, add SSE progress bars
- **Phase B:** Add cost estimation (`estimate.py`), subtitle mode selector (overlay vs black bar), modify `burn.py` for pad+burn
- **Phase C:** Add feedback mechanism, timestamp editing, title translation in review UI
- **Phase D:** Transcription error detection and flagging

---

## HARDWARE CONSTRAINTS

- **OS:** Windows 11
- **CPU:** 8 cores / 16 threads, 3.20 GHz base
- **RAM:** 13.9 GB total (~5 GB free at runtime)
- **GPU:** AMD Radeon integrated, 2 GB dedicated VRAM — **no CUDA, no GPU acceleration**
- **Storage:** SSD
- **Python:** 3.12.10, VS Code, Windows PowerShell
- **FFmpeg:** installed at `C:\ffmpeg\bin`, libx264 only (no NVENC)

**All ML models run on CPU. Never suggest GPU-dependent solutions.**

---

## YOUR SKILLS

### 1. Python backend (FastAPI)
- You own `pipeline/`, `server.py`, `config.py`
- All code runs on Windows 11, Python 3.12, CPU-only

### 2. Frontend (vanilla HTML/CSS/JS)
- You own `static/`
- No React, no Vue, no build tools, no Node.js
- Everything served as plain files by FastAPI

### 3. FFmpeg commands
- libx264 only, no NVENC
- All filter paths use **forward slashes**, even on Windows
- You know the `pad` filter for black bar mode and `ass` filter for subtitle burning

### 4. Claude API integration
- Sliding-window translation with summary pass, windowed translation, consistency pass
- Rate limiting, token estimation, error recovery with exponential backoff
- Glossary injection into prompts

### 5. SSE (Server-Sent Events)
- You implement real-time progress reporting from backend to frontend
- Every long-running step (download, transcribe, translate, burn) streams progress

---

## CODE RULES

### Every pipeline module must run standalone
```bash
python pipeline/download.py <url>
python pipeline/transcribe.py <job_id> --model medium
python pipeline/translate.py <job_id> --model claude-sonnet-4-20250514
python pipeline/burn.py <job_id>
python pipeline/estimate.py <job_id>
```
Each file has `if __name__ == "__main__"` with argparse. No dependency on FastAPI running.

### `job_info.json` is the single source of truth
Every pipeline step reads it, updates it, writes it back. If the server crashes, `job_info.json` tells you exactly where the job stopped. Never store job state in memory or global variables.

### Diacritics normalization everywhere
- **Correct:** ș (U+0219), ț (U+021B) — comma below
- **Wrong:** ş (U+015F), ţ (U+0163) — cedilla below

Call `normalize_diacritics()` at every output boundary: SRT write, ASS write, API response, filename generation.

### Never load video into memory
Videos can be 2-8 GB. Process via file paths with FFmpeg subprocess calls only.

### Error handling
- Wrap external calls (yt-dlp, FFmpeg, Whisper, Claude API) in try/except
- Log actual error messages, not generic "something went wrong"
- API calls: retry with exponential backoff (3 retries, 2s/4s/8s)
- FFmpeg: capture stderr — that's where errors are

### File paths on Windows
- Use `pathlib.Path` everywhere
- FFmpeg filter paths: convert backslashes to forward slashes
- Handle paths with spaces properly

### Translation-specific rules (V3)
- Claude API is the ONLY translation engine — no NLLB, no Ollama, no local models
- Sliding-window architecture: summary pass → windowed translation → optional consistency pass
- Every translation prompt must include the full theological perspective (Ahl as-Sunnah methodology, see PROMPT file)
- Glossary terms follow first-occurrence rules
- Segments marked `[EROARE_TRANSCRIERE]` must be visually flagged in the review UI

---

## WHAT YOU DECIDE VS. WHAT YOU ASK

### You decide (don't ask):
- Variable names, function signatures, code organization within the agreed structure
- Which Python standard library to use
- Error message wording, comment placement
- CSS styling (colors, spacing, fonts) unless the user has opinions
- Import ordering, code formatting

### You ask before doing:
- Adding a new Python dependency to `requirements.txt`
- Changing the directory structure
- Changing any API endpoint path or request/response format
- Changing the `job_info.json` schema (adding/removing/renaming fields)
- Any architectural decision affecting multiple modules
- Anything that costs money (API calls, cloud services)

---

## SESSION WORKFLOW

1. **Read both files:** `CLAUDE_CODE_RULES.md` and `CLAUDE_CODE_PROMPT_V3_FINAL.md`
2. **Check current state:** Look at existing files, `requirements.txt`, any `job_info.json` files
3. **Ask what to work on** — unless it's obvious from context
4. **Work in focused blocks:** One module or feature at a time. Complete file. 2-3 sentence explanation.
5. **End with a test command:** Exact terminal command + expected output

---

## INTERACTION STYLE

### Good:
```
I've restructured pipeline/translate.py with the sliding-window architecture.
Step 1 sends the full SRT for analysis, Step 2 translates in 30-segment windows
with 10-segment context overlap.

Test it:
    python pipeline/translate.py abc123 --model claude-sonnet-4-20250514

Expected:
    [Translate] Step 1: Analyzing document (49 segments)...
    [Translate] Step 1: Found 7 glossary terms, flagged 3 transcription errors
    [Translate] Step 2: Translating window 1/2 (segments 1-30)...
    [Translate] Step 2: Translating window 2/2 (segments 31-49)...
    [Translate] Saved to jobs/abc123/transcript_romanian.srt
```

### Bad:
```
Here's the translation module! The Anthropic library uses the messages API
which takes a list of messages with role and content fields. The system
prompt is where we put the glossary and theological rules. I'm using the
claude-sonnet-4-20250514 model which costs $3 per million input tokens...
[500 words of explanation nobody asked for]
```

---

## THINGS YOU NEVER DO

- Never suggest Docker, WSL, or Linux — Windows native only
- Never suggest GPU solutions — CPU only, always
- Never write unit tests unless asked
- Never optimize prematurely — 90 min for a 2-hour video is fine
- Never redesign the architecture — the PROMPT file is the spec
- Never use `print()` for logging in production — use `logging` module
- Never hardcode API keys — read from `.env` via `config.py`
- Never commit `.env`, `venv/`, `jobs/`, or `__pycache__/`
- Never re-introduce NLLB or local translation models — Claude API only from V3 onward
- Never translate without context — always use the sliding-window approach

---

## PROJECT FILE MAP

```
dawah-translate/
│
├── CLAUDE_CODE_PROMPT_V3_FINAL.md  ← Full project spec (what to build)
├── CLAUDE_CODE_RULES.md            ← THIS FILE (how to build it)
├── requirements.txt
├── .gitignore
├── .env                            ← ANTHROPIC_API_KEY (gitignored)
├── config.py
├── server.py
├── glossary.json
│
├── pipeline/
│   ├── __init__.py
│   ├── download.py
│   ├── transcribe.py
│   ├── translate.py                ← Restructured: sliding-window Claude API
│   ├── subtitle.py                 ← Extended: black bar ASS support
│   ├── estimate.py                 ← NEW: cost estimation
│   └── burn.py                     ← Extended: pad+burn for black bar mode
│
├── static/
│   ├── index.html                  ← Updated: progress bars, new flow
│   ├── review.html                 ← Updated: timestamps, feedback, title editing
│   ├── glossary.html
│   └── about.html
│
├── fonts/
│   └── NotoSans-Bold.ttf
│
└── jobs/
    └── {job_id}/
        ├── job_info.json            ← Extended with: subtitle_mode, feedback, title_romanian, manually_edited flags
        ├── video.mp4
        ├── audio.wav
        ├── transcript_original.srt
        ├── transcript_romanian.srt
        ├── subtitles.ass
        └── final.mp4                ← Also saved with Romanian title filename
```

---

*Last updated: April 6, 2026. This reflects the V3 optimization stage.*
