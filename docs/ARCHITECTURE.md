# Architecture Overview

## System Design

Dawah-Translate is a **self-hosted, single-machine web application** built as a FastAPI server with a pipeline architecture. Each job flows through a sequence of processing stages, with human review inserted before the expensive subtitle-burning step.

```
User Browser                     FastAPI Server                    Local Machine
┌──────────┐     HTTP/JSON      ┌──────────────┐                 ┌─────────────┐
│ index.html├───────────────────►│ POST /submit │────────────────►│ yt-dlp      │
│ review.html│◄──────────────────│ GET /status  │                 │ FFmpeg      │
│ glossary.html│                 │ GET/POST /srt│                 │ Whisper     │
└──────────┘                    │ POST /burn   │                 │ NLLB-200    │
                                └──────────────┘                 │ Claude API  │
                                                                 └─────────────┘
```

## Request Flow

### Job Submission
```
Browser → POST /api/submit { url, source_language, whisper_model, translation_engine }
       ← { job_id, status: "queued" }
       
Server creates job directory: jobs/{job_id}/
Server starts background thread: run_pipeline(job_id, ...)
Browser polls: GET /api/status/{job_id} every 3 seconds
```

### Pipeline Execution (Background Thread)
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Download    │────►│  Transcribe  │────►│  Translate   │────►│ Ready for    │
│  (yt-dlp +  │     │  (Whisper    │     │  (NLLB or   │     │ Review       │
│   FFmpeg)   │     │   CPU int8)  │     │   Claude)   │     │              │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                     │
                                                              Human reviews &
                                                              edits subtitles
                                                                     │
                                                              ┌──────▼───────┐
                                                              │  Burn        │
                                                              │  (FFmpeg +   │
                                                              │   ASS subs)  │
                                                              └──────┬───────┘
                                                                     │
                                                              ┌──────▼───────┐
                                                              │  Download    │
                                                              │  final.mp4   │
                                                              └──────────────┘
```

### Job Directory Structure
```
jobs/{job_id}/
├── job_info.json            # Job metadata and status tracking
├── video.mp4                # Downloaded source video (max 1080p)
├── audio.wav                # Extracted audio (16kHz, mono, PCM16)
├── transcript_original.srt  # Whisper transcription in source language
├── transcript_romanian.srt  # Translated Romanian subtitles
├── subtitles.ass            # ASS-formatted subtitles for burning
└── final.mp4                # Output video with burned-in subtitles
```

## Component Details

### Download (`pipeline/download.py`)
- Uses **yt-dlp** Python library (not CLI) for reliable YouTube downloads
- Caps at 1080p to balance quality vs. file size and burn time
- Extracts audio to **16kHz mono WAV** (PCM 16-bit) -- the exact format Whisper expects
- Reports download progress via hooks

### Transcription (`pipeline/transcribe.py`)
- Uses **faster-whisper** (CTranslate2-based Whisper reimplementation)
- Runs on **CPU with int8 quantization** -- no GPU needed
- Supports **large-v3** (best accuracy, 3GB) and **medium** (faster, 1.5GB)
- Enables **VAD filtering** to skip silent sections (saves time on lectures with pauses)
- Auto-detects language with confidence score
- Outputs standard SRT with millisecond-accurate timestamps

### Translation -- Claude API (`pipeline/translate.py`)
- Sends segments in **batches of 50** with context from previous batch
- System prompt includes:
  - Full glossary with first-occurrence/subsequent rules
  - Character limit per segment based on display duration
  - Instructions for natural Romanian with proper diacritics
- Tracks which glossary terms have been "introduced" across batches
- Retries with exponential backoff on API errors
- Saves progress incrementally (crash-safe)

### Translation -- NLLB-200 Local (`pipeline/translate_local.py`)
- Uses **facebook/nllb-200-distilled-600M** via CTranslate2 (int8, ~400 MB)
- Translates in batches of 20 for efficiency
- **Glossary post-processing**: scans source for Arabic terms, finds NLLB's Romanian output, replaces with correct transliteration
- Has a table of known NLLB output patterns for Islamic terms (e.g., NLLB translates "صلاة" as "rugăciune" which gets replaced with "Salah")
- Caches translator instance for re-translation requests

### Subtitle Utilities (`pipeline/subtitle.py`)
- **Romanian normalization**: Replaces cedilla characters (ş, ţ) with correct comma-below (ș, ț)
- **Reading speed check**: Calculates characters/second per segment (>21 = too fast)
- **Resegmentation**: Splits segments exceeding 2 lines x 42 chars, distributing time proportionally
- **SRT ↔ ASS conversion**: Generates ASS with Noto Sans Bold, 56pt, white text with black outline

### Subtitle Burning (`pipeline/burn.py`)
- Converts Romanian SRT to ASS format with styled template
- Burns using FFmpeg: `ffmpeg -vf ass=subtitles.ass:fontsdir=fonts/`
- Uses **libx264** codec, **medium** preset, **CRF 20** (good quality)
- Copies audio stream without re-encoding
- Reports burn progress by parsing FFmpeg's time output

### Web Server (`server.py`)
- **FastAPI** with uvicorn
- Background tasks via `asyncio.get_event_loop().run_in_executor()`
- Video streaming with **HTTP Range** request support (for seeking in browser)
- Static file serving for HTML/CSS/JS
- Job lifecycle management with JSON-based state

## Data Flow Diagram

```
                    ┌─────────────────────────────────────────┐
                    │              glossary.json               │
                    │  (Arabic → transliteration rules)       │
                    └────────┬──────────────┬─────────────────┘
                             │              │
  YouTube URL ──► download ──► transcribe ──► translate ──► review ──► burn ──► final.mp4
                    │            │              │            │          │
                    ▼            ▼              ▼            ▼          ▼
                 video.mp4   audio.wav    original.srt  romanian.srt  final.mp4
                             (16kHz)      (source lang)  (edited)    (with subs)
```

## Configuration (`config.py`)

| Setting | Default | Purpose |
|---------|---------|---------|
| `MAX_CHARS_PER_LINE` | 42 | Maximum characters per subtitle line |
| `MAX_LINES_PER_BLOCK` | 2 | Maximum lines per subtitle block |
| `MAX_READING_SPEED` | 21 cps | Characters per second threshold |
| `MIN_DISPLAY_DURATION` | 1.0s | Minimum subtitle display time |
| `MAX_DISPLAY_DURATION` | 7.0s | Maximum subtitle display time |
| `TRANSLATION_BATCH_SIZE` | 50 | Segments per Claude API call |
| `TRANSLATION_CONTEXT_SIZE` | 5 | Previous segments sent as context |
| `FFMPEG_PRESET` | "medium" | Encoding speed/quality tradeoff |
| `FFMPEG_CRF` | 20 | Video quality (lower = better, bigger) |
| `JOB_MAX_AGE_DAYS` | 7 | Auto-delete jobs older than this |

## Security Notes

- `.env` file contains API keys and is gitignored
- No authentication on the web UI (intended for local/personal use)
- Job files are stored locally and auto-cleaned
- YouTube URLs are not validated beyond what yt-dlp accepts
