# Dawah-Translate

**YouTube video translation pipeline for Romanian Islamic dawah content.**

Dawah-Translate takes a YouTube video in any language (Arabic, English, Turkish, French, etc.), transcribes it, translates the subtitles to Romanian with proper Islamic terminology handling, lets you review and edit them, and burns the subtitles directly into the video -- producing a ready-to-upload MP4.

![Pipeline Overview](docs/diagrams/pipeline-overview.svg)

---

## Features

- **Web UI** -- Submit YouTube URLs, monitor progress, review/edit subtitles side-by-side with the video
- **Dual translation engines**:
  - **Local (NLLB-200)** -- Free, runs entirely on your machine (~400 MB model), no API costs
  - **Claude API** -- Higher quality, context-aware translations with glossary enforcement
- **Islamic terminology glossary** -- Arabic terms like Salah, Tawheed, du'a are transliterated (not translated) with first-occurrence explanations
- **Romanian diacritics** -- Correct comma-below characters (ș, ț) enforced throughout
- **Reading speed validation** -- Flags subtitles that exceed 21 characters/second (too fast to read)
- **Subtitle review UI** -- Edit translations inline, re-translate individual segments, keyboard shortcuts for video navigation
- **FFmpeg subtitle burning** -- ASS-formatted subtitles with Noto Sans Bold, burned directly into the MP4
- **Job management** -- Track multiple jobs, retry failed ones, auto-cleanup old jobs

---

## Architecture

```
dawah-translate/
├── server.py              # FastAPI web server (API + static pages)
├── config.py              # Central configuration
├── glossary.json          # Islamic terminology glossary
├── requirements.txt       # Python dependencies
├── pipeline/
│   ├── download.py        # YouTube download via yt-dlp + audio extraction
│   ├── transcribe.py      # Speech-to-text via faster-whisper (CPU, int8)
│   ├── translate.py       # Translation via Claude API with glossary
│   ├── translate_local.py # Translation via NLLB-200 (local, no API)
│   ├── subtitle.py        # SRT parsing, writing, normalization, ASS conversion
│   └── burn.py            # FFmpeg subtitle burning into video
├── static/
│   ├── index.html         # Job submission form + job list
│   ├── review.html        # Subtitle review/edit UI with video player
│   ├── glossary.html      # Glossary editor
│   ├── about.html         # Project documentation & context
│   └── architecture.html  # Technical architecture & diagrams
└── fonts/
    └── NotoSans-Bold.ttf  # Font for burned subtitles
```

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **FFmpeg** installed and on PATH ([download](https://ffmpeg.org/download.html))
- *Optional*: Anthropic API key (for Claude-based translation)

### Installation

```bash
# Clone the repo
git clone https://github.com/enislucas/dawah-translate.git
cd dawah-translate/dawah-translate

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Anthropic API key (optional -- only needed for Claude translation)
# ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
python server.py
# Open http://localhost:8000
```

### Usage

1. Paste a YouTube URL into the form
2. Select source language, Whisper model, and translation engine
3. Click **Start Translation** and wait for the pipeline to complete
4. Review and edit subtitles in the review UI (video plays alongside)
5. Click **Approve & Burn Subtitles** to produce the final MP4
6. Download the finished video

---

## Pipeline Steps

| Step | Component | What it does |
|------|-----------|-------------|
| 1. Download | `pipeline/download.py` | Downloads video via yt-dlp (max 1080p), extracts 16kHz mono WAV audio |
| 2. Transcribe | `pipeline/transcribe.py` | Runs faster-whisper (large-v3 or medium) on CPU with VAD filtering |
| 3. Translate | `pipeline/translate.py` or `translate_local.py` | Translates SRT segments to Romanian with glossary post-processing |
| 4. Review | `static/review.html` | Human-in-the-loop: edit translations, check reading speed, re-translate segments |
| 5. Burn | `pipeline/burn.py` | Converts SRT to ASS, burns into video with FFmpeg (libx264, Noto Sans Bold) |

---

## Islamic Terminology Glossary

The glossary (`glossary.json`) controls how Arabic Islamic terms are handled in translation:

| Arabic | Transliteration | Rule |
|--------|----------------|------|
| صلاة | Salah | First: "Salah (rugaciunea rituala)", then: "Salah" |
| دعاء | du'a | First: "du'a (invocatie)", then: "du'a" |
| توحيد | Tawheed | First: "Tawheed (unicitatea lui Dumnezeu)", then: "Tawheed" |
| شرك | shirk | First: "shirk (asociere)", then: "shirk" |
| سنة | Sunnah | Always "Sunnah" |
| حديث | hadith | Always "hadith" |

Edit the glossary at `http://localhost:8000/glossary` or directly in `glossary.json`.

---

## Technology Choices & Rationale

| Choice | Why |
|--------|-----|
| **FastAPI** over n8n | n8n can't handle large video binaries in memory; FastAPI gives full control over streaming, background tasks, and file I/O |
| **faster-whisper** over Whisper API | Runs locally, no upload limits, supports CPU int8 quantization, handles long videos |
| **NLLB-200 local option** | Zero cost, offline capable, ~400 MB model, good enough for simple content |
| **Claude API option** | Context-aware translation, understands glossary rules natively, much better for Arabic theological content |
| **ASS subtitles** over raw SRT burn | Full styling control (font, size, outline, shadow), proper Romanian diacritic rendering |
| **Noto Sans Bold** | Excellent Romanian diacritic support (ș, ț, ă, î, â), free, widely available |
| **Review before burn** | Original prompt burned before review -- now human sees text first, burns only after approval |

---

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full roadmap.

### Current Status: Phase 1 (Core Pipeline) -- Complete

- [x] YouTube download + audio extraction
- [x] Whisper transcription (CPU, large-v3/medium)
- [x] Claude API translation with glossary
- [x] Local NLLB-200 translation with glossary post-processing
- [x] SRT parsing, normalization, resegmentation
- [x] Web UI (submission, review, glossary editor)
- [x] FFmpeg subtitle burning (ASS format)
- [x] Job management (retry, cleanup)

### Next: Phase 2 (Quality & Polish)

- [ ] Speaker diarization for multi-speaker videos
- [ ] Telegram/webhook notifications
- [ ] Google Drive upload integration
- [ ] Batch processing (queue multiple videos)
- [ ] Translation memory (reuse previous translations)

### Future: Phase 3 (Scale & Share)

- [ ] Docker containerization
- [ ] Multi-user support with authentication
- [ ] Cloud deployment option (Railway/Fly.io)
- [ ] YouTube direct upload via API

---

## Contributing

This is primarily a personal project for Romanian Islamic dawah content, but contributions are welcome. Please open an issue first to discuss changes.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgments

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) -- CTranslate2-based Whisper inference
- [NLLB-200](https://ai.meta.com/research/no-language-left-behind/) -- Meta's multilingual translation model
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) -- YouTube downloader
- [Anthropic Claude](https://www.anthropic.com/) -- AI translation with context awareness
- [FFmpeg](https://ffmpeg.org/) -- Video processing
- [Noto Sans](https://fonts.google.com/noto/specimen/Noto+Sans) -- Google's universal font family
