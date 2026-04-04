# Dawah-Translate Roadmap

## Vision

Build a complete, self-hosted pipeline that enables anyone to translate Islamic educational YouTube content into Romanian -- making dawah accessible to Romanian-speaking Muslims and those interested in Islam.

---

## Phase 1: Core Pipeline (COMPLETE)

The foundational pipeline that takes a YouTube URL and produces a subtitled MP4.

### 1.1 Video Intake
- [x] YouTube download via yt-dlp (max 1080p MP4)
- [x] Audio extraction to 16kHz mono WAV (optimal for Whisper)
- [x] Job ID generation and directory management
- [x] Metadata extraction (title, uploader, duration)

### 1.2 Transcription
- [x] faster-whisper integration (CPU mode, int8 quantization)
- [x] Support for large-v3 and medium models
- [x] Auto language detection with confidence reporting
- [x] VAD (Voice Activity Detection) filtering to skip silence
- [x] Progress reporting during transcription
- [x] SRT output with accurate timestamps

### 1.3 Translation
- [x] **Claude API path**: Batch translation with context window, glossary-aware system prompt
- [x] **NLLB-200 local path**: CTranslate2 int8 model, zero API cost
- [x] Islamic terminology glossary with first-occurrence/subsequent rules
- [x] Romanian diacritic normalization (cedilla -> comma-below)
- [x] Reading speed validation (max 21 chars/sec)
- [x] Automatic resegmentation for long subtitle blocks
- [x] Incremental save during translation

### 1.4 Review & Edit
- [x] Web-based review UI with video player + subtitle table
- [x] Inline editing with auto-save
- [x] Side-by-side original and translated text
- [x] Reading speed (CPS) indicator per segment
- [x] Re-translate individual segments
- [x] Keyboard shortcuts (Space, arrows, Enter)
- [x] Glossary editor page

### 1.5 Subtitle Burning
- [x] SRT to ASS conversion with custom styling
- [x] FFmpeg burning with Noto Sans Bold font
- [x] Progress reporting during burn
- [x] Final MP4 download with sanitized filename

### 1.6 Job Management
- [x] Job queuing and background processing
- [x] Status tracking and polling
- [x] Retry failed jobs from last successful step
- [x] Auto-cleanup of old jobs (configurable)

---

## Phase 2: Quality & Polish (NEXT)

Improve translation quality, user experience, and workflow efficiency.

### 2.1 Translation Quality
- [ ] **Speaker diarization**: Identify who is speaking (especially for debates/interviews)
- [ ] **Translation memory**: Cache and reuse translations of identical/similar segments across jobs
- [ ] **Glossary auto-detection**: Suggest new glossary terms based on content analysis
- [ ] **Multi-pass translation**: First pass for meaning, second pass for subtitle fitting
- [ ] **Quality scoring**: Automated scoring of translation naturalness and accuracy

### 2.2 User Experience
- [ ] **Progress bar**: Visual progress for each pipeline step (not just text)
- [ ] **Estimated time**: Show estimated completion time based on video duration
- [ ] **Dark/light theme toggle**: Currently dark-only
- [ ] **Mobile-responsive review UI**: Currently optimized for desktop
- [ ] **Undo/redo in editor**: Track edit history per segment
- [ ] **Bulk edit tools**: Find & replace across all segments

### 2.3 Notifications & Integration
- [ ] **Telegram notifications**: Alert when job is ready for review
- [ ] **Webhook support**: Generic webhook for any notification system
- [ ] **Google Drive upload**: Auto-upload final video to a configured Drive folder
- [ ] **Email notifications**: Optional email when job completes

### 2.4 Batch Processing
- [ ] **Queue system**: Submit multiple URLs, process sequentially
- [ ] **Playlist support**: Submit a YouTube playlist URL
- [ ] **Priority queue**: Prioritize shorter videos for faster turnaround
- [ ] **Concurrent downloads**: Download next video while current one is transcribing

---

## Phase 3: Scale & Share (FUTURE)

Make the tool accessible to others and deployable in different environments.

### 3.1 Deployment
- [ ] **Docker containerization**: Single `docker-compose up` to run everything
- [ ] **Docker GPU support**: NVIDIA GPU acceleration for Whisper and NLLB
- [ ] **Cloud deployment guide**: Railway, Fly.io, or VPS setup instructions
- [ ] **Reverse proxy config**: Nginx/Caddy examples for production

### 3.2 Multi-User
- [ ] **Authentication**: Simple password protection or OAuth
- [ ] **User roles**: Admin (full access) vs. reviewer (edit only)
- [ ] **Job ownership**: Each user sees only their jobs
- [ ] **Usage quotas**: Limit API costs per user

### 3.3 Content Distribution
- [ ] **YouTube direct upload**: Upload final video via YouTube API
- [ ] **Channel management**: Map source channels to output playlists
- [ ] **Thumbnail generation**: Auto-generate thumbnails with Romanian title overlay
- [ ] **SEO metadata**: Auto-generate Romanian titles, descriptions, tags

### 3.4 Additional Languages
- [ ] **Target language selection**: Not just Romanian -- support any NLLB target
- [ ] **Multi-language glossaries**: Per-language glossary files
- [ ] **RTL subtitle support**: For Arabic-target translations

---

## Known Limitations & Challenges

### Current
1. **Long videos (>2 hours)**: Whisper transcription on CPU can take 30-60+ minutes
2. **Arabic dialect handling**: Whisper is better with Modern Standard Arabic than dialects
3. **NLLB translation quality**: Good for simple content, struggles with religious nuance
4. **CPU-only**: No GPU acceleration yet (would 5-10x speed up transcription)
5. **Single concurrent job**: Pipeline runs one job at a time

### Architectural Decisions
- **Why not n8n?** Moving 2-hour 1080p videos through n8n nodes as binary objects crashes the service. FastAPI handles file I/O natively.
- **Why review BEFORE burn?** Original design burned subtitles before human review. Now the human sees and edits text first, avoiding expensive re-burns.
- **Why local + API options?** Local (NLLB) is free and offline but lower quality. Claude API is expensive but understands Islamic context. User chooses per job.
- **Why ASS over SRT for burning?** ASS supports font embedding, outline, shadow, and positioning -- critical for readable subtitles over video.

---

## Alternatives Considered

| Alternative | Why we didn't choose it |
|-------------|------------------------|
| **n8n automation** | Can't handle large video binaries, limited FFmpeg support |
| **Whisper API (cloud)** | Upload limits, no Arabic->Romanian direct path, costs per minute |
| **Google Translate API** | Poor Islamic terminology handling, no glossary control |
| **DeepL API** | No Arabic support |
| **Subtitle Edit (desktop app)** | Manual, no automation, no translation pipeline |
| **Auto-Sub (YouTube)** | No Romanian, no terminology control, can't burn in |
| **Remotion (video rendering)** | Overkill for subtitle overlay, complex setup |

---

## Expansion Ideas (Side-tracks)

These are not on the critical path but could add significant value:

1. **Quran verse detection**: Detect Quranic recitation in audio, auto-insert standard Romanian translation from established tafsir
2. **Islamic lecture database**: Index translated lectures by topic, speaker, and difficulty level
3. **Community review**: Allow multiple reviewers to collaborate on translation quality
4. **A/B translation testing**: Show two translations side by side, let reviewer pick the better one
5. **Subtitle style presets**: Different visual styles for different content types (lectures, debates, nasheed)
6. **Audio dubbing**: Use TTS to generate Romanian audio track (future, experimental)
