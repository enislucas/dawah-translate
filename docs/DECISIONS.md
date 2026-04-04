# Design Decisions & Alternatives

This document records key decisions made during the project, what alternatives were considered, and why.

---

## Decision 1: FastAPI over n8n

**Chosen**: FastAPI (Python web server)
**Rejected**: n8n (visual workflow automation)

### Why
The original plan used n8n for the entire pipeline. This was rejected because:

1. **Memory**: n8n passes data between nodes as binary objects in memory. A 2-hour 1080p video (~2-4 GB) would crash the n8n service
2. **FFmpeg**: n8n has limited FFmpeg support. Subtitle burning requires precise FFmpeg flags (ASS filter, font embedding, CRF settings) that are hard to configure in a node
3. **Long-running tasks**: Whisper transcription of a 2-hour video takes 30-60 minutes on CPU. n8n nodes aren't designed for tasks that long
4. **File I/O**: FastAPI can stream video files directly with HTTP Range support. n8n would need to serialize/deserialize the entire file per node
5. **Flexibility**: Python gives direct access to faster-whisper, CTranslate2, and the Anthropic SDK

### Tradeoff
n8n would have been simpler for non-engineers to modify. FastAPI requires Python knowledge to extend.

---

## Decision 2: Review BEFORE Burn (not after)

**Chosen**: Transcribe → Translate → Review → Burn
**Rejected**: Transcribe → Translate → Burn → Review

### Why
The original pipeline burned subtitles before the human saw the text. This meant:
- Reviewing required watching the entire burned video
- Any edit required re-burning (CPU-intensive, 10-30 minutes for long videos)
- Impossible to compare original and translated text side-by-side

Now the human sees the raw SRT alongside the video, edits text directly, and only burns once.

---

## Decision 3: Dual Translation Engines

**Chosen**: Both local NLLB-200 and Claude API
**Rejected**: Claude API only

### Why
1. **Cost**: Claude API costs ~$0.01-0.05 per video depending on length and model. NLLB is free
2. **Offline**: NLLB works without internet after initial model download
3. **Privacy**: Source text never leaves the machine with NLLB
4. **Quality tradeoff**: NLLB is adequate for simple English→Romanian. Claude is necessary for Arabic→Romanian theological content where glossary enforcement matters

### When to use which
- **NLLB**: Short clips, English source, simple content, tight budget
- **Claude**: Arabic source, lectures with theological terminology, content where accuracy matters

---

## Decision 4: faster-whisper over Whisper API

**Chosen**: faster-whisper (local CPU, int8)
**Rejected**: OpenAI Whisper API

### Why
1. **No upload limit**: Whisper API has a 25 MB file size limit. A 2-hour audio file is ~100+ MB
2. **No direct Arabic→Romanian**: Whisper API only transcribes or translates to English. We need the source language transcription first, then translate separately
3. **Cost**: Whisper API charges per minute of audio. Local is free
4. **Speed**: faster-whisper with int8 quantization is actually competitive with the API for medium-length videos
5. **Control**: VAD parameters, word-level timestamps, beam size are all configurable locally

### Tradeoff
First run downloads a 1.5-3 GB model. Transcription on CPU is 3-10x slower than GPU. For a team scenario, a GPU-equipped server would be ideal.

---

## Decision 5: ASS Subtitles over SRT Burning

**Chosen**: Convert SRT → ASS, then burn ASS
**Rejected**: Burn SRT directly with FFmpeg

### Why
1. **Font control**: ASS embeds font family, size, weight in the subtitle file
2. **Outline & shadow**: ASS supports text outline (3px) and shadow (1px) for readability over any video background
3. **Diacritics**: With ASS + Noto Sans, Romanian ș/ț render correctly. Raw SRT burning sometimes falls back to system fonts that lack proper glyphs
4. **Positioning**: ASS supports precise vertical positioning (margin from bottom)
5. **Resolution-aware**: ASS uses PlayRes (1920x1080) for consistent sizing across video resolutions

---

## Decision 6: Glossary Post-Processing (NLLB path)

**Chosen**: Translate with NLLB, then post-process to fix Islamic terms
**Rejected**: Pre-process (mask terms before translation), or fine-tune NLLB

### Why
1. **NLLB can't be prompted**: Unlike Claude, NLLB doesn't understand instructions. It just translates text
2. **Masking is fragile**: Replacing Arabic terms with placeholders before translation confuses NLLB's context understanding
3. **Post-processing is predictable**: We know NLLB translates "صلاة" to "rugăciune". We can reliably find and replace this
4. **Fine-tuning is expensive**: Would require a parallel corpus of Islamic content in Arabic→Romanian

### Limitation
Post-processing only works for known patterns. If NLLB produces an unexpected Romanian word for an Arabic term, the glossary won't catch it. This is where Claude's contextual understanding is superior.

---

## Decision 7: Job-based Architecture (not streaming)

**Chosen**: Create a job directory per video, process asynchronously
**Rejected**: Real-time streaming pipeline

### Why
1. **Resilience**: If the server crashes, job state is preserved on disk. Can retry from last step
2. **Review workflow**: Human review is inherently asynchronous -- you can't stream through a review step
3. **Large files**: Video files need to be on disk, not in memory
4. **Simplicity**: Each job is a directory with predictable file names. Easy to debug, inspect, and clean up

---

## Decision 8: Single-Page HTML (no framework)

**Chosen**: Vanilla HTML/CSS/JS served as static files
**Rejected**: React, Vue, Svelte

### Why
1. **Simplicity**: The UI has 3 pages total. A framework would add build tooling, node_modules, and complexity for no benefit
2. **Portability**: Anyone can open index.html and understand it. No build step needed
3. **Performance**: Zero JS framework overhead. Page loads instantly
4. **Maintainability**: For a personal project, vanilla JS is easier to modify than a framework with state management

### Tradeoff
If the UI grows significantly (e.g., multi-user dashboard, playlist management), a framework might become worth it.
