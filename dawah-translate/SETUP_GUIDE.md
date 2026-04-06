# Dawah-Translate: Setup & Usage Guide

A self-hosted tool that downloads Islamic YouTube lectures, transcribes the Arabic/English audio, translates subtitles to Romanian with Islamic terminology awareness, lets you review and edit in a browser, and burns hardcoded subtitles into the final video.

**Everything runs on your own PC. No cloud services required. No ongoing costs.**

---

## Table of Contents

1. [What You Need (Prerequisites)](#1-what-you-need-prerequisites)
2. [One-Time Installation](#2-one-time-installation)
3. [Starting the Application](#3-starting-the-application)
4. [Using the Application](#4-using-the-application)
5. [Understanding Translation Modes](#5-understanding-translation-modes)
6. [Reviewing & Editing Subtitles](#6-reviewing--editing-subtitles)
7. [Exporting Subtitles](#7-exporting-subtitles)
8. [Quality Checks & Creedal Safety](#8-quality-checks--creedal-safety)
9. [Troubleshooting](#9-troubleshooting)
10. [Glossary of Technical Terms](#10-glossary-of-technical-terms)

---

## 1. What You Need (Prerequisites)

Before installing Dawah-Translate, you need these programs on your computer. If you already have them, skip to Step 2.

### Required Software

| Software | What It Does | Download Link |
|----------|-------------|---------------|
| **Python 3.11 or 3.12** | Runs the application code | https://www.python.org/downloads/ |
| **FFmpeg** | Processes video and audio files | https://ffmpeg.org/download.html |
| **Ollama** | Runs the local AI translation model | https://ollama.com/download |
| **Git** | Downloads the project code | https://git-scm.com/downloads |

### Hardware Requirements

- **CPU**: Any modern processor (4+ cores recommended)
- **RAM**: At least 8 GB (14 GB+ recommended)
- **Disk**: ~10 GB free space (for AI models and video files)
- **GPU**: Not required. Everything runs on CPU.
- **Internet**: Required only for downloading YouTube videos and AI models. Translation itself is fully offline.

---

## 2. One-Time Installation

Open a terminal (Command Prompt or PowerShell on Windows, Terminal on Mac/Linux) and follow these steps:

### Step 1: Install Python

Download Python from https://www.python.org/downloads/ and install it.

**IMPORTANT (Windows):** During installation, check the box that says **"Add Python to PATH"**.

Verify it works:
```
python --version
```
You should see something like `Python 3.12.x`.

### Step 2: Install FFmpeg

**Windows:**
1. Download from https://github.com/BtbN/FFmpeg-Builds/releases (choose `ffmpeg-master-latest-win64-gpl.zip`)
2. Extract the ZIP file
3. Copy the `ffmpeg.exe`, `ffprobe.exe`, and `ffplay.exe` files from the `bin` folder to a folder like `C:\ffmpeg\`
4. Add `C:\ffmpeg\` to your system PATH:
   - Search "Environment Variables" in the Start menu
   - Click "Edit the system environment variables"
   - Click "Environment Variables..."
   - Under "System variables", find `Path`, click "Edit..."
   - Click "New" and type `C:\ffmpeg`
   - Click OK on all windows

**Mac:**
```
brew install ffmpeg
```

**Linux:**
```
sudo apt install ffmpeg
```

Verify it works:
```
ffmpeg -version
```

### Step 3: Install Ollama

Download and install from https://ollama.com/download

After installation, Ollama runs automatically in the background. You can verify:
```
ollama --version
```

Now download the AI model used for translation refinement (~3.3 GB download, one-time):
```
ollama pull gemma3:4b
```
Wait for the download to complete. This only needs to happen once.

### Step 4: Download the Project

```
git clone https://github.com/enislucas/dawah-translate.git
cd dawah-translate
```

### Step 5: Create a Virtual Environment and Install Dependencies

**Windows:**
```
cd dawah-translate
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```
cd dawah-translate
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This installs all the Python libraries the application needs. It may take a few minutes.

### Step 6: Download the Whisper Transcription Model

The first time you run the app and submit a video, it will automatically download the Whisper speech recognition model (~3 GB). This happens only once and the model is cached for future use.

---

## 3. Starting the Application

Every time you want to use Dawah-Translate, you need to:

### Step 1: Make sure Ollama is running

Ollama usually starts automatically when your PC boots. To check:
```
ollama list
```
If you see `gemma3:4b` in the list, you're good. If not:
```
ollama serve
```
(Leave this terminal window open.)

### Step 2: Activate the virtual environment and start the server

Open a **new** terminal window:

**Windows:**
```
cd path\to\dawah-translate\dawah-translate
venv\Scripts\activate
python server.py
```

**Mac/Linux:**
```
cd path/to/dawah-translate/dawah-translate
source venv/bin/activate
python server.py
```

You should see:
```
Dawah-Translate server starting...
Server ready at http://localhost:8000
```

### Step 3: Open your browser

Go to: **http://localhost:8000**

You should see the Dawah-Translate home page.

---

## 4. Using the Application

### Submitting a Video for Translation

1. Open **http://localhost:8000** in your browser
2. Paste a YouTube URL into the URL field (e.g., `https://www.youtube.com/watch?v=...`)
3. Select the **Source Language** (Arabic, English, or Auto-detect)
4. Select the **Translation Engine** (see Section 5 below)
5. Click **Submit**

The application will:
- Download the video from YouTube
- Extract the audio
- Transcribe the speech to text (this takes a while for long videos)
- Translate the transcript to Romanian
- Run quality checks

You can watch the progress on the home page. When it says **"Ready for review"**, click the job to open the review page.

### How Long Does It Take?

| Step | ~Time for a 30-minute video |
|------|---------------------------|
| Download | 1-2 minutes |
| Transcription | 10-30 minutes (CPU) |
| NLLB Translation | ~30 seconds |
| Ollama Refinement | 5-15 minutes |
| Quality Check | ~2 seconds |

---

## 5. Understanding Translation Modes

The app offers three translation modes:

### Local Fast (NLLB-200 only)
- **Speed**: Very fast (~30 seconds)
- **Quality**: Basic. Good for getting the general meaning, but Islamic terminology will often be wrong.
- **Cost**: Free
- **When to use**: Quick preview or when you plan to manually edit everything.

### Local Quality (NLLB + Ollama) -- RECOMMENDED
- **Speed**: Moderate (~8-15 minutes for 50 segments)
- **Quality**: Good. The AI model corrects Islamic terminology errors and makes the Romanian more natural.
- **Cost**: Free
- **When to use**: Default choice. Best balance of quality and cost.
- **Requires**: Ollama must be running with gemma3:4b model.

### Cloud (Claude API)
- **Speed**: Fast (~1-2 minutes)
- **Quality**: Best. Uses Anthropic's Claude AI via the internet.
- **Cost**: ~$0.50-3.00 per video (requires API key)
- **When to use**: When you need the highest quality and don't mind the cost.
- **Requires**: An Anthropic API key in your `.env` file.

---

## 6. Reviewing & Editing Subtitles

After translation completes, click the job to open the **Review Page**:

### The Review Interface

- **Top**: Quality banner (green = good, yellow = warnings, red = critical issues)
- **Video player**: Watch the video with subtitle preview overlay
- **Subtitle table**: All segments listed with:
  - **#**: Segment number
  - **Timecode**: Start and end times
  - **Original**: The source language text (Arabic/English)
  - **Romanian**: The translated text (click to edit!)
  - **CPS**: Characters per second (reading speed)
  - **QA**: Quality score (100 = perfect)
  - **Action**: Re-translate button

### Editing Subtitles

- **Click** any Romanian text to edit it directly
- Changes are **saved automatically** after you stop typing (800ms delay)
- The video player syncs with the table -- the current segment is highlighted in blue

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause video |
| Left Arrow | Jump to previous segment |
| Right Arrow | Jump to next segment |
| Enter | Play current segment |
| Escape | Stop editing text |

### Re-translating a Segment

If a single segment is bad, click the **Re-translate** button on that row. It will re-run the translation for just that segment.

### Bottom Bar Buttons

- **Export**: Download subtitles in various formats (SRT, VTT, TXT, Bilingual, ASS)
- **Save to Memory**: Saves your reviewed translations to a local database. Future translations of the same phrases will use your corrections.
- **Re-check Quality**: Re-runs quality checks after you've made edits
- **Approve & Burn Subtitles**: Burns the subtitles permanently into the video file

### Burning Subtitles

When you're satisfied with the subtitles:
1. Click **Approve & Burn Subtitles**
2. Wait for FFmpeg to process the video (this can take several minutes for long videos)
3. When done, a **Download Final Video** button appears
4. Click it to download the video with hardcoded Romanian subtitles

---

## 7. Exporting Subtitles

Click the **Export** button in the bottom bar to download subtitles in these formats:

| Format | Description | Use Case |
|--------|-------------|----------|
| **SRT** | Standard subtitle file | Most video players, YouTube upload |
| **VTT** | Web Video Text Tracks | Web players, HTML5 video |
| **TXT** | Plain text transcript | Reading, sharing text |
| **Bilingual** | Arabic + Romanian SRT | Study, comparison |
| **ASS** | Styled subtitles | Advanced video editing |

---

## 8. Quality Checks & Creedal Safety

The application automatically checks every translated segment for issues:

### Critical Issues (Red)
These are **theological errors** that could misrepresent Islamic beliefs:
- "Islamism" instead of "Islam" (implies political ideology)
- "Unification" instead of "Tawheed" (monotheism)
- Feminine pronouns for prophets/scholars
- "Punishment" instead of "good end" (العاقبة mistranslation)
- "Galilee" instead of "the illustrious" (الجليل mistranslation)

**You must fix critical issues before publishing.**

### Warnings (Yellow)
- Known NLLB hallucination patterns (e.g., "council" for da'wah)
- Untranslated or garbled terms
- Very fast reading speed

### Info (Blue)
- Line too long for subtitle display
- Slightly fast reading speed
- Minor formatting issues

### The Quality Score

Each segment gets a score from 0 to 100:
- **95-100**: Good quality
- **80-94**: Some issues, review recommended
- **Below 80**: Significant issues, must review

---

## 9. Troubleshooting

### "Ollama is not running" warning on the home page

Make sure Ollama is running. Open a terminal and run:
```
ollama serve
```
Leave this window open while using the app.

### "Model not found" error

You need to download the AI model:
```
ollama pull gemma3:4b
```

### Transcription is very slow

This is normal on CPU. The Whisper large-v3 model is very accurate but slow. For faster (less accurate) transcription, select a smaller model (medium or small) in the submission form.

### FFmpeg not found

Make sure FFmpeg is installed and in your system PATH. Test with:
```
ffmpeg -version
```

### Video download fails

Make sure you have an internet connection and yt-dlp is up to date:
```
pip install --upgrade yt-dlp
```

### "Port 8000 already in use"

Another application (or a previous instance of Dawah-Translate) is using port 8000. Either close it or use a different port:
```
python server.py --port 8001
```
Then go to http://localhost:8001

### Application crashes with "out of memory"

The Whisper model and Ollama model together need about 6-8 GB of RAM. Close other applications to free up memory. You can also use smaller models:
- Whisper: select "medium" instead of "large-v3"
- Ollama: use a smaller model (not recommended for quality)

---

## 10. Glossary of Technical Terms

| Term | Meaning |
|------|---------|
| **SRT** | SubRip Text, a standard subtitle file format |
| **Whisper** | OpenAI's speech recognition model that converts audio to text |
| **NLLB-200** | Meta's translation model that supports 200 languages |
| **Ollama** | A tool that runs AI models locally on your computer |
| **gemma3:4b** | Google's 4-billion parameter AI model, used for refining translations |
| **FFmpeg** | A tool for processing video and audio files |
| **CPS** | Characters Per Second -- how fast the subtitle text must be read |
| **Tawheed** | Islamic monotheism (the oneness of Allah) |
| **Aqeedah** | Islamic creed/beliefs |
| **Translation Memory** | A database that remembers your corrections to improve future translations |

---

## Summary: What Needs to Be Running

For the application to work, you need **three things running**:

1. **Ollama** (the AI model server) -- usually starts automatically
2. **The Python server** (`python server.py`) -- started manually each time
3. **A web browser** pointed at http://localhost:8000

That's it. No internet needed after the initial setup (except to download new YouTube videos).
