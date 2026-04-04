"""Central configuration for dawah-translate pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"
FONTS_DIR = BASE_DIR / "fonts"
STATIC_DIR = BASE_DIR / "static"
GLOSSARY_PATH = BASE_DIR / "glossary.json"

# Ensure directories exist
JOBS_DIR.mkdir(exist_ok=True)
FONTS_DIR.mkdir(exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Subtitle constraints ──────────────────────────────────────────────
MAX_LINES_PER_BLOCK = 2
MAX_CHARS_PER_LINE = 42
MIN_DISPLAY_DURATION = 1.0       # seconds
MAX_DISPLAY_DURATION = 7.0       # seconds
MAX_READING_SPEED = 21           # characters per second
MIN_GAP_BETWEEN_SUBS = 0.080    # 80 milliseconds

# ── Whisper defaults ──────────────────────────────────────────────────
DEFAULT_WHISPER_MODEL = "large-v3"
WHISPER_CPU_THREADS = 8

# ── Translation defaults ─────────────────────────────────────────────
DEFAULT_CLAUDE_MODEL = "sonnet"
TRANSLATION_BATCH_SIZE = 50
TRANSLATION_CONTEXT_SIZE = 5     # previous segments sent as context
TRANSLATION_MAX_RETRIES = 3

# ── Claude model IDs ─────────────────────────────────────────────────
CLAUDE_MODELS = {
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}

# ── FFmpeg settings ──────────────────────────────────────────────────
FFMPEG_PRESET = "medium"
FFMPEG_CRF = 20

# ── Job cleanup ──────────────────────────────────────────────────────
JOB_MAX_AGE_DAYS = 7
