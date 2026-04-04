"""
SRT subtitle utilities: parsing, writing, segmentation, Romanian normalization, ASS conversion.

Functions:
    parse_srt(filepath) → list of segment dicts
    write_srt(segments, filepath)
    normalize_romanian(text) → text with correct diacritics
    check_reading_speed(segment) → chars/sec
    compute_max_chars(segment) → max chars for duration at 21 chars/sec
    resegment(segments) → split long segments to fit subtitle constraints
    srt_to_ass(srt_path, ass_path, font_dir) → convert SRT to ASS with style
"""

import re
from pathlib import Path

# ── Romanian diacritics normalization ─────────────────────────────────

def normalize_romanian(text: str) -> str:
    """
    Replace cedilla variants with correct comma-below variants.
    AI models frequently output the wrong Unicode codepoints for ș and ț.
    """
    replacements = {
        '\u015F': '\u0219',  # ş → ș
        '\u015E': '\u0218',  # Ş → Ș
        '\u0163': '\u021B',  # ţ → ț
        '\u0162': '\u021A',  # Ţ → Ț
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ── SRT timestamp parsing ────────────────────────────────────────────

def _parse_timestamp(ts: str) -> float:
    """Parse SRT timestamp 'HH:MM:SS,mmm' to seconds."""
    ts = ts.strip()
    match = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)', ts)
    if not match:
        raise ValueError(f"Invalid timestamp: {ts}")
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp 'HH:MM:SS,mmm'."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ── SRT parsing and writing ──────────────────────────────────────────

def parse_srt(filepath: str | Path) -> list[dict]:
    """
    Parse an SRT file into a list of segment dicts.

    Each segment: {"index": int, "start": float, "end": float, "text": str}
    """
    filepath = Path(filepath)
    # Try UTF-8 BOM first, then plain UTF-8
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            content = filepath.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError(f"Cannot decode SRT file: {filepath}")

    segments = []
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # Line 1: index number
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Line 2: timestamps
        ts_match = re.match(
            r'(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)',
            lines[1].strip()
        )
        if not ts_match:
            continue

        start = _parse_timestamp(ts_match.group(1))
        end = _parse_timestamp(ts_match.group(2))

        # Lines 3+: subtitle text
        text = '\n'.join(lines[2:]).strip()

        segments.append({
            "index": index,
            "start": start,
            "end": end,
            "text": text,
        })

    return segments


def write_srt(segments: list[dict], filepath: str | Path) -> None:
    """Write segments to an SRT file (UTF-8 with BOM)."""
    filepath = Path(filepath)
    with open(filepath, "w", encoding="utf-8-sig") as f:
        for i, seg in enumerate(segments, 1):
            start = _format_timestamp(seg["start"])
            end = _format_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


# ── Reading speed and character limits ────────────────────────────────

def check_reading_speed(segment: dict) -> float:
    """
    Calculate reading speed in characters per second.
    Returns chars/sec. Values > 21 are too fast to read.
    """
    duration = segment["end"] - segment["start"]
    if duration <= 0:
        return float('inf')
    # Count visible characters (exclude newlines and leading/trailing spaces)
    text = segment["text"].strip().replace('\n', '')
    return len(text) / duration


def compute_max_chars(segment: dict, max_cps: int = 21) -> int:
    """
    Compute the max number of characters that fit in this segment's
    time window at the given reading speed (default 21 chars/sec).
    """
    duration = segment["end"] - segment["start"]
    return int(duration * max_cps)


# ── Resegmentation ───────────────────────────────────────────────────

def resegment(segments: list[dict], max_chars_per_line: int = 42,
              max_lines: int = 2) -> list[dict]:
    """
    Split segments that are too long to fit subtitle constraints.

    Each output segment has at most max_lines lines, each at most
    max_chars_per_line characters. Splitting is done at word boundaries.
    """
    max_total_chars = max_chars_per_line * max_lines
    result = []

    for seg in segments:
        text = seg["text"].strip()
        # If it fits, keep as-is (but reflow into max 2 lines)
        if len(text.replace('\n', ' ')) <= max_total_chars:
            reflowed = _reflow_text(text.replace('\n', ' '), max_chars_per_line, max_lines)
            result.append({**seg, "text": reflowed})
            continue

        # Need to split into multiple subtitle blocks
        words = text.replace('\n', ' ').split()
        sub_segments = _split_words_into_blocks(words, max_chars_per_line, max_lines)

        # Distribute time proportionally across sub-segments
        total_chars = sum(len(s.replace('\n', '')) for s in sub_segments)
        if total_chars == 0:
            result.append(seg)
            continue

        duration = seg["end"] - seg["start"]
        current_time = seg["start"]

        for sub_text in sub_segments:
            char_ratio = len(sub_text.replace('\n', '')) / total_chars
            sub_duration = duration * char_ratio
            sub_end = min(current_time + sub_duration, seg["end"])

            result.append({
                "index": 0,  # Will be renumbered
                "start": current_time,
                "end": sub_end,
                "text": sub_text,
            })
            current_time = sub_end

    # Renumber
    for i, seg in enumerate(result, 1):
        seg["index"] = i

    return result


def _reflow_text(text: str, max_chars_per_line: int, max_lines: int) -> str:
    """Reflow text into max_lines lines, each at most max_chars_per_line."""
    if len(text) <= max_chars_per_line:
        return text

    # Split into 2 lines as evenly as possible
    words = text.split()
    best_split = len(words)
    best_diff = float('inf')

    for i in range(1, len(words)):
        line1 = ' '.join(words[:i])
        line2 = ' '.join(words[i:])
        if len(line1) <= max_chars_per_line and len(line2) <= max_chars_per_line:
            diff = abs(len(line1) - len(line2))
            if diff < best_diff:
                best_diff = diff
                best_split = i

    line1 = ' '.join(words[:best_split])
    line2 = ' '.join(words[best_split:])

    if line2:
        return f"{line1}\n{line2}"
    return line1


def _split_words_into_blocks(words: list[str], max_chars_per_line: int,
                              max_lines: int) -> list[str]:
    """Split words into multiple subtitle blocks, each fitting the constraints."""
    max_total = max_chars_per_line * max_lines
    blocks = []
    current_words = []
    current_len = 0

    for word in words:
        word_len = len(word)
        # +1 for the space before the word (unless it's the first)
        new_len = current_len + (1 if current_words else 0) + word_len

        if new_len > max_total and current_words:
            block_text = ' '.join(current_words)
            blocks.append(_reflow_text(block_text, max_chars_per_line, max_lines))
            current_words = [word]
            current_len = word_len
        else:
            current_words.append(word)
            current_len = new_len

    if current_words:
        block_text = ' '.join(current_words)
        blocks.append(_reflow_text(block_text, max_chars_per_line, max_lines))

    return blocks


# ── SRT → ASS conversion ─────────────────────────────────────────────

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans,56,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,30,30,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _to_ass_timestamp(seconds: float) -> str:
    """Convert seconds to ASS timestamp format: H:MM:SS.cc"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int(round((seconds % 1) * 100))
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def srt_to_ass(srt_path: str | Path, ass_path: str | Path,
               font_dir: str | Path = None) -> None:
    """
    Convert an SRT file to ASS format with the dawah-translate subtitle style.

    Args:
        srt_path: Path to input SRT file
        ass_path: Path to output ASS file
        font_dir: Path to fonts directory (unused in ASS content, used by FFmpeg)
    """
    segments = parse_srt(srt_path)

    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(ASS_HEADER)

        for seg in segments:
            start = _to_ass_timestamp(seg["start"])
            end = _to_ass_timestamp(seg["end"])
            # ASS uses \N for line breaks instead of \n
            text = seg["text"].replace('\n', '\\N')
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
