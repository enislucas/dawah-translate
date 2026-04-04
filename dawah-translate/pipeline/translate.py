"""
Translate subtitles to Romanian using Claude API with Islamic terminology glossary.

Usage:
    python pipeline/translate.py <job_id> [--claude-model sonnet]

Reads transcript_original.srt and produces transcript_romanian.srt.
"""

import sys
import json
import time
import argparse
import re
from pathlib import Path

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    JOBS_DIR, GLOSSARY_PATH, ANTHROPIC_API_KEY,
    CLAUDE_MODELS, DEFAULT_CLAUDE_MODEL,
    TRANSLATION_BATCH_SIZE, TRANSLATION_CONTEXT_SIZE,
    TRANSLATION_MAX_RETRIES, MAX_CHARS_PER_LINE, MAX_LINES_PER_BLOCK,
    MAX_READING_SPEED,
)
from pipeline.subtitle import (
    parse_srt, write_srt, normalize_romanian,
    check_reading_speed, compute_max_chars, resegment,
)


def load_glossary() -> dict:
    """Load the Islamic terminology glossary from glossary.json."""
    if not GLOSSARY_PATH.exists():
        print("  Warning: glossary.json not found, proceeding without glossary")
        return {}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def format_glossary_for_prompt(glossary: dict) -> str:
    """Format glossary as a readable block for the Claude system prompt."""
    if not glossary:
        return "(No glossary provided)"
    lines = []
    for arabic, info in glossary.items():
        trans = info["transliteration"]
        expl = info.get("ro_explanation", "")
        rule = info.get("rule", "")
        entry = f"  {arabic} → {trans}"
        if expl:
            entry += f" ({expl})"
        if rule:
            entry += f" | Rule: {rule}"
        lines.append(entry)
    return "\n".join(lines)


def build_system_prompt(source_language: str, glossary: dict,
                        introduced_terms: set) -> str:
    """Build the system prompt for Claude translation."""
    glossary_text = format_glossary_for_prompt(glossary)
    introduced_text = ", ".join(sorted(introduced_terms)) if introduced_terms else "(none yet)"

    return f"""You are a professional subtitle translator specializing in Islamic educational content for a Romanian-speaking audience.

TASK: Translate the following subtitle segments from {source_language} to Romanian.

RULES:
1. GLOSSARY: Use the provided glossary for Islamic terms. On first occurrence in this batch, include the Romanian explanation in parentheses. After first occurrence, use only the transliterated term. If a term was already introduced in a previous batch (marked with [INTRODUCED]), use only the transliterated term.
2. CONCISENESS: Each subtitle segment has a maximum character count shown in [max_chars]. Your translation MUST fit within this limit. If the original is verbose, condense — a subtitle is not a transcript.
3. NATURAL ROMANIAN: Use proper diacritics (ș, ț, ă, î, â) with comma-below, never cedilla. Write in natural, conversational Romanian — not formal/literary.
4. DO NOT over-arabicize. Only use Arabic/transliterated terms from the glossary. All other words must be in plain Romanian.
5. PRESERVE SEGMENT COUNT: Return exactly the same number of segments with the same index numbers.
6. LINE BREAKS: If a segment exceeds 42 characters, split into two lines with \\n. Max 2 lines.

GLOSSARY:
{glossary_text}

PREVIOUSLY INTRODUCED TERMS: {introduced_text}

INPUT FORMAT (one segment per line):
[index] [max_chars:n] | source_text

OUTPUT FORMAT (one segment per line, nothing else):
[index] | translated_text"""


def build_batch_input(segments: list[dict], context_segments: list[dict] = None) -> str:
    """Format a batch of segments for the Claude prompt."""
    lines = []

    # Add context from previous batch (not to be translated)
    if context_segments:
        lines.append("CONTEXT (do not translate, for reference only):")
        for seg in context_segments:
            lines.append(f"  [{seg['index']}] {seg['text']}")
        lines.append("")
        lines.append("TRANSLATE THE FOLLOWING:")

    for seg in segments:
        max_chars = compute_max_chars(seg)
        # Ensure minimum of 20 chars
        max_chars = max(max_chars, 20)
        lines.append(f"[{seg['index']}] [max_chars:{max_chars}] | {seg['text']}")

    return "\n".join(lines)


def parse_translation_response(response_text: str, expected_count: int) -> dict:
    """
    Parse Claude's response into a dict of {index: translated_text}.

    Expected format: [index] | translated_text
    """
    translations = {}
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'\[(\d+)\]\s*\|\s*(.+)', line)
        if match:
            idx = int(match.group(1))
            text = match.group(2).strip()
            translations[idx] = text

    return translations


def translate_batch(client, model_id: str, system_prompt: str,
                    segments: list[dict], context_segments: list[dict] = None,
                    max_retries: int = TRANSLATION_MAX_RETRIES) -> dict:
    """
    Send a batch of segments to Claude for translation.

    Returns dict of {index: translated_text}.
    Retries on failure with exponential backoff.
    """
    user_message = build_batch_input(segments, context_segments)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            response_text = response.content[0].text
            translations = parse_translation_response(response_text, len(segments))

            if len(translations) < len(segments):
                missing = [s["index"] for s in segments if s["index"] not in translations]
                print(f"  Warning: missing translations for segments {missing}")

            return translations

        except Exception as e:
            wait_time = 2 ** attempt * 5  # 5s, 10s, 20s
            print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


def translate_single_segment(segment: dict, source_language: str,
                             claude_model: str = None) -> str:
    """
    Translate a single segment via Claude API (used for re-translation in review UI).

    Returns the translated text.
    """
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")

    if claude_model is None:
        claude_model = DEFAULT_CLAUDE_MODEL
    model_id = CLAUDE_MODELS.get(claude_model, claude_model)

    glossary = load_glossary()
    system_prompt = build_system_prompt(source_language, glossary, set())

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user_message = build_batch_input([segment])

    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = response.content[0].text
    translations = parse_translation_response(response_text, 1)

    if segment["index"] in translations:
        text = normalize_romanian(translations[segment["index"]])
        return text

    # Fallback: return the raw response if parsing failed
    return normalize_romanian(response_text.strip())


def run_translation(job_id: str, claude_model: str = None) -> str:
    """
    Translate all segments for the given job.

    Args:
        job_id: The job identifier
        claude_model: "sonnet" or "opus"

    Returns:
        Path to the output Romanian SRT file.
    """
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env — add your key to the .env file")

    if claude_model is None:
        claude_model = DEFAULT_CLAUDE_MODEL
    model_id = CLAUDE_MODELS.get(claude_model, claude_model)

    # Load job info
    job_dir = JOBS_DIR / job_id
    info_path = job_dir / "job_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        job_info = json.load(f)

    srt_original = job_dir / "transcript_original.srt"
    srt_romanian = job_dir / "transcript_romanian.srt"

    if not srt_original.exists():
        raise FileNotFoundError(f"Original SRT not found: {srt_original}")

    # Parse original subtitles
    segments = parse_srt(srt_original)
    if not segments:
        raise ValueError("No segments found in the original SRT file")

    print(f"Job: {job_id}")
    print(f"Segments: {len(segments)}")
    print(f"Claude model: {claude_model} ({model_id})")
    print()

    # Resegment if needed (split long segments)
    original_count = len(segments)
    segments = resegment(segments)
    if len(segments) != original_count:
        print(f"  Resegmented: {original_count} → {len(segments)} segments")
        # Save resegmented original
        write_srt(segments, srt_original)

    # Update job status
    job_info["status"] = "translating"
    job_info["claude_model"] = claude_model
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    # Load glossary
    glossary = load_glossary()
    source_language = job_info.get("source_language", "auto")
    detected_language = job_info.get("detected_language", source_language)
    lang_name = {"ar": "Arabic", "en": "English"}.get(detected_language, detected_language)

    print(f"  Source language: {lang_name}")
    print(f"  Glossary terms: {len(glossary)}")
    print()

    # Initialize Claude client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Track introduced glossary terms across batches
    introduced_terms = set()

    # Translate in batches
    batch_size = TRANSLATION_BATCH_SIZE
    total_batches = (len(segments) + batch_size - 1) // batch_size
    translated_segments = []
    total_input_tokens = 0
    total_output_tokens = 0

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(segments))
        batch = segments[start:end]

        # Context: last N segments from previous batch
        context = []
        if batch_idx > 0 and translated_segments:
            context_start = max(0, len(translated_segments) - TRANSLATION_CONTEXT_SIZE)
            context = translated_segments[context_start:]

        print(f"  Batch {batch_idx + 1}/{total_batches} "
              f"(segments {start + 1}-{end} of {len(segments)})...")

        # Build system prompt with current introduced terms
        system_prompt = build_system_prompt(lang_name, glossary, introduced_terms)

        # Translate
        batch_start = time.time()
        translations = translate_batch(
            client, model_id, system_prompt, batch, context
        )
        batch_time = time.time() - batch_start

        # Apply translations
        for seg in batch:
            idx = seg["index"]
            if idx in translations:
                translated_text = normalize_romanian(translations[idx])
                translated_seg = {**seg, "text": translated_text}
            else:
                # Keep original if translation missing
                translated_seg = {**seg, "text": f"[UNTRANSLATED] {seg['text']}"}
                print(f"    Warning: segment {idx} not translated")

            translated_segments.append(translated_seg)

            # Track which glossary terms were introduced
            for arabic, info in glossary.items():
                trans = info["transliteration"]
                if trans.lower() in translated_text.lower():
                    introduced_terms.add(trans)

        print(f"    Done in {batch_time:.1f}s")

        # Save progress incrementally
        write_srt(translated_segments, srt_romanian)

    # Final validation
    print()
    print("Validating translations...")
    speed_warnings = 0
    for seg in translated_segments:
        speed = check_reading_speed(seg)
        if speed > MAX_READING_SPEED:
            speed_warnings += 1

    if speed_warnings:
        print(f"  Warning: {speed_warnings} segments exceed {MAX_READING_SPEED} chars/sec reading speed")
        print("  These will be highlighted in the review UI for editing")
    else:
        print("  All segments within reading speed limits")

    # Update job info
    job_info["status"] = "translated"
    job_info["srt_romanian_path"] = str(srt_romanian)
    job_info["translated_segment_count"] = len(translated_segments)
    job_info["speed_warnings"] = speed_warnings
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    print()
    print(f"Done! {len(translated_segments)} segments translated → {srt_romanian}")
    return str(srt_romanian)


# ── Standalone entry point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate subtitles with Claude")
    parser.add_argument("job_id", help="Job ID to translate")
    parser.add_argument("--claude-model", default=None, choices=["sonnet", "opus"],
                        help="Claude model (default: sonnet)")
    args = parser.parse_args()

    print("=" * 60)
    print("DAWAH-TRANSLATE — Subtitle Translation")
    print("=" * 60)

    try:
        srt_path = run_translation(args.job_id, args.claude_model)
        print()
        print(f"Next step: Review at http://localhost:8000/review/{args.job_id}")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
