"""
Translate subtitles to Romanian using Claude API — Two-Layer Architecture (V3).

Four-step pipeline:
    Step 1: Summary pass — analyze full SRT, build glossary, flag errors, translate title
    Step 2: Sliding-window translation — 50-segment windows, focus on accuracy
    Step 3: Quality review pass — full polish and error correction
    Step 4: Consistency pass — (optional, for long videos) check term consistency

Usage:
    python pipeline/translate.py <job_id> [--model sonnet]

Reads transcript_original.srt and produces transcript_romanian.srt.
"""

import sys
import json
import logging
import math
import re
import time
import asyncio
import argparse
from pathlib import Path

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    JOBS_DIR, GLOSSARY_PATH, ANTHROPIC_API_KEY,
    CLAUDE_MODELS, DEFAULT_CLAUDE_MODEL,
    TRANSLATION_MAX_RETRIES,
    WINDOW_SIZE, WINDOW_OVERLAP,
    REVIEW_WINDOW_SIZE, REVIEW_WINDOW_OVERLAP,
    MAX_READING_SPEED,
)
from pipeline.subtitle import (
    parse_srt, write_srt, normalize_romanian,
    check_reading_speed, resegment, merge_micro_segments,
)

logger = logging.getLogger("dawah.translate")

# ── Progress callback type ───────────────────────────────────────────
# progress_callback(step, progress_pct, message)
# Called by run_translation so the server can push SSE events.


def load_glossary() -> dict:
    """Load the Islamic terminology glossary from glossary.json."""
    if not GLOSSARY_PATH.exists():
        logger.warning("glossary.json not found, proceeding without glossary")
        return {}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def format_glossary_for_prompt(glossary: dict) -> str:
    """Format glossary as a readable block for Claude prompts."""
    if not glossary:
        return "(No glossary provided)"
    lines = []
    for arabic, info in glossary.items():
        trans = info["transliteration"]
        expl = info.get("ro_explanation", "")
        rule = info.get("rule", "")
        entry = f"  {arabic} -> {trans}"
        if expl:
            entry += f" ({expl})"
        if rule:
            entry += f" | Rule: {rule}"
        lines.append(entry)
    return "\n".join(lines)


def _format_srt_for_analysis(segments: list[dict]) -> str:
    """Format all segments as numbered lines for the summary pass."""
    lines = []
    for seg in segments:
        lines.append(f"[{seg['index']}] ({seg['start']:.1f}-{seg['end']:.1f}s) {seg['text']}")
    return "\n".join(lines)


# ── Step 1: Summary Pass ─────────────────────────────────────────────

def _build_step1_prompt(source_language: str, glossary: dict, video_title: str) -> str:
    """System prompt for the full-document analysis pass."""
    glossary_text = format_glossary_for_prompt(glossary)
    return f"""You are a knowledgeable Muslim translator who understands Islamic theology (aqeedah), jurisprudence (fiqh), and the terminology of classical Islamic scholarship.

You follow the creed and methodology of Ahl as-Sunnah wal-Jama'ah, in the tradition of scholars such as Ibn Taymiyyah, Ibn al-Qayyim, Ibn Katheer, Muhammad ibn Abdul-Wahhab, Sheikh Ibn Baz, Sheikh al-Uthaymeen, Sheikh al-Albani, Sheikh al-Fawzan, and Sheikh Abdur-Razzaq al-Badr.

TASK: Analyze the following auto-transcribed subtitle file ({source_language}) and produce a structured analysis.

## GLOSSARY (reference)
{glossary_text}

## OUTPUT FORMAT (respond in EXACTLY this JSON structure, nothing else):
{{
  "speaker": "Name or description of the primary speaker",
  "topic": "Brief topic summary (1-2 sentences)",
  "content_summary": "3-5 sentence summary of the full content",
  "detected_terms": {{
    "arabic_term": "recommended Romanian translation"
  }},
  "transcription_errors": [
    {{
      "segment_index": 0,
      "issue": "description of the problem",
      "likely_correct": "what the audio probably said"
    }}
  ],
  "title_romanian": "Natural Romanian translation of the video title"
}}

## TITLE TRANSLATION
The original video title is: "{video_title}"
Translate into natural Romanian — should sound like a native Romanian speaker wrote it.
Not a literal translation. Concise, clear, suitable as a video title.

## TRANSCRIPTION ERROR DETECTION
The source was auto-transcribed by Whisper. Be CONSERVATIVE — only flag segments that are truly unintelligible:
- Non-Arabic words injected mid-sentence (English "refresh", city names like "Istanbul" that don't belong)
- Random characters or completely nonsensical sequences
- Clear code-switching artifacts
Do NOT flag slightly corrupted but inferable Arabic. For example, "السم" is clearly "السنة", "بائل" is clearly "مباين" — these should be translated with best effort, not flagged.
Only include genuinely broken segments in transcription_errors."""


def run_step1_summary(client, model_id: str, segments: list[dict],
                      source_language: str, glossary: dict, video_title: str,
                      progress_cb=None) -> dict:
    """
    Step 1: Send the full SRT for analysis.
    Returns the parsed summary dict.
    """
    if progress_cb:
        progress_cb("translate", 0, "Stage 1/4: Analyzing document...")

    system_prompt = _build_step1_prompt(source_language, glossary, video_title)
    user_message = _format_srt_for_analysis(segments)

    logger.info("Step 1: Sending %d segments for analysis...", len(segments))

    response = _call_claude(client, model_id, system_prompt, user_message, max_tokens=4096)

    # Parse JSON response — Claude sometimes wraps in ```json blocks
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        summary = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Step 1: Could not parse JSON response, using defaults")
        summary = {
            "speaker": "Unknown",
            "topic": "Islamic lecture",
            "content_summary": "",
            "detected_terms": {},
            "transcription_errors": [],
            "title_romanian": video_title,
        }

    logger.info("Step 1 complete: speaker=%s, flagged_errors=%d, terms=%d",
                summary.get("speaker", "?"),
                len(summary.get("transcription_errors", [])),
                len(summary.get("detected_terms", {})))

    if progress_cb:
        n_errors = len(summary.get("transcription_errors", []))
        n_terms = len(summary.get("detected_terms", {}))
        progress_cb("translate", 5, f"Stage 1/4 done: {n_terms} terms, {n_errors} flagged errors")

    return summary


# ── Step 2: Sliding-Window Translation ───────────────────────────────

def _build_step2_system_prompt(source_language: str, glossary: dict,
                               summary: dict) -> str:
    """Short, accuracy-focused system prompt for Stage 2 (raw translation)."""
    glossary_text = format_glossary_for_prompt(glossary)

    errors_text = "(none detected)"
    errors = summary.get("transcription_errors", [])
    if errors:
        error_lines = []
        for e in errors:
            error_lines.append(f"  Segment {e['segment_index']}: {e['issue']} (likely: {e.get('likely_correct', '?')})")
        errors_text = "\n".join(error_lines)

    return f"""You are translating Islamic lecture subtitles from {source_language} into Romanian.

You follow the methodology of Ahl as-Sunnah wal-Jamā'ah (Ibn Taymiyyah, Ibn Baz, al-Uthaymeen, al-Albani, al-Fawzan).

DOCUMENT CONTEXT:
Speaker: {summary.get('speaker', 'Unknown')}
Topic: {summary.get('topic', 'Islamic lecture')}
Summary: {summary.get('content_summary', '')}

FLAGGED TRANSCRIPTION ERRORS (from analysis pass):
{errors_text}

RULES — your only job is ACCURACY:

1. Translate EVERY segment. Never output [EROARE_TRANSCRIERE] or [NETRADUS] or a blank — only use the ~ prefix when you genuinely cannot decode the text, and always follow ~ with a real Romanian guess. Dialect Arabic (مش, ايش, بس, شوف, وش, ليش) is NORMAL — translate it.

2. Get the meaning RIGHT. Use the surrounding context (previous translations + lookahead) to make sure your output makes sense in the flow.

3. Glossary terms (use exactly):
{glossary_text}
   First occurrence of a glossary term: transliteration + Romanian explanation in parentheses. After that: transliteration only.

4. WELL-KNOWN DU'AS AND QURAN: When the speaker quotes a known du'a, hadith, or Quran verse, translate the MEANING into Romanian. Do not transliterate Arabic du'as. Examples:
   - "اللهم اجعلني خيرا مما يظنون واغفر لي ما لا يعلمون" → "O, Allah, fă-mă mai bun decât cred ei despre mine și iartă-mi ceea ce ei nu știu"
   - "لا حول ولا قوة إلا بالله" → "Nu există putere și forță decât prin Allah"
   - "الحمد لله" → "Alhamdulillah" (this one stays transliterated — fixed phrase)

5. SINGULAR vs PLURAL: When the speaker talks about HIMSELF (Arabic أنا, ـي, my), use Romanian SINGULAR ("eu", "mă", "îmi", "meu"). When about a group (نحن, ـنا), use plural. Pay close attention — do not turn singular into plural.

6. WHISPER GARBLES well-known phrases. Recognize them:
   - "أحول الأقوى تسنى منه" / similar → almost certainly "لا حول ولا قوة إلا بالله"
   - "يمكن أنتم أفقر مني" → likely "يمكن أنتم أعلم مني" ("perhaps you know more than me")
   - "بسم الله الرحمن الرحيم" can come through as fragments — recognize and restore
   Translate the INTENDED phrase, not the garbled surface text.

7. NO CONTENT LOSS: every proper name, place, date, and number in the source must appear somewhere in the translation. If "يوم خيبر" is in the source, "Khaybar" must be in your output.

8. ISLAMIC FORMULA DETECTION: When Whisper transcribes Arabic formulas as English transliteration (e.g., "Bismillah ar-Rahman ar-Rahim", "Alhamdulillahi Rabbil Alameen", "SubhanAllah", "Allahu Akbar", "La hawla wa la quwwata illa billah", "JazakAllahu khairan"), recognize these as known Islamic formulas and translate according to the glossary rules:
   First occurrence: Transliteration (Explicație română)
   Example: "Bismillah ir-Rahman ir-Rahim (În numele lui Allah, Cel Milostiv, Cel Îndurător)"
   Subsequent occurrences: Just transliteration
   Example: "Bismillah ir-Rahman ir-Rahim"
   NEVER leave these as raw English transliteration without at least the standard transliteration form.

9. SALAWAT / ﷺ SYMBOL: When the speaker says ANY of these variations, render it as the ﷺ symbol in Romanian output:
   - "sallallahu alayhi wa sallam", "sallallahu alaihi wasallam", "sala Allahu alayhi wa sallam"
   - "peace be upon him", "PBUH", "peace and blessings be upon him"
   - "صلى الله عليه وسلم", "عليه الصلاة والسلام"
   Do NOT transliterate it as "sallallahu alayhi wa sallam" and do NOT translate it as "pacea fie asupra lui" every time — just use the ﷺ symbol.
   Example: "The Prophet ﷺ said..." → "Profetul ﷺ a spus..."

10. QURAN VERSE HANDLING: When the speaker directly quotes a Quran verse in Arabic:
   - If you KNOW the exact surah and ayah (you are certain), tag it: {{index}}|[QURAN:{{surah}}:{{ayah}}] {{arabic_text}} — {{romanian_translation}}
   - If you recognize it as Quran but aren't sure of the exact reference, still translate the MEANING into Romanian. Just don't add the [QURAN:X:Y] tag.
   - NEVER leave a Quran quote as English transliteration — always provide the Romanian meaning.
   When the speaker PARAPHRASES Quran in English (e.g., "Allah says in the Quran that..."), just translate normally into Romanian. No tag needed.
   It is BETTER to tag a verse you're certain about than to tag nothing. Only skip the tag when you're genuinely unsure of the reference.

OUTPUT FORMAT (one segment per line, nothing else):
{{index}}|{{romanian translation}}

Do not output explanations, headers, or commentary. Translate every active segment. Do not translate context or lookahead segments — those are for reference only."""



def _build_step2_user_message(context_segments: list[dict],
                              active_segments: list[dict],
                              lookahead_segments: list[dict]) -> str:
    """Build the user message for a translation window."""
    parts = []

    if context_segments:
        parts.append("## CONTEXT (already translated -- DO NOT re-translate, just maintain consistency):")
        for seg in context_segments:
            parts.append(f"[{seg['index']}] {seg['text']}")
        parts.append("")

    parts.append("## TRANSLATE THESE SEGMENTS:")
    for seg in active_segments:
        parts.append(f"[{seg['index']}] {seg['text']}")
    parts.append("")

    if lookahead_segments:
        parts.append("## LOOKAHEAD (upcoming source text -- read for context, DO NOT translate):")
        for seg in lookahead_segments:
            parts.append(f"[{seg['index']}] {seg['text']}")

    return "\n".join(parts)


def _parse_window_response(response_text: str) -> dict:
    """Parse windowed translation response into {index: translated_text}."""
    translations = {}
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(r'(\d+)\s*\|\s*(.+)', line)
        if match:
            idx = int(match.group(1))
            text = match.group(2).strip()
            translations[idx] = text
    return translations


def is_likely_romanian(text: str) -> bool:
    """Quick check: Romanian text should contain diacritics or common Romanian words.

    Strips ~ prefix before checking. Skips segments under 10 visible chars.
    """
    clean = text.lstrip('~').strip()
    if len(clean) < 10:
        return True  # Single words — too short to detect reliably
    romanian_indicators = [
        'și', 'că', 'este', 'sunt', 'pentru', 'care', 'acest', 'aceasta',
        'din', 'sau', 'dar', 'mai', 'cum', 'când', 'unde', 'cine',
        'deci', 'apoi', 'între', 'prin', 'către', 'despre', 'peste',
        'lui', 'cele', 'doar', 'astfel',
    ]
    text_lower = clean.lower()
    matches = sum(1 for word in romanian_indicators if word in text_lower)
    if matches >= 2 or any(c in clean for c in 'șțăîâ'):
        return True
    # If it looks English, it's definitely not Romanian
    if _is_likely_english(clean):
        return False
    # Ambiguous — could be short Romanian without diacritics. Give benefit of doubt
    # only for shorter texts (under 40 chars). Longer texts should have indicators.
    return len(clean) < 40


def _is_likely_english(text: str) -> bool:
    """Detect English text that should never appear in Romanian output."""
    english_indicators = [
        'the ', 'and ', 'this ', 'that ', 'you ', 'they ',
        'with ', 'from ', 'because ', 'which ', 'have ',
        'what ', 'about ', 'when ', 'there ', 'their ',
        'would ', 'should ', 'could ', 'being ', "you're ",
        "don't ", "isn't ", "it's ", 'longer ',
    ]
    text_lower = text.lower()
    matches = sum(1 for word in english_indicators if word in text_lower)
    return matches >= 3


def _count_translated(window_translations: dict, active_segments: list[dict]) -> int:
    """Count how many active segments got a valid translation."""
    count = 0
    for seg in active_segments:
        if seg["index"] in window_translations:
            text = window_translations[seg["index"]]
            if text and len(text.strip()) > 0:
                count += 1
    return count


def _translate_window_with_retry(client, model_id: str, system_prompt: str,
                                 segments: list[dict], context: list[dict],
                                 lookahead: list[dict],
                                 glossary: dict, introduced_terms: set,
                                 translated: list[dict], offset: int,
                                 win_label: str = "") -> None:
    """Translate a window with automatic retry on failure.

    On failure (< 80% segments translated), retries with two half-windows.
    On double failure, falls back to micro-windows of 10 segments.

    Mutates `translated` in place and updates `introduced_terms`.
    """
    window_size = len(segments)

    def _apply_translations(trans_dict: dict, active: list[dict], base_offset: int):
        """Apply a translation dict to the translated list."""
        for seg in active:
            idx = seg["index"]
            if idx in trans_dict:
                text = normalize_romanian(trans_dict[idx])
                # Check for untranslated text (still in source language)
                if len(text) > 20 and not is_likely_romanian(text):
                    logger.warning("Seg %d appears untranslated (not Romanian): %.60s", idx, text)
                translated[base_offset + (idx - active[0]["index"])]["text"] = text
                # Track glossary introductions
                for arabic, info in glossary.items():
                    trans = info["transliteration"]
                    if trans.lower() in text.lower():
                        introduced_terms.add(trans)
            else:
                translated[base_offset + (idx - active[0]["index"])]["text"] = f"~{seg['text']}"
                logger.warning("Segment %d not in response", idx)

    # ── Attempt 1: full window ──────────────────────────────────────
    user_message = _build_step2_user_message(context, segments, lookahead)
    response_text, stop_reason, in_tokens, out_tokens = _call_claude(
        client, model_id, system_prompt, user_message,
        max_tokens=8192, return_metadata=True,
    )
    window_translations = _parse_window_response(response_text)
    translated_count = _count_translated(window_translations, segments)

    if translated_count >= window_size * 0.8:
        # Success — apply and return
        _apply_translations(window_translations, segments, offset)
        logger.info("%s: %d/%d segments translated (stop=%s, in=%d, out=%d)",
                    win_label, translated_count, window_size,
                    stop_reason, in_tokens, out_tokens)
        return

    # ── Diagnostic logging for failed window ────────────────────────
    logger.warning(
        "%s FAILED: only %d/%d segments parsed. stop_reason=%s, "
        "tokens_in=%d, tokens_out=%d",
        win_label, translated_count, window_size,
        stop_reason, in_tokens, out_tokens,
    )
    logger.warning("Raw response (first 500 chars): %.500s", response_text)

    # ── Attempt 2: split into two halves ────────────────────────────
    mid = window_size // 2
    first_half = segments[:mid]
    second_half = segments[mid:]

    logger.info("%s: Retrying as two half-windows (%d + %d segments)",
                win_label, mid, window_size - mid)

    time.sleep(2)  # Rate limit protection

    # First half: context = original context, lookahead = start of second half
    user_a = _build_step2_user_message(context, first_half, segments[mid:mid + 10])
    resp_a, stop_a, in_a, out_a = _call_claude(
        client, model_id, system_prompt, user_a,
        max_tokens=8192, return_metadata=True,
    )
    trans_a = _parse_window_response(resp_a)
    count_a = _count_translated(trans_a, first_half)

    time.sleep(2)

    # Second half: context = end of first half (translated), lookahead = original lookahead
    # Use the first-half translations as context if available
    half_context = []
    for seg in first_half[-10:]:
        idx = seg["index"]
        if idx in trans_a:
            half_context.append({"index": idx, "text": trans_a[idx]})
        else:
            half_context.append(seg)

    user_b = _build_step2_user_message(half_context, second_half, lookahead[:10])
    resp_b, stop_b, in_b, out_b = _call_claude(
        client, model_id, system_prompt, user_b,
        max_tokens=8192, return_metadata=True,
    )
    trans_b = _parse_window_response(resp_b)
    count_b = _count_translated(trans_b, second_half)

    total_retry = count_a + count_b
    logger.info("%s half-retry: %d+%d = %d/%d (stop=%s/%s)",
                win_label, count_a, count_b, total_retry, window_size,
                stop_a, stop_b)

    if total_retry >= window_size * 0.8:
        # Apply both halves
        _apply_translations(trans_a, first_half, offset)
        _apply_translations(trans_b, second_half, offset + mid)
        return

    # ── Attempt 3: micro-windows of 10 segments ────────────────────
    logger.warning("%s: Half-windows also failed (%d/%d). "
                   "Falling back to micro-windows of 10.",
                   win_label, total_retry, window_size)

    for i in range(0, window_size, 10):
        micro_batch = segments[i:i + 10]
        # Context: 5 segments before this micro-batch
        if i > 0:
            ctx = segments[max(0, i - 5):i]
            # Try to use already-translated text for context
            micro_ctx = []
            for seg in ctx:
                t_idx = offset + (seg["index"] - segments[0]["index"])
                micro_ctx.append(translated[t_idx])
        else:
            micro_ctx = context[-5:] if context else []

        # Lookahead: 5 segments after this micro-batch
        if i + 10 < window_size:
            micro_look = segments[i + 10:i + 15]
        else:
            micro_look = lookahead[:5]

        time.sleep(1)  # Rate limit

        user_m = _build_step2_user_message(micro_ctx, micro_batch, micro_look)
        resp_m = _call_claude(client, model_id, system_prompt, user_m, max_tokens=4096)
        trans_m = _parse_window_response(resp_m)

        micro_count = _count_translated(trans_m, micro_batch)
        logger.info("%s micro %d-%d: %d/%d segments",
                    win_label, i, i + len(micro_batch), micro_count, len(micro_batch))

        _apply_translations(trans_m, micro_batch, offset + i)


def run_step2_translation(client, model_id: str, segments: list[dict],
                          source_language: str, glossary: dict, summary: dict,
                          progress_cb=None) -> list[dict]:
    """
    Step 2: Translate in sliding windows (Layer 1 — accuracy focus).
    Returns list of translated segment dicts.
    """
    system_prompt = _build_step2_system_prompt(source_language, glossary, summary)

    window_size = WINDOW_SIZE
    overlap = WINDOW_OVERLAP
    num_windows = max(1, math.ceil(len(segments) / window_size))

    logger.info("Step 2: Translating %d segments in %d windows (size=%d, overlap=%d)",
                len(segments), num_windows, window_size, overlap)

    # Build translated segments list — start as copies of original
    translated = [{**seg} for seg in segments]
    # Track which terms have been introduced (for glossary first-occurrence logic)
    introduced_terms = set()

    for win_idx in range(num_windows):
        # Inter-window delay to prevent 429 rate limiting (Tier 2: 90K output tokens/min)
        if win_idx > 0:
            time.sleep(2)

        start = win_idx * window_size
        end = min(start + window_size, len(segments))
        active = segments[start:end]

        # Context: already-translated segments before the window
        ctx_start = max(0, start - overlap)
        context = translated[ctx_start:start] if start > 0 else []

        # Lookahead: original segments after the window
        look_end = min(end + overlap, len(segments))
        lookahead = segments[end:look_end] if end < len(segments) else []

        # Progress: Stage 2 occupies 5-60% range
        pct = int(5 + (win_idx / num_windows) * 55)
        msg = f"Stage 2/4: Translating window {win_idx + 1}/{num_windows} (segments {start + 1}-{end})..."
        logger.info(msg)
        if progress_cb:
            progress_cb("translate", pct, msg)

        win_label = f"Window {win_idx + 1}/{num_windows}"
        _translate_window_with_retry(
            client, model_id, system_prompt,
            active, context, lookahead,
            glossary, introduced_terms,
            translated, start,
            win_label=win_label,
        )

    if progress_cb:
        progress_cb("translate", 60, "Stage 2/4 complete. Starting quality review...")

    return translated


# ── Step 3: Quality Review Pass (Layer 2) ────────────────────────────

def _build_step3_review_prompt(source_language: str, glossary: dict) -> str:
    """Short, polish-only system prompt for Stage 3 (Romanian quality review)."""
    return f"""You are reviewing a Romanian subtitle translation for naturalness and grammar.

The translation is already accurate — DO NOT change the meaning. Only fix how it sounds in Romanian.

FIX THESE PATTERNS:
1. Stiff phrasing: "Aceasta este falsă" → "Așa ceva este fals". Use natural spoken Romanian, not textbook.
2. Dangling pronouns: if "-l", "-o", "lui", "ei" has no clear antecedent in the same or previous subtitle, replace with the actual noun or "acest lucru" / "aceasta".
3. Heavy relative clauses: "pe care nu îl cunoaște" → "ce nu cunoaște". Simplify.
4. Contractions: use "n-are", "nu-i", "e" instead of "este" where natural.
5. Repetition: don't reuse the same connector ("aceasta", "pentru că", "atunci") in consecutive segments.
6. Arabic verbal nouns: "evitarea lui" → "a evita acest lucru".
7. Singular/plural: if the speaker is talking about himself (eu), keep singular — never inflate to "noi"/"noastră".

8. English transliteration of Arabic: if you see Arabic transliteration in Latin script (e.g., "fatawakkal ala Allah", "innahu alhaqqun", "wa quli alhaqqu min rabbikum", "innaka ala alhaqqil mubeen") that has NOT been translated into Romanian, translate the MEANING into Romanian. A Romanian viewer cannot read Latin-script Arabic — replace it with the Romanian meaning. Also: any remaining English text (e.g., "you have home you're no longer...") must be translated into Romanian — English should never appear in the final output.

DO NOT change:
- Theological terms or glossary words
- The MEANING of any segment
- Proper nouns and names (people, places, books)
- Quran/hadith/du'a quotes that are already correctly translated into Romanian
- Diacritics that are already correct

OUTPUT FORMAT (one segment per line):
{{index}}|{{polished text}}

Return EVERY segment, even unchanged ones. No explanations, no commentary."""


def _build_step3_review_user_message(original_segments: list[dict],
                                     translated_segments: list[dict],
                                     context_original: list[dict] = None,
                                     context_translated: list[dict] = None) -> str:
    """Build the user message for a review window."""
    parts = []

    if context_original and context_translated:
        parts.append("## CONTEXT (already reviewed -- for flow reference only):")
        for orig, trans in zip(context_original, context_translated):
            parts.append(f"[{orig['index']}] {orig['text']}  =>  {trans['text']}")
        parts.append("")

    parts.append("## REVIEW THESE SEGMENTS:")
    parts.append("")
    parts.append(f"### ORIGINAL:")
    for seg in original_segments:
        parts.append(f"[{seg['index']}] {seg['text']}")
    parts.append("")
    parts.append(f"### DRAFT TRANSLATION:")
    for seg in translated_segments:
        parts.append(f"[{seg['index']}] {seg['text']}")

    return "\n".join(parts)


def run_step3_review(client, model_id: str,
                     original_segments: list[dict],
                     translated_segments: list[dict],
                     source_language: str, glossary: dict,
                     progress_cb=None) -> list[dict]:
    """
    Step 3: Quality review pass (Layer 2 — polish and error correction).
    Sends original + draft to Claude for review. Uses windowing for long videos.
    Returns the reviewed/corrected translated segments.
    """
    system_prompt = _build_step3_review_prompt(source_language, glossary)
    num_segments = len(translated_segments)

    # Make a working copy
    reviewed = [{**seg} for seg in translated_segments]

    if num_segments <= 80:
        # Single call — send everything
        logger.info("Step 3: Reviewing all %d segments in single call", num_segments)
        if progress_cb:
            progress_cb("translate", 65, "Stage 3/4: Polishing Romanian (all segments)...")

        user_message = _build_step3_review_user_message(original_segments, translated_segments)
        response_text = _call_claude(client, model_id, system_prompt, user_message,
                                     max_tokens=8192)
        corrections = _parse_window_response(response_text)

        applied = 0
        for seg in reviewed:
            if seg["index"] in corrections:
                new_text = normalize_romanian(corrections[seg["index"]])
                if new_text != seg["text"]:
                    seg["text"] = new_text
                    applied += 1

        logger.info("Step 3: Applied %d corrections out of %d segments", applied, num_segments)

    else:
        # Windowed review for long videos
        window_size = REVIEW_WINDOW_SIZE
        overlap = REVIEW_WINDOW_OVERLAP
        num_windows = max(1, math.ceil(num_segments / window_size))

        logger.info("Step 3: Reviewing %d segments in %d windows (size=%d, overlap=%d)",
                    num_segments, num_windows, window_size, overlap)

        total_applied = 0
        for win_idx in range(num_windows):
            start = win_idx * window_size
            end = min(start + window_size, num_segments)

            active_orig = original_segments[start:end]
            active_trans = translated_segments[start:end]

            # Context: already-reviewed segments before the window
            ctx_start = max(0, start - overlap)
            ctx_orig = original_segments[ctx_start:start] if start > 0 else None
            ctx_trans = reviewed[ctx_start:start] if start > 0 else None

            # Progress: Stage 3 occupies 60-95% range
            pct = int(60 + (win_idx / num_windows) * 35)
            msg = f"Stage 3/4: Polishing window {win_idx + 1}/{num_windows} (segments {start + 1}-{end})..."
            logger.info(msg)
            if progress_cb:
                progress_cb("translate", pct, msg)

            user_message = _build_step3_review_user_message(
                active_orig, active_trans, ctx_orig, ctx_trans
            )
            response_text = _call_claude(client, model_id, system_prompt, user_message,
                                         max_tokens=8192)
            corrections = _parse_window_response(response_text)

            applied = 0
            for seg in reviewed[start:end]:
                if seg["index"] in corrections:
                    new_text = normalize_romanian(corrections[seg["index"]])
                    if new_text != seg["text"]:
                        seg["text"] = new_text
                        applied += 1

            total_applied += applied
            logger.info("  Window %d: %d corrections applied", win_idx + 1, applied)

        logger.info("Step 3: Total %d corrections across %d windows", total_applied, num_windows)

    if progress_cb:
        progress_cb("translate", 95, "Stage 3/4 complete.")

    return reviewed


# ── Stage 4: Subtitle Validation (Python only — no LLM) ──────────────

def _reflow_two_lines(text: str, max_chars: int = 42) -> str:
    """Reflow text into at most 2 lines, splitting at the closest space to midpoint."""
    flat = text.replace("\\N", " ").replace("\n", " ")
    flat = " ".join(flat.split())  # collapse whitespace
    if len(flat) <= max_chars:
        return flat
    # Find best split near the middle
    mid = len(flat) // 2
    best = None
    for offset in range(min(mid, 30)):
        if mid - offset > 0 and flat[mid - offset] == " ":
            best = mid - offset
            break
        if mid + offset < len(flat) and flat[mid + offset] == " ":
            best = mid + offset
            break
    if best is None:
        return flat  # no space found — leave as-is
    return flat[:best].rstrip() + "\\N" + flat[best:].lstrip()


def validate_subtitles(segments: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Stage 4: Deterministic subtitle validation. No LLM calls.

    - Normalizes diacritics (cedilla → comma-below)
    - Reflows segments with 3+ lines down to 2 lines
    - Warns when a single line exceeds 42 chars
    - Warns when reading speed exceeds 21 chars/sec

    Returns (segments, warnings).
    """
    warnings: list[str] = []

    for seg in segments:
        text = normalize_romanian(seg["text"])

        # Reflow if 3+ lines
        lines = text.split("\\N") if "\\N" in text else text.split("\n")
        if len(lines) > 2:
            text = _reflow_two_lines(text)
            lines = text.split("\\N")

        # Per-line char check (warning only — adaptive font handles overflow)
        for ln in lines:
            if len(ln) > 42:
                warnings.append(
                    f"Seg {seg['index']}: line exceeds 42 chars ({len(ln)})"
                )

        # Reading-speed check
        duration = max(seg["end"] - seg["start"], 0.1)
        visible = text.replace("\\N", "").replace("\n", "")
        cps = len(visible) / duration
        if cps > MAX_READING_SPEED:
            warnings.append(
                f"Seg {seg['index']}: reading speed {cps:.1f} cps (max {MAX_READING_SPEED})"
            )

        seg["text"] = text

    return segments, warnings


# ── Claude API helper ────────────────────────────────────────────────

def _call_claude(client, model_id: str, system_prompt: str, user_message: str,
                 max_tokens: int = 4096,
                 max_retries: int = TRANSLATION_MAX_RETRIES,
                 return_metadata: bool = False) -> str | tuple:
    """Call Claude API with retries and exponential backoff.

    Args:
        return_metadata: If True, returns (text, stop_reason, input_tokens, output_tokens)
                         instead of just text.

    Returns response text, or tuple with metadata if return_metadata=True.
    """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text
            if return_metadata:
                return (
                    text,
                    response.stop_reason,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )
            return text
        except Exception as e:
            wait_time = 2 ** attempt * 5  # 5s, 10s, 20s
            logger.error("API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                logger.info("Retrying in %ds...", wait_time)
                time.sleep(wait_time)
            else:
                raise


# ── Single-segment retranslation (for review UI) ────────────────────

def translate_single_segment(segment: dict, source_language: str,
                             claude_model: str = None,
                             prev_original: list[dict] = None,
                             prev_translated: list[dict] = None,
                             next_original: list[dict] = None,
                             summary: dict = None,
                             introduced_terms: set = None) -> str:
    """
    Translate a single segment via Claude API with full context.

    Sends: system prompt (built from full glossary + summary) + N previous
    translated segments (context) + the target segment + N next original
    segments (lookahead). The system prompt is also annotated with which
    glossary terms have already been introduced earlier in the video so
    Claude knows whether to use the first-occurrence explanation or just
    the transliteration.

    Returns the translated text for the single segment.
    """
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")

    if claude_model is None:
        claude_model = DEFAULT_CLAUDE_MODEL
    model_id = CLAUDE_MODELS.get(claude_model, claude_model)

    glossary = load_glossary()

    if summary is None:
        summary = {"speaker": "Unknown", "topic": "Islamic lecture",
                   "content_summary": "", "detected_terms": {},
                   "transcription_errors": []}

    system_prompt = _build_step2_system_prompt(source_language, glossary, summary)

    # Annotate system prompt with already-introduced glossary terms so Claude
    # knows whether to repeat the first-occurrence explanation.
    if introduced_terms:
        intro_list = ", ".join(sorted(introduced_terms))
        system_prompt += (
            "\n\n## ALREADY-INTRODUCED GLOSSARY TERMS\n"
            f"The following terms have already appeared earlier in this video and were "
            f"introduced with their Romanian explanation: {intro_list}.\n"
            f"For these terms, use ONLY the transliteration (no parenthetical explanation). "
            f"Treat any other glossary term as a first occurrence."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build context-aware user message: previous (translated) + active + lookahead (original)
    context_segs = []
    if prev_original and prev_translated:
        for orig, trans in zip(prev_original, prev_translated):
            context_segs.append({"index": orig["index"], "text": trans["text"]})

    user_message = _build_step2_user_message(
        context_segments=context_segs,
        active_segments=[segment],
        lookahead_segments=next_original or [],
    )

    response_text = _call_claude(client, model_id, system_prompt, user_message, max_tokens=1024)
    translations = _parse_window_response(response_text)

    if segment["index"] in translations:
        return normalize_romanian(translations[segment["index"]])

    # Fallback: return the raw response if parsing failed
    return normalize_romanian(response_text.strip())


# ── Main orchestration ───────────────────────────────────────────────

def run_translation(job_id: str, claude_model: str = None,
                    progress_cb=None) -> str:
    """
    Translate all segments for the given job using the 4-stage pipeline.

    Stages:
        1. Document analysis — speaker, topic, glossary terms, real errors (0-5%)
        2. Raw translation — sliding window, accuracy focus only (5-60%)
        3. Quality review — Romanian polish (naturalness, grammar) (60-95%)
        4. Subtitle validation — Python only, no LLM (95-100%)

    Args:
        job_id: The job identifier
        claude_model: "sonnet" or "opus"
        progress_cb: Optional callback(step, progress_pct, message) for SSE progress

    Returns:
        Path to the output Romanian SRT file.
    """
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env -- add your key to the .env file")

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

    logger.info("Job: %s | Segments: %d | Model: %s (%s)",
                job_id, len(segments), claude_model, model_id)

    # Resegment if needed (split long segments)
    original_count = len(segments)
    segments = resegment(segments)
    if len(segments) != original_count:
        logger.info("Resegmented: %d -> %d segments", original_count, len(segments))
        write_srt(segments, srt_original)

    # Load glossary and determine source language
    glossary = load_glossary()
    source_language = job_info.get("source_language", "auto")
    detected_language = job_info.get("detected_language", source_language)
    lang_name = {"ar": "Arabic", "en": "English", "fr": "French",
                 "ur": "Urdu"}.get(detected_language, detected_language)

    video_title = job_info.get("video_title", "Unknown")

    logger.info("Source language: %s | Glossary terms: %d", lang_name, len(glossary))

    # Initialize Claude client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Step 1: Summary pass (0-5%) ──────────────────────────────────
    summary = run_step1_summary(client, model_id, segments, lang_name,
                                glossary, video_title, progress_cb)

    # Save summary and title to job_info
    job_info["document_summary"] = summary
    job_info["title_romanian"] = summary.get("title_romanian", video_title)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    # ── Step 2: Translation pass — Layer 1 (5-55%) ───────────────────
    translated_segments = run_step2_translation(client, model_id, segments,
                                                lang_name, glossary, summary,
                                                progress_cb)

    # Save draft after Layer 1
    write_srt(translated_segments, srt_romanian)

    # ── Step 3: Quality review pass — Layer 2 (55-85%) ───────────────
    translated_segments = run_step3_review(client, model_id,
                                           segments, translated_segments,
                                           lang_name, glossary,
                                           progress_cb)

    # Save after Stage 3
    write_srt(translated_segments, srt_romanian)

    # ── Stage 4: Subtitle validation (Python only, no LLM) ───────────
    if progress_cb:
        progress_cb("translate", 95, "Stage 4/4: Validating subtitles...")

    translated_segments, validation_warnings = validate_subtitles(translated_segments)

    # Merge micro-segments AFTER translation+validation so we work on Romanian text lengths
    pre_merge = len(translated_segments)
    translated_segments = merge_micro_segments(translated_segments)
    if len(translated_segments) != pre_merge:
        logger.info("Micro-segment merge: %d -> %d segments", pre_merge, len(translated_segments))

    write_srt(translated_segments, srt_romanian)

    if validation_warnings:
        logger.info("Stage 4: %d validation warnings", len(validation_warnings))
        for w in validation_warnings[:20]:
            logger.info("  %s", w)

    speed_warnings = sum(1 for seg in translated_segments
                         if check_reading_speed(seg) > MAX_READING_SPEED)

    # Update job info
    job_info["status"] = "translated"
    job_info["srt_romanian_path"] = str(srt_romanian)
    job_info["translated_segment_count"] = len(translated_segments)
    job_info["speed_warnings"] = speed_warnings
    job_info["validation_warnings"] = validation_warnings
    job_info["claude_model"] = claude_model
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    if progress_cb:
        progress_cb("translate", 100, f"Translation complete! {len(translated_segments)} segments.")

    logger.info("Done! %d segments translated -> %s", len(translated_segments), srt_romanian)
    return str(srt_romanian)


# ── Standalone entry point ───────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[Translate] %(message)s")

    parser = argparse.ArgumentParser(description="Translate subtitles with Claude (two-layer)")
    parser.add_argument("job_id", help="Job ID to translate")
    parser.add_argument("--model", default=None, dest="claude_model",
                        choices=["sonnet", "opus"],
                        help="Claude model (default: sonnet)")
    args = parser.parse_args()

    print("=" * 60)
    print("DAWAH-TRANSLATE -- Subtitle Translation (V3 Two-Layer)")
    print("=" * 60)

    def _cli_progress(step, pct, msg):
        print(f"  [{step}] {pct}% | {msg}")

    try:
        srt_path = run_translation(args.job_id, args.claude_model, progress_cb=_cli_progress)
        print()
        print(f"Next step: Review at http://localhost:8000/review/{args.job_id}")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
