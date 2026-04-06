"""
Post-edit NLLB translations using a local LLM via Ollama.

Stage 2 of the hybrid translation pipeline:
    NLLB-200 (rough draft) -> Local LLM (refined Romanian)

Usage:
    python pipeline/translate_refine.py <job_id> [--model gemma3:4b]

Requires Ollama running locally: http://localhost:11434
Install: https://ollama.com   then:  ollama pull gemma3:4b
"""

import sys
import json
import time
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    JOBS_DIR, GLOSSARY_PATH, MAX_READING_SPEED,
    MAX_CHARS_PER_LINE, MAX_LINES_PER_BLOCK,
)
from pipeline.subtitle import (
    parse_srt, write_srt, normalize_romanian,
    check_reading_speed, compute_max_chars,
)

# ── Constants ────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
REFINE_BATCH_SIZE = 5  # Small batches for CPU inference speed


# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Romanian subtitle editor specializing in Islamic content. You receive rough machine translations (from NLLB-200) and refine them. The machine translation is often WRONG on Islamic terminology — treat it as a rough draft only. Always verify meaning against the SOURCE text.

TASK: Edit each subtitle segment to be natural Romanian with correct Islamic terminology.

RULES:
1. Use the GLOSSARY for Islamic terms. First occurrence: include Romanian explanation in parentheses. After: just the transliterated term.
2. Each segment has a [max_chars] limit. Your output MUST fit. Be concise — this is a subtitle, not a transcript.
3. Use proper Romanian diacritics: ș (not ş), ț (not ţ), ă, î, â.
4. Fix grammatical errors from the machine translation.
5. Keep the meaning faithful to the SOURCE text. The DRAFT is just a starting point — if it's wrong, fix it using the source.
6. Max 2 lines per segment. Use \\n for line breaks if needed.
7. Do NOT over-arabicize. Only use Arabic terms from the glossary.

CRITICAL ERROR PATTERNS TO WATCH FOR (the machine translation makes these mistakes frequently):

A. CONTEXTUAL RELIGIOUS TERMS:
   - "الصلاة والسلام على" = salawat formula = "binecuvantarea si pacea fie asupra..." (divine blessings). صلاة here is NOT ritual prayer.
   - "التوحيد" = Tawheed (Islamic monotheism). NEVER "unificarea" (political unification).
   - "العاقبة" in du'a = "sfarsitul bun" (the good end). NEVER "pedeapsa" (punishment).
   - "الإسلام" = "Islam" (the religion). NEVER "islamism/islamismul" (political ideology).

B. HONORIFICS MISTAKEN FOR NOUNS/PLACES:
   - "الجليل" after a title/person = "cel ilustru" (the illustrious). NEVER "Galileea" (Galilee).
   - "سماحة" = "Eminenta Sa" (His Eminence). NEVER "semana" (week).
   - "سيدنا" = "stapanul nostru" (our master, MASCULINE). NEVER "doamna noastra" (our lady).

C. PHONETIC CONFUSIONS:
   - "عما" (about what) is NOT "عمي" (my uncle). If draft says "unchiul" in a context about speech, it's wrong.
   - "الدعوة" = "da'wah" (Islamic call). NEVER "consiliul" (the council).

D. GENDER ERRORS:
   - In references to prophets, Allah, or male scholars: always masculine pronouns (Sau/Sa, not ei).
   - "سيدنا" = "stapanul nostru" (MASCULINE). NEVER "doamna noastra".

E. ARABIC HUMILITY FORMULAS:
   - "أنا دون ما قال" = "sunt mai prejos de ceea ce a spus" (I am less than what he said). This is modesty.
   - "الجزيرة العربية" = "Peninsula Araba". NEVER "insula araba" (Arabian island)."""


def build_glossary_text(glossary: dict) -> str:
    """Format glossary for inclusion in the prompt."""
    lines = []
    for arabic, info in glossary.items():
        trans = info.get("transliteration", "")
        expl = info.get("ro_explanation", "")
        rule = info.get("rule", "")
        entry = f"  {arabic}"
        if trans:
            entry += f" -> {trans}"
        if expl:
            entry += f" ({expl})"
        if rule:
            entry += f" | {rule}"
        lines.append(entry)
    return "\n".join(lines)


def build_user_prompt(batch: list, nllb_segments: dict, glossary: dict,
                      introduced_terms: set, context_segments: list = None) -> str:
    """Build the user message for a batch of segments."""
    glossary_text = build_glossary_text(glossary)
    introduced_text = ", ".join(sorted(introduced_terms)) if introduced_terms else "(none yet)"

    parts = [
        f"GLOSSARY:\n{glossary_text}",
        f"\nPREVIOUSLY INTRODUCED TERMS: {introduced_text}",
    ]

    if context_segments:
        parts.append("\nCONTEXT (previous segments, do not edit):")
        for seg in context_segments:
            parts.append(f"  [{seg['index']}] {seg['text']}")

    parts.append("\nEDIT THE FOLLOWING SEGMENTS:")
    for seg in batch:
        max_chars = compute_max_chars(seg)
        max_chars = max(max_chars, 20)
        nllb_draft = nllb_segments.get(seg["index"], "")
        source = seg["text"]
        parts.append(f"[{seg['index']}] [max_chars:{max_chars}] | SOURCE: {source} | DRAFT: {nllb_draft}")

    parts.append("\nOutput format (one per line, nothing else):")
    parts.append("[index] | refined_text")

    return "\n".join(parts)


def clean_refined_text(text: str) -> str:
    """Remove raw Arabic text that the LLM sometimes leaves in parentheses."""
    # Remove parenthesized Arabic text like (صلى الله عليه وسلم)
    text = re.sub(r'\s*\([^)]*[\u0600-\u06FF]+[^)]*\)', '', text)
    # Remove any remaining standalone Arabic characters
    text = re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+', '', text)
    # Clean up double spaces and trailing/leading whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove trailing punctuation artifacts like ", ." or " ."
    text = re.sub(r'\s+([.,])', r'\1', text)
    return text


def parse_refine_response(response_text: str) -> dict:
    """Parse LLM response into {index: refined_text}."""
    results = {}
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Accept both [1] | text and 1 | text formats
        match = re.match(r'\[?(\d+)\]?\s*\|\s*(.+)', line)
        if match:
            idx = int(match.group(1))
            text = match.group(2).strip()
            results[idx] = text
    return results


# ── Ollama API ───────────────────────────────────────────────────────

def check_ollama_running() -> bool:
    """Check if Ollama is running and accessible."""
    import httpx
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def check_model_available(model: str) -> bool:
    """Check if the specified model is pulled in Ollama."""
    import httpx
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        data = resp.json()
        available = [m.get("name", "") for m in data.get("models", [])]
        # Match both "qwen3:4b" and "qwen3:4b-latest" etc.
        return any(model in name or name.startswith(model) for name in available)
    except Exception:
        return False


def call_ollama(model: str, system: str, user: str, temperature: float = 0.3) -> str:
    """Call Ollama chat API and return the response text."""
    import httpx

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 4096,
        },
    }

    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=600.0,  # 10 min timeout for CPU inference
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


# ── Main refinement pipeline ────────────────────────────────────────

def _check_memory(source_text: str) -> str | None:
    """Check translation memory for a human-vetted translation."""
    try:
        from pipeline.memory import TranslationMemory
        tm = TranslationMemory()
        matches = tm.lookup(source_text, min_confidence=0.8)
        if matches and matches[0].get("is_human_edited"):
            return matches[0]["translated_text"]
    except Exception:
        pass
    return None


def load_glossary() -> dict:
    if not GLOSSARY_PATH.exists():
        return {}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_refinement(job_id: str, ollama_model: str = None) -> str:
    """
    Post-edit NLLB translations using Ollama.

    Reads transcript_original.srt (source) and transcript_romanian.srt (NLLB draft),
    sends both to Ollama for refinement, writes back to transcript_romanian.srt.

    Args:
        job_id: The job identifier
        ollama_model: Ollama model name (default: qwen3:4b)

    Returns:
        Path to the refined SRT file.
    """
    if ollama_model is None:
        ollama_model = DEFAULT_OLLAMA_MODEL

    # Check Ollama
    if not check_ollama_running():
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve\n"
            "Install from: https://ollama.com"
        )
    if not check_model_available(ollama_model):
        raise RuntimeError(
            f"Model '{ollama_model}' not found in Ollama.\n"
            f"Pull it with: ollama pull {ollama_model}"
        )

    job_dir = JOBS_DIR / job_id
    info_path = job_dir / "job_info.json"

    with open(info_path, "r", encoding="utf-8") as f:
        job_info = json.load(f)

    source_srt_path = job_dir / "transcript_original.srt"
    nllb_srt_path = job_dir / "transcript_romanian.srt"

    if not source_srt_path.exists():
        raise FileNotFoundError(f"Original SRT not found: {source_srt_path}")
    if not nllb_srt_path.exists():
        raise FileNotFoundError(
            f"NLLB draft SRT not found: {nllb_srt_path}\n"
            "Run local (fast) translation first: python pipeline/translate_local.py <job_id>"
        )

    # Parse both SRT files
    source_segments = parse_srt(source_srt_path)
    nllb_segments_list = parse_srt(nllb_srt_path)

    # Build lookup: index -> NLLB draft text
    nllb_lookup = {seg["index"]: seg["text"] for seg in nllb_segments_list}

    if not source_segments:
        raise ValueError("No segments in source SRT")

    print(f"Job: {job_id}")
    print(f"Segments: {len(source_segments)}")
    print(f"Ollama model: {ollama_model}")
    print(f"Batch size: {REFINE_BATCH_SIZE}")
    print()

    # Update job status
    job_info["status"] = "refining"
    job_info["refinement_model"] = ollama_model
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    # Load glossary
    glossary = load_glossary()
    print(f"Glossary: {len(glossary)} terms")
    print()

    # Refine in batches
    total_batches = (len(source_segments) + REFINE_BATCH_SIZE - 1) // REFINE_BATCH_SIZE
    refined_segments = []
    introduced_terms = set()
    refine_start = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * REFINE_BATCH_SIZE
        end = min(start + REFINE_BATCH_SIZE, len(source_segments))
        batch = source_segments[start:end]

        # Context: last 3 refined segments
        context = refined_segments[-3:] if refined_segments else None

        print(f"  Batch {batch_idx + 1}/{total_batches} "
              f"(segments {start + 1}-{end})...", end="", flush=True)

        batch_start = time.time()

        user_prompt = build_user_prompt(
            batch, nllb_lookup, glossary, introduced_terms, context
        )

        try:
            response = call_ollama(ollama_model, SYSTEM_PROMPT, user_prompt)
            parsed = parse_refine_response(response)

            # Apply refinements
            for seg in batch:
                idx = seg["index"]

                # Check translation memory for human-vetted translation
                tm_text = _check_memory(seg["text"])
                if tm_text:
                    refined_text = tm_text
                elif idx in parsed:
                    refined_text = clean_refined_text(normalize_romanian(parsed[idx]))
                else:
                    # Fall back to NLLB draft if LLM didn't return this segment
                    refined_text = nllb_lookup.get(idx, seg["text"])
                    print(f"\n    Warning: segment {idx} not in LLM output, using NLLB draft")

                # Track introduced glossary terms
                for arabic, info in glossary.items():
                    trans = info.get("transliteration", "")
                    if trans and trans.lower() in refined_text.lower():
                        introduced_terms.add(trans)

                # Use source timing, refined text
                refined_segments.append({
                    "index": idx,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": refined_text,
                })

            batch_time = time.time() - batch_start
            elapsed = time.time() - refine_start
            remaining = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
            print(f" done ({batch_time:.1f}s) | ~{remaining:.0f}s remaining")

        except Exception as e:
            print(f"\n    Error in batch {batch_idx + 1}: {e}")
            # Fall back to NLLB drafts for this batch
            for seg in batch:
                idx = seg["index"]
                refined_segments.append({
                    "index": idx,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": nllb_lookup.get(idx, seg["text"]),
                })

        # Save incrementally
        write_srt(refined_segments, nllb_srt_path)

    refine_time = time.time() - refine_start
    print(f"\n  Refinement complete in {refine_time:.1f}s")
    print()

    # Validate
    print("Validating...")
    speed_warnings = 0
    for seg in refined_segments:
        speed = check_reading_speed(seg)
        if speed > MAX_READING_SPEED:
            speed_warnings += 1

    if speed_warnings:
        print(f"  {speed_warnings} segments exceed {MAX_READING_SPEED} chars/sec")
    else:
        print(f"  All segments within reading speed limits")

    # Update job info
    job_info["status"] = "translated"
    job_info["translation_engine"] = "local_quality"
    job_info["refinement_model"] = ollama_model
    job_info["refinement_time"] = round(refine_time, 1)
    job_info["speed_warnings"] = speed_warnings
    job_info["srt_romanian_path"] = str(nllb_srt_path)
    job_info["translated_segment_count"] = len(refined_segments)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(refined_segments)} segments refined -> {nllb_srt_path}")
    print(f"Cost: $0.00 (local refinement)")
    return str(nllb_srt_path)


def refine_single_segment(segment: dict, nllb_text: str,
                          source_language: str = "ar",
                          ollama_model: str = None) -> str:
    """Refine a single segment (for review UI re-translate button)."""
    if ollama_model is None:
        ollama_model = DEFAULT_OLLAMA_MODEL

    if not check_ollama_running():
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")

    glossary = load_glossary()
    user_prompt = build_user_prompt(
        [segment], {segment["index"]: nllb_text}, glossary, set()
    )

    response = call_ollama(ollama_model, SYSTEM_PROMPT, user_prompt)
    parsed = parse_refine_response(response)

    if segment["index"] in parsed:
        return clean_refined_text(normalize_romanian(parsed[segment["index"]]))
    return clean_refined_text(normalize_romanian(nllb_text))


# ── Standalone entry point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-edit translations with Ollama")
    parser.add_argument("job_id", help="Job ID to refine")
    parser.add_argument("--model", default=None,
                        help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})")
    args = parser.parse_args()

    print("=" * 60)
    print("DAWAH-TRANSLATE -- LLM Post-Editing (Ollama)")
    print("=" * 60)
    print("Refines NLLB-200 drafts using a local LLM.")
    print("Requires: ollama serve + ollama pull gemma3:4b")
    print()

    try:
        srt_path = run_refinement(args.job_id, args.model)
        print()
        print(f"Next step: Review at http://localhost:8000/review/{args.job_id}")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
