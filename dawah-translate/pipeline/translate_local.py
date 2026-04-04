"""
Local translation using NLLB-200 (Meta's No Language Left Behind) + glossary post-processing.

No API calls needed. Runs entirely on CPU using CTranslate2 for efficiency.

Usage:
    python pipeline/translate_local.py <job_id> [--language ar]

Model: facebook/nllb-200-distilled-600M (CTranslate2 int8 quantized, ~400 MB)
First run downloads the model to ~/.cache/huggingface/

Supported source languages:
    ar  -> Arabic (arb_Arab)
    en  -> English (eng_Latn)
    tr  -> Turkish (tur_Latn)
    fr  -> French (fra_Latn)

Target: Romanian (ron_Latn)
"""

import sys
import json
import time
import re
import argparse
from pathlib import Path

# Allow running as standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    JOBS_DIR, GLOSSARY_PATH, MAX_READING_SPEED,
)
from pipeline.subtitle import (
    parse_srt, write_srt, normalize_romanian,
    check_reading_speed, resegment,
)


# ── NLLB language codes ──────────────────────────────────────────────

NLLB_LANG_CODES = {
    "ar": "arb_Arab",
    "en": "eng_Latn",
    "tr": "tur_Latn",
    "fr": "fra_Latn",
    "ro": "ron_Latn",
    "auto": "arb_Arab",  # Default fallback for auto-detect
}

TARGET_LANG = "ron_Latn"

# CTranslate2-converted model — no PyTorch needed, ~400 MB int8
NLLB_CT2_MODEL = "JustFrederik/nllb-200-distilled-600M-ct2-int8"


# ── Glossary loading and processing ──────────────────────────────────

def load_glossary() -> dict:
    """Load glossary.json."""
    if not GLOSSARY_PATH.exists():
        return {}
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_glossary(source_text: str, translated_text: str, glossary: dict,
                   introduced_terms: set) -> tuple[str, set]:
    """
    Post-process a translation to apply glossary rules.

    Strategy:
    1. Scan the Arabic source for glossary terms
    2. For each found term, check if NLLB's translation contains a Romanian
       equivalent or a garbled version
    3. Replace/inject the correct transliteration according to the rules

    Returns (processed_text, updated_introduced_terms).
    """
    if not glossary:
        return translated_text, introduced_terms

    result = translated_text
    new_introduced = set(introduced_terms)

    for arabic_term, info in glossary.items():
        # Check if this Arabic term appears in the source
        if arabic_term not in source_text:
            continue

        translit = info["transliteration"]
        ro_expl = info.get("ro_explanation", "")
        rule = info.get("rule", "")

        # Determine what to insert
        if "Never translate" in rule or "Always" in rule:
            # Always keep transliterated form
            replacement = translit
        elif translit in new_introduced:
            # Already introduced — just use transliteration
            replacement = translit
        else:
            # First occurrence — include explanation if available
            if ro_expl:
                replacement = f"{translit} ({ro_expl})"
            else:
                replacement = translit
            new_introduced.add(translit)

        # Try to find and replace NLLB's translation of this term
        # NLLB may have translated it to the Romanian equivalent, or
        # it may have transliterated it differently, or left it in Arabic
        replaced = False

        # Strategy 1: If the Romanian explanation exists in the output, replace it
        if ro_expl and ro_expl.lower() in result.lower():
            # Case-insensitive replacement of the Romanian equivalent
            pattern = re.compile(re.escape(ro_expl), re.IGNORECASE)
            result = pattern.sub(replacement, result, count=1)
            replaced = True

        # Strategy 2: If Arabic term leaked through into the translation
        if not replaced and arabic_term in result:
            result = result.replace(arabic_term, replacement, 1)
            replaced = True

        # Strategy 3: If a common mistranslation pattern exists
        # (NLLB sometimes produces partial transliterations)
        if not replaced:
            # Check for common NLLB Romanian translations of Islamic terms
            common_translations = _get_common_translations(arabic_term)
            for common in common_translations:
                if common.lower() in result.lower():
                    pattern = re.compile(re.escape(common), re.IGNORECASE)
                    result = pattern.sub(replacement, result, count=1)
                    replaced = True
                    break

        # Strategy 4: If nothing was found to replace, append at end
        # (only for key terms that really should be there)
        if not replaced and ro_expl:
            # Don't force-append — NLLB may have used a synonym
            pass

    return result, new_introduced


def _get_common_translations(arabic_term: str) -> list[str]:
    """
    Return common Romanian translations that NLLB might produce
    for known Islamic terms, so we can find and replace them.
    """
    # Map of Arabic terms to likely NLLB Romanian outputs
    known_outputs = {
        "صلاة": ["rugăciune", "rugaciune", "rugăciunea", "salat"],
        "دعاء": ["rugăciune", "invocație", "invocare", "dua"],
        "زكاة": ["zakat", "almă", "caritate", "danie"],
        "توحيد": ["monoteism", "unicitate", "tawhid", "unitatea"],
        "شرك": ["politeism", "asociere", "idolatrie", "shirk"],
        "سنة": ["tradiție", "traditie", "sunna", "suna"],
        "حديث": ["tradiție", "traditie", "hadis", "povestire"],
        "تقوى": ["pietate", "evlavie", "frică", "teamă", "conștiință"],
        "فقه": ["jurisprudență", "drept", "fiqh"],
        "عقيدة": ["credință", "credinta", "doctrină", "aqida"],
    }
    return known_outputs.get(arabic_term, [])


# ── NLLB Translation Engine ─────────────────────────────────────────

class NLLBTranslator:
    """Local translator using NLLB-200 via CTranslate2."""

    def __init__(self, source_lang: str = "ar"):
        self.source_lang_code = NLLB_LANG_CODES.get(source_lang, "arb_Arab")
        self.target_lang_code = TARGET_LANG
        self.translator = None
        self.tokenizer = None

    def load_model(self):
        """Load the NLLB CTranslate2 model and tokenizer."""
        import ctranslate2
        from transformers import AutoTokenizer

        print(f"  Loading NLLB-200 model ({NLLB_CT2_MODEL})...")
        print(f"  (First run downloads ~400 MB — cached for future runs)")

        # Download and load CTranslate2 model
        model_path = self._get_model_path()
        self.translator = ctranslate2.Translator(
            model_path,
            device="cpu",
            inter_threads=4,
            intra_threads=2,
            compute_type="int8",
        )

        # Load the tokenizer from the original model
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            src_lang=self.source_lang_code,
        )

        print(f"  Model loaded. Source: {self.source_lang_code} -> Target: {self.target_lang_code}")

    def _get_model_path(self) -> str:
        """Download and return path to CTranslate2 model."""
        from huggingface_hub import snapshot_download
        return snapshot_download(
            NLLB_CT2_MODEL,
            local_files_only=False,
        )

    def translate_text(self, text: str) -> str:
        """Translate a single text string."""
        if not text.strip():
            return text

        # Tokenize
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text)
        )

        # Translate with CTranslate2
        target_prefix = [self.target_lang_code]
        results = self.translator.translate_batch(
            [tokens],
            target_prefix=[target_prefix],
            beam_size=5,
            max_decoding_length=256,
            repetition_penalty=1.2,
        )

        # Decode — skip the language code token
        output_tokens = results[0].hypotheses[0]
        if output_tokens and output_tokens[0] == self.target_lang_code:
            output_tokens = output_tokens[1:]

        translated = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        )

        return translated.strip()

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate multiple texts efficiently in one batch."""
        if not texts:
            return []

        # Tokenize all texts
        all_tokens = []
        for text in texts:
            if not text.strip():
                all_tokens.append(["</s>"])
            else:
                tokens = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer.encode(text)
                )
                all_tokens.append(tokens)

        # Translate batch
        target_prefix = [[self.target_lang_code]] * len(all_tokens)
        results = self.translator.translate_batch(
            all_tokens,
            target_prefix=target_prefix,
            beam_size=5,
            max_decoding_length=256,
            repetition_penalty=1.2,
        )

        # Decode all results
        translations = []
        for result in results:
            output_tokens = result.hypotheses[0]
            if output_tokens and output_tokens[0] == self.target_lang_code:
                output_tokens = output_tokens[1:]

            translated = self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(output_tokens),
                skip_special_tokens=True,
            )
            translations.append(translated.strip())

        return translations


# ── Main translation pipeline ────────────────────────────────────────

def run_translation_local(job_id: str, language: str = None) -> str:
    """
    Translate all segments for the given job using local NLLB-200.

    Args:
        job_id: The job identifier
        language: Source language code ("ar", "en", or None for auto)

    Returns:
        Path to the output Romanian SRT file.
    """
    # Load job info
    job_dir = JOBS_DIR / job_id
    info_path = job_dir / "job_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        job_info = json.load(f)

    srt_original = job_dir / "transcript_original.srt"
    srt_romanian = job_dir / "transcript_romanian.srt"

    if not srt_original.exists():
        raise FileNotFoundError(f"Original SRT not found: {srt_original}")

    # Determine source language
    if language and language != "auto":
        src_lang = language
    else:
        src_lang = job_info.get("detected_language", job_info.get("source_language", "ar"))
        if src_lang == "auto":
            src_lang = "ar"

    # Parse original subtitles
    segments = parse_srt(srt_original)
    if not segments:
        raise ValueError("No segments found in the original SRT file")

    print(f"Job: {job_id}")
    print(f"Segments: {len(segments)}")
    print(f"Source language: {src_lang}")
    print(f"Translation engine: NLLB-200 (local, no API)")
    print()

    # Resegment if needed
    original_count = len(segments)
    segments = resegment(segments)
    if len(segments) != original_count:
        print(f"  Resegmented: {original_count} -> {len(segments)} segments")
        write_srt(segments, srt_original)

    # Update job status
    job_info["status"] = "translating"
    job_info["translation_engine"] = "nllb-200-local"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    # Load model
    print("[1/3] Loading NLLB-200 model...")
    translator = NLLBTranslator(source_lang=src_lang)
    load_start = time.time()
    translator.load_model()
    print(f"  Loaded in {time.time() - load_start:.1f}s")
    print()

    # Load glossary
    glossary = load_glossary()
    print(f"[2/3] Translating {len(segments)} segments...")
    if glossary:
        print(f"  Glossary: {len(glossary)} terms loaded for post-processing")

    # Translate in batches of 20 for efficiency
    batch_size = 20
    translated_segments = []
    introduced_terms = set()
    trans_start = time.time()

    for batch_start in range(0, len(segments), batch_size):
        batch_end = min(batch_start + batch_size, len(segments))
        batch = segments[batch_start:batch_end]

        # Extract source texts
        source_texts = [seg["text"] for seg in batch]

        # Translate batch
        translated_texts = translator.translate_batch(source_texts)

        # Apply glossary post-processing and Romanian normalization
        for i, (seg, src_text, trans_text) in enumerate(zip(batch, source_texts, translated_texts)):
            # Apply glossary
            processed_text, introduced_terms = apply_glossary(
                src_text, trans_text, glossary, introduced_terms
            )

            # Normalize Romanian diacritics
            processed_text = normalize_romanian(processed_text)

            translated_segments.append({
                **seg,
                "text": processed_text,
            })

        # Progress
        done = len(translated_segments)
        elapsed = time.time() - trans_start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (len(segments) - done) / rate if rate > 0 else 0
        print(
            f"\r  Progress: {done}/{len(segments)} segments | "
            f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining",
            end="", flush=True,
        )

        # Save incrementally
        write_srt(translated_segments, srt_romanian)

    trans_time = time.time() - trans_start
    print(f"\n  Translation complete in {trans_time:.1f}s")
    print()

    # Validation
    print("[3/3] Validating translations...")
    speed_warnings = 0
    for seg in translated_segments:
        speed = check_reading_speed(seg)
        if speed > MAX_READING_SPEED:
            speed_warnings += 1

    if speed_warnings:
        print(f"  {speed_warnings} segments exceed {MAX_READING_SPEED} chars/sec")
    else:
        print(f"  All segments within reading speed limits")

    # Update job info
    job_info["status"] = "translated"
    job_info["srt_romanian_path"] = str(srt_romanian)
    job_info["translated_segment_count"] = len(translated_segments)
    job_info["speed_warnings"] = speed_warnings
    job_info["translation_time"] = round(trans_time, 1)
    job_info["translation_engine"] = "nllb-200-local"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(job_info, f, indent=2, ensure_ascii=False)

    print()
    print(f"Done! {len(translated_segments)} segments translated -> {srt_romanian}")
    print(f"Cost: $0.00 (local translation)")
    return str(srt_romanian)


# ── Single segment re-translation (for review UI) ────────────────────

# Keep a cached translator instance for re-translation requests
_cached_translator = None


def translate_single_segment_local(segment: dict, source_language: str = "ar") -> str:
    """Translate a single segment locally (for re-translate button in review UI)."""
    global _cached_translator

    if _cached_translator is None:
        _cached_translator = NLLBTranslator(source_lang=source_language)
        _cached_translator.load_model()

    translated = _cached_translator.translate_text(segment["text"])

    # Apply glossary
    glossary = load_glossary()
    processed, _ = apply_glossary(segment["text"], translated, glossary, set())
    return normalize_romanian(processed)


# ── Standalone entry point ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate subtitles locally with NLLB-200")
    parser.add_argument("job_id", help="Job ID to translate")
    parser.add_argument("--language", default=None, choices=["ar", "en", "tr", "fr", "auto"],
                        help="Source language (default: auto from job info)")
    args = parser.parse_args()

    print("=" * 60)
    print("DAWAH-TRANSLATE — Local Translation (NLLB-200)")
    print("=" * 60)
    print("No API key needed. Runs entirely on your machine.")
    print()

    try:
        srt_path = run_translation_local(args.job_id, args.language)
        print()
        print(f"Next step: Review at http://localhost:8000/review/{args.job_id}")
        print(f"  Or burn directly: python pipeline/burn.py {args.job_id}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
