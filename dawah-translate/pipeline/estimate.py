"""
Cost estimation for the Claude translation pipeline.

Estimates the input/output token usage for the 3 LLM stages:
    Stage 1 — Document analysis (single call, full SRT)
    Stage 2 — Sliding-window translation (N windows)
    Stage 3 — Quality review (M windows)

Stage 4 (validation) is pure Python and has zero LLM cost.

Returns a price range (low / mid / high) in EUR for each available model.
"""

import math
from pathlib import Path

from config import (
    JOBS_DIR,
    CLAUDE_MODELS,
    WINDOW_SIZE, WINDOW_OVERLAP,
    REVIEW_WINDOW_SIZE, REVIEW_WINDOW_OVERLAP,
)
from pipeline.subtitle import parse_srt


# ── Pricing (USD per million tokens) ─────────────────────────────────
# Verify against docs.anthropic.com/pricing before relying on these.
PRICING_USD_PER_MTOK = {
    "claude-sonnet-4-20250514": {"input": 3.0,  "output": 15.0},
    "claude-opus-4-20250514":   {"input": 15.0, "output": 75.0},
}

# USD → EUR conversion (approximate; refresh occasionally)
USD_TO_EUR = 0.92

# Approximate characters per token by language. Arabic and Urdu pack
# fewer characters per token than English/French.
CHARS_PER_TOKEN = {
    "ar": 2.5, "Arabic": 2.5,
    "ur": 2.5, "Urdu": 2.5,
    "en": 4.0, "English": 4.0,
    "fr": 3.5, "French": 3.5,
}
DEFAULT_CHARS_PER_TOKEN = 3.0

# Romanian output token estimate: ~3.5 chars/token, average translated
# segment ~60 chars → ~17 tokens per segment.
ROMANIAN_CHARS_PER_TOKEN = 3.5
AVG_RO_CHARS_PER_SEGMENT = 60

# Fixed system-prompt overhead per Claude call (input tokens)
STAGE1_PROMPT_OVERHEAD = 700
STAGE2_PROMPT_OVERHEAD = 1200   # short Stage 2 prompt + glossary
STAGE3_PROMPT_OVERHEAD = 600    # short Stage 3 prompt


def _tokens(chars: float, ratio: float) -> float:
    return chars / ratio if ratio else 0.0


def estimate_for_segments(segments: list[dict], source_lang: str) -> dict:
    """
    Compute the projected input/output token totals for a given SRT.

    Returns a dict with token counts for each stage.
    """
    num_segments = len(segments)
    total_source_chars = sum(len(s["text"]) for s in segments)
    avg_source_chars = total_source_chars / num_segments if num_segments else 0

    ratio = CHARS_PER_TOKEN.get(source_lang, DEFAULT_CHARS_PER_TOKEN)
    source_tokens = _tokens(total_source_chars, ratio)

    # Estimated output: avg Romanian chars per segment / Romanian token ratio
    ro_tokens_per_seg = AVG_RO_CHARS_PER_SEGMENT / ROMANIAN_CHARS_PER_TOKEN

    # ── Stage 1 ───────────────────────────────────────────────────────
    # Send full SRT once for analysis. Output is small JSON summary.
    stage1_input = source_tokens + STAGE1_PROMPT_OVERHEAD
    stage1_output = 600  # JSON summary

    # ── Stage 2 ───────────────────────────────────────────────────────
    # Sliding window: WINDOW_SIZE active + 2*WINDOW_OVERLAP context.
    # Each call sends prompt overhead + (active+context) source chars.
    num_windows_2 = max(1, math.ceil(num_segments / WINDOW_SIZE))
    chars_per_window_2 = avg_source_chars * (WINDOW_SIZE + 2 * WINDOW_OVERLAP)
    stage2_input_per_call = STAGE2_PROMPT_OVERHEAD + _tokens(chars_per_window_2, ratio)
    stage2_output_per_call = WINDOW_SIZE * ro_tokens_per_seg
    stage2_input = stage2_input_per_call * num_windows_2
    stage2_output = stage2_output_per_call * num_windows_2

    # ── Stage 3 ───────────────────────────────────────────────────────
    # Review pass. Sends BOTH original + draft per window. For short videos
    # (≤80 segments) it's a single call; otherwise windowed.
    if num_segments <= 80:
        num_windows_3 = 1
        # Single call sends everything
        active_segs_3 = num_segments
        context_segs_3 = 0
    else:
        num_windows_3 = max(1, math.ceil(num_segments / REVIEW_WINDOW_SIZE))
        active_segs_3 = REVIEW_WINDOW_SIZE
        context_segs_3 = REVIEW_WINDOW_OVERLAP

    # Per call: original chars + draft chars (Romanian)
    src_chars_per_call_3 = avg_source_chars * (active_segs_3 + context_segs_3)
    ro_chars_per_call_3 = AVG_RO_CHARS_PER_SEGMENT * (active_segs_3 + context_segs_3)
    stage3_input_per_call = (
        STAGE3_PROMPT_OVERHEAD
        + _tokens(src_chars_per_call_3, ratio)
        + _tokens(ro_chars_per_call_3, ROMANIAN_CHARS_PER_TOKEN)
    )
    stage3_output_per_call = active_segs_3 * ro_tokens_per_seg
    stage3_input = stage3_input_per_call * num_windows_3
    stage3_output = stage3_output_per_call * num_windows_3

    total_input = stage1_input + stage2_input + stage3_input
    total_output = stage1_output + stage2_output + stage3_output

    return {
        "num_segments": num_segments,
        "total_source_chars": total_source_chars,
        "source_language": source_lang,
        "stage1": {"input": int(stage1_input), "output": int(stage1_output)},
        "stage2": {
            "input": int(stage2_input), "output": int(stage2_output),
            "num_windows": num_windows_2,
        },
        "stage3": {
            "input": int(stage3_input), "output": int(stage3_output),
            "num_windows": num_windows_3,
        },
        "total_input_tokens": int(total_input),
        "total_output_tokens": int(total_output),
    }


def cost_for_model(tokens: dict, model_id: str) -> dict:
    """Convert a token estimate to a cost range (EUR) for one model."""
    pricing = PRICING_USD_PER_MTOK.get(model_id)
    if not pricing:
        return {"low_eur": None, "mid_eur": None, "high_eur": None,
                "model_id": model_id, "error": "unknown model"}

    inp = tokens["total_input_tokens"]
    out = tokens["total_output_tokens"]
    mid_usd = (inp / 1e6) * pricing["input"] + (out / 1e6) * pricing["output"]
    mid_eur = mid_usd * USD_TO_EUR

    # ±30% range to account for prompt-overhead variability and Whisper noise
    low = mid_eur * 0.7
    high = mid_eur * 1.3

    return {
        "model_id": model_id,
        "input_usd_per_mtok": pricing["input"],
        "output_usd_per_mtok": pricing["output"],
        "mid_eur": round(mid_eur, 3),
        "low_eur": round(low, 3),
        "high_eur": round(high, 3),
    }


def estimate_job(job_id: str) -> dict:
    """
    Build a full cost estimate for a transcribed job.

    Reads transcript_original.srt and job_info.json, computes token usage,
    then prices it for every available Claude model.

    Returns a dict shaped for the config-screen UI.
    """
    import json
    job_dir = JOBS_DIR / job_id
    info_path = job_dir / "job_info.json"
    srt_path = job_dir / "transcript_original.srt"

    if not srt_path.exists():
        raise FileNotFoundError(f"Original SRT not found for job {job_id}")
    if not info_path.exists():
        raise FileNotFoundError(f"Job info not found for {job_id}")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    segments = parse_srt(srt_path)
    source_lang = info.get("detected_language") or info.get("source_language", "ar")

    tokens = estimate_for_segments(segments, source_lang)

    # Price for each named model alias
    models = {}
    for alias, model_id in CLAUDE_MODELS.items():
        models[alias] = cost_for_model(tokens, model_id)

    return {
        "job_id": job_id,
        "video_title": info.get("video_title", ""),
        "video_duration": info.get("video_duration", 0),
        "detected_language": source_lang,
        "language_confidence": info.get("language_confidence"),
        "segment_count": tokens["num_segments"],
        "tokens": tokens,
        "models": models,
    }
