"""
Translation quality scoring, creedal safety validation, and auto-flagging.

Runs as a post-translation validation step on every segment. Produces a
quality report (quality_report.json) per job with per-segment scores and flags.

Three validation layers:
    1. CREEDAL SAFETY — flags theological red flags (highest priority)
    2. TERMINOLOGY — checks glossary compliance and known NLLB hallucinations
    3. TECHNICAL — reading speed, length, diacritics, formatting

Usage:
    from pipeline.quality import validate_job, validate_segment
    report = validate_job(job_id)
"""

import re
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JOBS_DIR, GLOSSARY_PATH, MAX_READING_SPEED, MAX_CHARS_PER_LINE
from pipeline.subtitle import parse_srt, check_reading_speed, compute_max_chars


# ── Severity levels ─────────────────────────────────────────────────

CRITICAL = "critical"   # Creedal error — must be fixed before publishing
WARNING = "warning"     # Likely wrong — human review strongly recommended
INFO = "info"           # Style/technical issue — optional fix


# ══════════════════════════════════════════════════════════════════════
# LAYER 1: CREEDAL SAFETY
# ══════════════════════════════════════════════════════════════════════

# Patterns that indicate theological errors in Romanian translations.
# Each rule: (compiled_regex, severity, flag_code, human-readable message)

CREEDAL_RULES = [
    # ── Attributes of Allah ──────────────────────────────────────────
    # Attributing human limitations to Allah
    (re.compile(r'\bAllah\b.*\b(nu poate|nu știe|nu vede|nu aude|a greșit|se odihnește)', re.I),
     CRITICAL, "AQEEDAH_ALLAH_LIMITATION",
     "Possible attribution of limitation to Allah. Verify against source."),

    # Attributing divinity to other than Allah
    (re.compile(r'\b(Isus|Iisus|profetul|Muhammad)\b.*\b(este Dumnezeu|divin|divinitate)\b', re.I),
     CRITICAL, "AQEEDAH_SHIRK_ATTRIBUTION",
     "Possible attribution of divinity to other than Allah. Verify."),

    # ── Tawheed violations ───────────────────────────────────────────
    # "islamism" implies political ideology, not the religion
    (re.compile(r'\bislamism(ul|ului)?\b', re.I),
     CRITICAL, "CREED_ISLAMISM",
     "'Islamism/islamismul' found — should be 'Islam/Islamul'. 'Islamism' implies political ideology."),

    # "unificarea" instead of Tawheed
    (re.compile(r'\bunificarea\b.*\b(steag|islam|monoteism|credință)\b', re.I),
     CRITICAL, "CREED_TAWHEED_AS_UNIFICATION",
     "'Unificarea' in religious context — likely mistranslation of Tawheed (Islamic monotheism)."),

    (re.compile(r'\bunificării\b', re.I),
     WARNING, "CREED_TAWHEED_AS_UNIFICATION_2",
     "'Unificării' may be mistranslation of Tawheed."),

    # ── Prophet Muhammad (SAW) respect ───────────────────────────────
    # Feminine reference to masculine prophets/scholars
    (re.compile(r'\b(doamna noastră|doamnei noastre)\b.*\b(Muhammad|Mohammed|Profet|imam)\b', re.I),
     CRITICAL, "CREED_GENDER_PROPHET",
     "Feminine 'doamna noastră' used for Prophet/imam — must be masculine 'stăpânul nostru'."),

    (re.compile(r'\b(Muhammad|Mohammed|Profet)\b.*\b(doamna|doamnei)\b', re.I),
     CRITICAL, "CREED_GENDER_PROPHET_2",
     "Feminine form used in reference to Prophet Muhammad (SAW)."),

    # ── Salawat errors ───────────────────────────────────────────────
    # "pedeapsa" instead of "sfârșitul bun" (العاقبة)
    (re.compile(r'\bpedeapsa\b.*\b(musulman|credincio[șs]|Islam)\b', re.I),
     CRITICAL, "CREED_AQIBAH_AS_PUNISHMENT",
     "'Pedeapsa' in du'a context — likely mistranslation of العاقبة (the good end)."),

    (re.compile(r'\bpedeapsa pentru musulmani\b', re.I),
     CRITICAL, "CREED_PUNISHMENT_MUSLIMS",
     "'Pedeapsa pentru musulmani' is a critical mistranslation. Source likely says 'sfârșitul bun'."),

    # ── Geography / honorifics ───────────────────────────────────────
    (re.compile(r'\bGalile[ea]i?\b', re.I),
     WARNING, "TERM_GALILEEA",
     "'Galileea/Galileii' — likely mistranslation of الجليل (cel ilustru/the illustrious)."),

    (re.compile(r'\b[Ss]emana\b.*\b([Ss]heikh|[Ss]eic|imam|muft)\b', re.I),
     WARNING, "TERM_SAMAHA_AS_WEEK",
     "'Semana' near a scholar title — likely mistranslation of سماحة (Eminența Sa)."),

    (re.compile(r'\binsula\s+arab[ăa]\b', re.I),
     WARNING, "TERM_ISLAND_NOT_PENINSULA",
     "'Insula arabă' — should be 'Peninsula Arabă'."),

    # ── Intercession / worship of dead ───────────────────────────────
    (re.compile(r'\b(se roagă la|se închină la|adoră)\b.*\b(mormânt|mort|profet|sfânt)\b', re.I),
     CRITICAL, "AQEEDAH_WORSHIP_DEAD",
     "Possible shirk: worship/prayer directed at dead/graves. Verify against source."),

    # ── Innovation (bid'ah) presented as sunnah ──────────────────────
    (re.compile(r'\bSunnah\b.*\b(nou[ăa]|modern[ăa]|inventat[ăa])\b', re.I),
     WARNING, "CREED_BIDAH_AS_SUNNAH",
     "Possible confusion between Sunnah and innovation (bid'ah). Verify."),

    # ── Quran as "created" ───────────────────────────────────────────
    (re.compile(r'\bCoranul?\b.*\b(creat|făcut|inventat|scris de)\b', re.I),
     CRITICAL, "AQEEDAH_QURAN_CREATED",
     "Quran described as 'created/invented/written by' — contradicts Ahl us-Sunnah aqeedah."),
]


# ══════════════════════════════════════════════════════════════════════
# LAYER 2: TERMINOLOGY VALIDATION
# ══════════════════════════════════════════════════════════════════════

# Known NLLB hallucination patterns — these appear frequently and are always wrong
NLLB_HALLUCINATION_PATTERNS = [
    (re.compile(r'\bconsiliul\s+(islamic|musulman)\b', re.I),
     WARNING, "NLLB_COUNCIL",
     "'Consiliul islamic' — NLLB hallucinates this for الدعوة (da'wah)."),

    (re.compile(r'\bgramatic[ăa]\b', re.I),
     WARNING, "NLLB_GRAMMAR",
     "'Gramatică' in non-linguistic context — NLLB hallucination. Check source."),

    (re.compile(r'\bunchiul\b.*\b(spus|zis|vorbit|iertat)\b', re.I),
     WARNING, "NLLB_UNCLE",
     "'Unchiul' in speech context — may be phonetic confusion of عما (about) with عمي (uncle)."),

    (re.compile(r'\bfazali\b', re.I),
     WARNING, "NLLB_FAZALI",
     "'Fazali' is untranslated/garbled — likely فضيلة (Excelența Sa, honorific)."),

    (re.compile(r'\brenovator\b', re.I),
     INFO, "NLLB_RENOVATOR",
     "'Renovator' — unusual word choice. Consider 'înnoitor' or 'reformator'."),

    # Duplicate adjectives (NLLB repeats words)
    (re.compile(r'\b(\w{4,})\b.*\b\1\b.*\b\1\b', re.I),
     INFO, "NLLB_TRIPLE_REPEAT",
     "Word repeated 3+ times — possible NLLB hallucination loop."),
]


def check_glossary_compliance(text: str, source_text: str, glossary: dict) -> list:
    """Check if known Arabic terms in source are correctly translated."""
    flags = []

    for arabic_term, info in glossary.items():
        if arabic_term not in source_text:
            continue

        transliteration = info.get("transliteration", "")
        ro_explanation = info.get("ro_explanation", "")
        rule = info.get("rule", "")

        # Check if the correct transliteration or explanation appears
        term_found = False
        if transliteration and transliteration.lower() in text.lower():
            term_found = True
        if ro_explanation and ro_explanation.lower() in text.lower():
            term_found = True

        # Special case: terms with no transliteration (like الجليل) — check explanation
        if not transliteration and ro_explanation:
            if ro_explanation.lower() in text.lower():
                term_found = True

        if not term_found and rule:
            # Check for known wrong translations mentioned in the rule
            never_patterns = re.findall(r"NEVER\s+'([^']+)'", rule)
            for wrong in never_patterns:
                if wrong.lower() in text.lower():
                    flags.append({
                        "severity": CRITICAL if "NEVER" in rule else WARNING,
                        "code": f"GLOSS_{arabic_term[:10]}",
                        "message": f"'{wrong}' found — glossary says NEVER use this for {arabic_term}. {rule}",
                    })

    return flags


# ══════════════════════════════════════════════════════════════════════
# LAYER 3: TECHNICAL VALIDATION
# ══════════════════════════════════════════════════════════════════════

def check_technical(segment: dict) -> list:
    """Check technical subtitle constraints."""
    flags = []
    text = segment["text"]

    # Reading speed
    speed = check_reading_speed(segment)
    if speed > MAX_READING_SPEED * 1.5:
        flags.append({
            "severity": WARNING,
            "code": "TECH_SPEED_CRITICAL",
            "message": f"Reading speed {speed:.0f} chars/sec (limit: {MAX_READING_SPEED}). Way too fast to read.",
        })
    elif speed > MAX_READING_SPEED:
        flags.append({
            "severity": INFO,
            "code": "TECH_SPEED",
            "message": f"Reading speed {speed:.0f} chars/sec (limit: {MAX_READING_SPEED}). Consider shortening.",
        })

    # Line count
    lines = text.split('\n')
    if len(lines) > 2:
        flags.append({
            "severity": WARNING,
            "code": "TECH_LINES",
            "message": f"Segment has {len(lines)} lines (max: 2).",
        })

    # Line length
    for i, line in enumerate(lines):
        if len(line) > MAX_CHARS_PER_LINE:
            flags.append({
                "severity": INFO,
                "code": "TECH_LINE_LENGTH",
                "message": f"Line {i+1} is {len(line)} chars (max: {MAX_CHARS_PER_LINE}).",
            })

    # Romanian diacritics — check for cedilla variants
    if '\u015F' in text or '\u0163' in text:
        flags.append({
            "severity": INFO,
            "code": "TECH_DIACRITICS",
            "message": "Contains cedilla diacritics (ş/ţ) instead of comma-below (ș/ț).",
        })

    # Empty or near-empty
    if len(text.strip()) < 2:
        flags.append({
            "severity": WARNING,
            "code": "TECH_EMPTY",
            "message": "Segment is empty or near-empty.",
        })

    # Very short duration with long text
    duration = segment["end"] - segment["start"]
    if duration < 0.5 and len(text) > 10:
        flags.append({
            "severity": WARNING,
            "code": "TECH_FLASH",
            "message": f"Duration {duration:.2f}s is too short for {len(text)} chars — will flash on screen.",
        })

    return flags


# ══════════════════════════════════════════════════════════════════════
# MAIN VALIDATION
# ══════════════════════════════════════════════════════════════════════

def validate_segment(segment: dict, source_text: str = "", glossary: dict = None) -> dict:
    """
    Validate a single translated segment.

    Returns:
        {
            "index": int,
            "score": float (0-100),
            "flags": [{"severity": str, "code": str, "message": str}, ...],
            "has_critical": bool,
            "has_warning": bool,
        }
    """
    if glossary is None:
        glossary = {}

    flags = []
    text = segment["text"]

    # Layer 1: Creedal safety
    for pattern, severity, code, message in CREEDAL_RULES:
        if pattern.search(text):
            flags.append({"severity": severity, "code": code, "message": message})

    # Layer 2: Terminology
    for pattern, severity, code, message in NLLB_HALLUCINATION_PATTERNS:
        if pattern.search(text):
            flags.append({"severity": severity, "code": code, "message": message})

    # Layer 2b: Glossary compliance
    if source_text:
        flags.extend(check_glossary_compliance(text, source_text, glossary))

    # Layer 3: Technical
    flags.extend(check_technical(segment))

    # Compute score (start at 100, deduct per flag)
    score = 100.0
    for flag in flags:
        if flag["severity"] == CRITICAL:
            score -= 25
        elif flag["severity"] == WARNING:
            score -= 10
        elif flag["severity"] == INFO:
            score -= 3
    score = max(0.0, score)

    has_critical = any(f["severity"] == CRITICAL for f in flags)
    has_warning = any(f["severity"] == WARNING for f in flags)

    return {
        "index": segment["index"],
        "score": round(score, 1),
        "flags": flags,
        "has_critical": has_critical,
        "has_warning": has_warning,
    }


def validate_job(job_id: str) -> dict:
    """
    Validate all translated segments for a job.
    Writes quality_report.json to the job directory.

    Returns:
        {
            "job_id": str,
            "total_segments": int,
            "average_score": float,
            "critical_count": int,
            "warning_count": int,
            "segments": [segment_report, ...],
            "summary": str,
        }
    """
    job_dir = JOBS_DIR / job_id

    # Load translated SRT
    ro_srt = job_dir / "transcript_romanian.srt"
    if not ro_srt.exists():
        raise FileNotFoundError(f"Romanian SRT not found: {ro_srt}")
    translated = parse_srt(ro_srt)

    # Load original SRT for glossary compliance checking
    orig_srt = job_dir / "transcript_original.srt"
    source_lookup = {}
    if orig_srt.exists():
        for seg in parse_srt(orig_srt):
            source_lookup[seg["index"]] = seg["text"]

    # Load glossary
    glossary = {}
    if GLOSSARY_PATH.exists():
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            glossary = json.load(f)

    # Validate each segment
    segment_reports = []
    critical_count = 0
    warning_count = 0
    total_score = 0.0

    for seg in translated:
        source_text = source_lookup.get(seg["index"], "")
        report = validate_segment(seg, source_text, glossary)
        segment_reports.append(report)

        if report["has_critical"]:
            critical_count += 1
        if report["has_warning"]:
            warning_count += 1
        total_score += report["score"]

    avg_score = total_score / len(translated) if translated else 0

    # Generate summary
    if critical_count > 0:
        summary = f"NEEDS REVIEW: {critical_count} critical issues (creedal/theological errors detected)"
    elif warning_count > 0:
        summary = f"Review recommended: {warning_count} warnings found"
    elif avg_score >= 90:
        summary = "Good quality — minor issues only"
    else:
        summary = f"Average quality (score: {avg_score:.0f}/100)"

    result = {
        "job_id": job_id,
        "total_segments": len(translated),
        "average_score": round(avg_score, 1),
        "critical_count": critical_count,
        "warning_count": warning_count,
        "info_count": sum(1 for r in segment_reports
                         if any(f["severity"] == INFO for f in r["flags"])),
        "segments": segment_reports,
        "summary": summary,
    }

    # Save report
    report_path = job_dir / "quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def print_report(report: dict) -> None:
    """Print a human-readable quality report to stdout."""
    print(f"\n{'='*60}")
    print(f"QUALITY REPORT: {report['job_id']}")
    print(f"{'='*60}")
    print(f"Segments: {report['total_segments']}")
    print(f"Average score: {report['average_score']}/100")
    print(f"Critical: {report['critical_count']} | "
          f"Warning: {report['warning_count']} | "
          f"Info: {report['info_count']}")
    print(f"Summary: {report['summary']}")
    print()

    # Print flagged segments
    for seg in report["segments"]:
        if not seg["flags"]:
            continue

        severity_icon = "!!" if seg["has_critical"] else "!" if seg["has_warning"] else "~"
        print(f"  [{seg['index']}] Score: {seg['score']}/100 {severity_icon}")
        for flag in seg["flags"]:
            prefix = {"critical": "  CRITICAL", "warning": "  WARNING", "info": "  info"}
            print(f"    {prefix.get(flag['severity'], '  ?')}  [{flag['code']}] {flag['message']}")
        print()


# ── Standalone entry point ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate translation quality")
    parser.add_argument("job_id", help="Job ID to validate")
    args = parser.parse_args()

    report = validate_job(args.job_id)
    print_report(report)
