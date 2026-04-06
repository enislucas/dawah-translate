"""
Translation memory — learns from human corrections to improve future translations.

Stores source-translation pairs in SQLite. When a segment is edited by a human
in the review UI, the correction is saved. Future translations of similar source
text will use the stored correction as a reference.

Schema:
    translations(
        id, source_text, source_lang, translated_text, created_at, updated_at,
        job_id, segment_index, is_human_edited, confidence
    )

Usage:
    from pipeline.memory import TranslationMemory
    tm = TranslationMemory()
    tm.store(source_text, translated_text, source_lang, job_id, seg_idx, human_edited=True)
    matches = tm.lookup(source_text, source_lang)
"""

import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR


DB_PATH = BASE_DIR / "translation_memory.db"


class TranslationMemory:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_text TEXT NOT NULL,
                    source_lang TEXT DEFAULT 'ar',
                    translated_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    job_id TEXT,
                    segment_index INTEGER,
                    is_human_edited INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.5,
                    source_hash TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_hash
                ON translations(source_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_human_edited
                ON translations(is_human_edited)
            """)
            conn.commit()

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison (strip, lowercase, collapse whitespace)."""
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    @staticmethod
    def _hash(text: str) -> str:
        """Create a simple hash for fast lookup."""
        import hashlib
        normalized = TranslationMemory._normalize(text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def store(self, source_text: str, translated_text: str,
              source_lang: str = "ar", job_id: str = None,
              segment_index: int = None, human_edited: bool = False,
              confidence: float = None) -> int:
        """
        Store a translation pair.
        If an exact source match exists, update it (human edits take priority).

        Returns:
            The row ID.
        """
        if confidence is None:
            confidence = 0.9 if human_edited else 0.5

        source_hash = self._hash(source_text)
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Check for existing exact match
            existing = conn.execute(
                "SELECT id, is_human_edited, confidence FROM translations "
                "WHERE source_hash = ? AND source_lang = ?",
                (source_hash, source_lang)
            ).fetchone()

            if existing:
                row_id, was_human, old_confidence = existing
                # Human edits always override; machine translations only update
                # if the new confidence is higher
                if human_edited or (not was_human and confidence >= old_confidence):
                    conn.execute(
                        "UPDATE translations SET translated_text=?, updated_at=?, "
                        "job_id=?, segment_index=?, is_human_edited=?, confidence=? "
                        "WHERE id=?",
                        (translated_text, now, job_id, segment_index,
                         int(human_edited), confidence, row_id)
                    )
                    conn.commit()
                    return row_id
                return row_id
            else:
                cursor = conn.execute(
                    "INSERT INTO translations "
                    "(source_text, source_lang, translated_text, created_at, updated_at, "
                    " job_id, segment_index, is_human_edited, confidence, source_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (source_text, source_lang, translated_text, now, now,
                     job_id, segment_index, int(human_edited), confidence, source_hash)
                )
                conn.commit()
                return cursor.lastrowid

    def lookup(self, source_text: str, source_lang: str = "ar",
               min_confidence: float = 0.0) -> list:
        """
        Look up translations for a source text.
        Returns exact matches sorted by confidence (human edits first).

        Returns:
            List of dicts with keys: translated_text, confidence, is_human_edited, job_id
        """
        source_hash = self._hash(source_text)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT translated_text, confidence, is_human_edited, job_id, updated_at "
                "FROM translations "
                "WHERE source_hash = ? AND source_lang = ? AND confidence >= ? "
                "ORDER BY is_human_edited DESC, confidence DESC, updated_at DESC",
                (source_hash, source_lang, min_confidence)
            ).fetchall()

        return [dict(row) for row in rows]

    def store_job_corrections(self, job_id: str, source_lang: str = "ar"):
        """
        Compare original and edited SRT files for a job.
        Store all segments as translation memory entries.
        Human-edited segments (where text differs from initial translation) get higher confidence.
        """
        from pipeline.subtitle import parse_srt

        job_dir = Path(BASE_DIR) / "jobs" / job_id
        orig_srt = job_dir / "transcript_original.srt"
        ro_srt = job_dir / "transcript_romanian.srt"

        if not orig_srt.exists() or not ro_srt.exists():
            return 0

        original = {s["index"]: s["text"] for s in parse_srt(orig_srt)}
        translated = {s["index"]: s["text"] for s in parse_srt(ro_srt)}

        count = 0
        for idx in original:
            if idx not in translated:
                continue

            source = original[idx]
            trans = translated[idx]

            # We assume all reviewed segments are human-vetted
            self.store(
                source_text=source,
                translated_text=trans,
                source_lang=source_lang,
                job_id=job_id,
                segment_index=idx,
                human_edited=True,
                confidence=0.9,
            )
            count += 1

        return count

    def get_stats(self) -> dict:
        """Get translation memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
            human = conn.execute(
                "SELECT COUNT(*) FROM translations WHERE is_human_edited = 1"
            ).fetchone()[0]
            machine = total - human
            langs = conn.execute(
                "SELECT source_lang, COUNT(*) FROM translations GROUP BY source_lang"
            ).fetchall()

        return {
            "total_entries": total,
            "human_edited": human,
            "machine_generated": machine,
            "by_language": {lang: count for lang, count in langs},
        }
