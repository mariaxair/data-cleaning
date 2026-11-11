"""SQLite OCR cache and deterministic completion manifest."""

import sqlite3
from pathlib import Path

from .models import OCRResult


class PipelineCache:
    """Persist page transcriptions and completed-file fingerprints."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS ocr_cache "
                "(cache_key TEXT PRIMARY KEY, text TEXT NOT NULL, confidence REAL, provider TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS processed_files "
                "(source_key TEXT PRIMARY KEY, source_hash TEXT NOT NULL, config_hash TEXT NOT NULL, "
                "output_path TEXT NOT NULL, report_path TEXT NOT NULL, status TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get_ocr(self, cache_key: str) -> OCRResult | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT text, confidence, provider FROM ocr_cache WHERE cache_key=?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        return OCRResult(text=row[0], confidence=row[1], provider=row[2])

    def put_ocr(self, cache_key: str, result: OCRResult) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO ocr_cache(cache_key, text, confidence, provider) VALUES(?, ?, ?, ?)",
                (cache_key, result.text, result.confidence, result.provider),
            )

    def is_complete(self, source_key: str, source_hash: str, config_hash: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT output_path, report_path, status FROM processed_files "
                "WHERE source_key=? AND source_hash=? AND config_hash=?",
                (source_key, source_hash, config_hash),
            ).fetchone()
        return bool(row and row[2] == "complete" and Path(row[0]).exists() and Path(row[1]).exists())

    def mark_processed(
        self, source_key: str, source_hash: str, config_hash: str,
        output_path: Path, report_path: Path, status: str,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO processed_files"
                "(source_key, source_hash, config_hash, output_path, report_path, status) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (source_key, source_hash, config_hash, str(output_path), str(report_path), status),
            )
