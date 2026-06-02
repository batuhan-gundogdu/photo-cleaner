"""SQLite-backed index of processed photos with an in-RAM embeddings matrix.

The connection is opened with ``check_same_thread=False`` because the UI
thread writes (insert / set_setting / find_by_sha256) and the worker thread
reads (snapshot). A ``threading.Lock`` serializes access to the connection
to satisfy Python's sqlite3 module — SQLite itself handles concurrency via
WAL.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from photo_cleaner.config import EMBED_DIM

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS photos (
  id            INTEGER PRIMARY KEY,
  path          TEXT NOT NULL UNIQUE,
  original_name TEXT NOT NULL,
  sha256        TEXT NOT NULL,
  embedding     BLOB NOT NULL,
  detected_date TEXT,
  edited_date   TEXT NOT NULL,
  date_action   TEXT NOT NULL CHECK (date_action IN ('keep','change')),
  processed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_sha256 ON photos(sha256);

CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class PhotoRecord:
    path: Path
    original_name: str
    sha256: str
    embedding: np.ndarray  # (EMBED_DIM,) float32, L2-normalized
    detected_date: str | None
    edited_date: str
    date_action: str  # 'keep' | 'change'
    processed_at: str


class Index:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----- photos -----

    def insert(self, rec: PhotoRecord) -> None:
        if rec.embedding.shape != (EMBED_DIM,) or rec.embedding.dtype != np.float32:
            raise ValueError("embedding must be float32 shape (EMBED_DIM,)")
        params = (
            str(rec.path),
            rec.original_name,
            rec.sha256,
            rec.embedding.tobytes(),
            rec.detected_date,
            rec.edited_date,
            rec.date_action,
            rec.processed_at,
        )
        self._with_retry(
            lambda: self._conn.execute(
                """INSERT INTO photos
                   (path, original_name, sha256, embedding, detected_date,
                    edited_date, date_action, processed_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                params,
            )
        )

    def find_by_path(self, path: Path) -> PhotoRecord | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT path, original_name, sha256, embedding, detected_date,
                          edited_date, date_action, processed_at
                   FROM photos WHERE path = ? LIMIT 1""",
                (str(path),),
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def delete_by_path(self, path: Path) -> None:
        self._with_retry(
            lambda: self._conn.execute(
                "DELETE FROM photos WHERE path = ?", (str(path),)
            )
        )

    def find_by_sha256(self, sha: str) -> PhotoRecord | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT path, original_name, sha256, embedding, detected_date,
                          edited_date, date_action, processed_at
                   FROM photos WHERE sha256 = ? LIMIT 1""",
                (sha,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def snapshot(self) -> tuple[np.ndarray, list[str], list[Path]]:
        """Return (matrix (N,EMBED_DIM), sha256_list, path_list) sorted by id."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT embedding, sha256, path FROM photos ORDER BY id"
            ).fetchall()
        if not rows:
            return (
                np.zeros((0, EMBED_DIM), dtype=np.float32),
                [],
                [],
            )
        mat = np.stack(
            [np.frombuffer(r[0], dtype=np.float32) for r in rows], axis=0
        )
        shas = [r[1] for r in rows]
        paths = [Path(r[2]) for r in rows]
        return mat, shas, paths

    # ----- settings -----

    def get_setting(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_setting(self, key: str, value: str) -> None:
        self._with_retry(
            lambda: self._conn.execute(
                """INSERT INTO settings(key, value) VALUES(?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )
        )

    # ----- internals -----

    def _with_retry(self, op) -> None:
        """Run a write op under the lock with exponential backoff on lock errors."""
        delays = (0.05, 0.2, 0.5)
        last_exc: Exception | None = None
        for attempt in range(len(delays) + 1):
            try:
                with self._lock:
                    op()
                    self._conn.commit()
                return
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if attempt == len(delays):
                    break
                time.sleep(delays[attempt])
        raise last_exc  # type: ignore[misc]


def _row_to_record(row) -> PhotoRecord:
    return PhotoRecord(
        path=Path(row[0]),
        original_name=row[1],
        sha256=row[2],
        embedding=np.frombuffer(row[3], dtype=np.float32),
        detected_date=row[4],
        edited_date=row[5],
        date_action=row[6],
        processed_at=row[7],
    )
