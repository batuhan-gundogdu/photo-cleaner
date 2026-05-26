"""Recursive walk of the Pictures root that yields unprocessed image files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from photo_cleaner.config import DONE_PREFIX, SUPPORTED_EXTS


def iter_unprocessed(root: Path) -> Iterator[Path]:
    """Yield paths under ``root`` (recursive) for supported images that are
    not already prefixed with ``DONE_PREFIX``. Order is deterministic:
    sorted by (folder-as-string, filename).
    """
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if p.name.startswith(DONE_PREFIX):
            continue
        candidates.append(p)
    candidates.sort(key=lambda p: (str(p.parent), p.name))
    yield from candidates
