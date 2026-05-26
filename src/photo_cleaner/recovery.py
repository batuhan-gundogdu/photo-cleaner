"""Heal partial state from a crash between DB insert and filesystem rename."""
from __future__ import annotations

import hashlib
from pathlib import Path

from photo_cleaner.config import DONE_PREFIX
from photo_cleaner.index import Index


def fix_up_pending_renames(root: Path, index: Index) -> list[Path]:
    """Heal photos whose DB row was inserted but rename never happened.

    Iterates DB rows (small) rather than the full library: for each row whose
    intended path starts with ``DONE_PREFIX``, check if the un-prefixed sibling
    exists on disk and the prefixed target does not. If so, verify SHA256
    matches the row, then perform the missing rename. Skipping the library walk
    keeps startup fast on big libraries (e.g. 8k+ Google Drive photos).
    """
    del root  # kept for API stability; new algorithm doesn't need it
    _matrix, shas, paths = index.snapshot()
    if not paths:
        return []
    healed: list[Path] = []
    for db_path, db_sha in zip(paths, shas):
        if not db_path.name.startswith(DONE_PREFIX):
            continue
        unprefixed = db_path.parent / db_path.name[len(DONE_PREFIX):]
        if not unprefixed.exists() or db_path.exists():
            continue
        if _sha256_of_file(unprefixed) != db_sha:
            # Different content under the un-prefixed name — leave it alone.
            continue
        unprefixed.rename(db_path)
        healed.append(db_path)
    return healed


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
