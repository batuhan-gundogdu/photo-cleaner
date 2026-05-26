"""Heal partial state from a crash between DB insert and filesystem rename."""
from __future__ import annotations

import hashlib
from pathlib import Path

from photo_cleaner.config import DONE_PREFIX
from photo_cleaner.index import Index
from photo_cleaner.scanner import iter_unprocessed


def fix_up_pending_renames(root: Path, index: Index) -> list[Path]:
    """Walk ``root`` for unprocessed files; for each, if a DB row exists with
    a matching SHA256, perform the missing rename. Returns the list of
    intended (post-rename) paths that were healed.
    """
    healed: list[Path] = []
    for p in iter_unprocessed(root):
        sha = _sha256_of_file(p)
        rec = index.find_by_sha256(sha)
        if rec is None:
            continue
        intended = p.parent / f"{DONE_PREFIX}{p.name}"
        if rec.path != intended:
            # The DB has a row for this content but at a different intended
            # path (e.g., file moved between runs). Best to leave it alone
            # and let the user reprocess.
            continue
        p.rename(intended)
        healed.append(intended)
    return healed


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
