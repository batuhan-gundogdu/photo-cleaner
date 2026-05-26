from __future__ import annotations

from pathlib import Path

import numpy as np

from photo_cleaner.config import DONE_PREFIX
from photo_cleaner.index import Index, PhotoRecord
from photo_cleaner.recovery import fix_up_pending_renames


def test_fix_up_renames_when_db_has_row_but_file_lacks_prefix(tmp_path, make_jpeg):
    # Pretend a previous run inserted the row, but crashed before renaming.
    p = make_jpeg(tmp_path / "x.jpg")
    sha = "a" * 64
    idx = Index(tmp_path / "i.db")
    # NOTE: the path stored in the DB is the *post-rename* path that the
    # previous run intended to write.
    intended = tmp_path / f"{DONE_PREFIX}x.jpg"
    v = np.ones(512, dtype="float32")
    v /= np.linalg.norm(v)
    idx.insert(
        PhotoRecord(
            path=intended,
            original_name="x.jpg",
            sha256=sha,
            embedding=v,
            detected_date="2010-01-01",
            edited_date="2010-01-01",
            date_action="keep",
            processed_at="2026-01-01T00:00:00",
        )
    )
    # Override the sha lookup by stubbing _sha256_of_file via patching:
    import photo_cleaner.recovery as rec_mod

    rec_mod._sha256_of_file = lambda _p: sha  # type: ignore[attr-defined]

    healed = fix_up_pending_renames(tmp_path, idx)
    assert healed == [intended]
    assert intended.exists()
    assert not p.exists()


def test_fix_up_noop_when_no_pending(tmp_path, make_jpeg):
    make_jpeg(tmp_path / "new.jpg")
    idx = Index(tmp_path / "i.db")
    assert fix_up_pending_renames(tmp_path, idx) == []
