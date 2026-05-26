from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from photo_cleaner.index import Index, PhotoRecord


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype("float32")
    return v / np.linalg.norm(v)


def _record(path: Path, sha: str, emb, edited="2010-06-12") -> PhotoRecord:
    return PhotoRecord(
        path=path,
        original_name="orig.jpg",
        sha256=sha,
        embedding=emb,
        detected_date="2010-06-12",
        edited_date=edited,
        date_action="change",
        processed_at="2026-05-21T10:00:00",
    )


def test_insert_and_load_matrix(tmp_path):
    db = tmp_path / "i.db"
    idx = Index(db)
    e1, e2 = _emb(1), _emb(2)
    idx.insert(_record(tmp_path / "_done_a.jpg", "a" * 64, e1))
    idx.insert(_record(tmp_path / "_done_b.jpg", "b" * 64, e2))
    matrix, shas, paths = idx.snapshot()
    assert matrix.shape == (2, 512)
    assert matrix.dtype == np.float32
    assert set(shas) == {"a" * 64, "b" * 64}
    assert len(paths) == 2


def test_settings_round_trip(tmp_path):
    idx = Index(tmp_path / "i.db")
    assert idx.get_setting("last_edited_date") is None
    idx.set_setting("last_edited_date", "2010-06-12")
    assert idx.get_setting("last_edited_date") == "2010-06-12"
    idx.set_setting("last_edited_date", "2011-01-01")
    assert idx.get_setting("last_edited_date") == "2011-01-01"


def test_insert_duplicate_path_raises(tmp_path):
    idx = Index(tmp_path / "i.db")
    e = _emb(1)
    idx.insert(_record(tmp_path / "_done_a.jpg", "a" * 64, e))
    with pytest.raises(Exception):
        idx.insert(_record(tmp_path / "_done_a.jpg", "a" * 64, e))


def test_find_by_sha256(tmp_path):
    idx = Index(tmp_path / "i.db")
    e = _emb(7)
    rec = _record(tmp_path / "_done_a.jpg", "f" * 64, e)
    idx.insert(rec)
    found = idx.find_by_sha256("f" * 64)
    assert found is not None
    assert found.path == rec.path
    assert idx.find_by_sha256("0" * 64) is None
