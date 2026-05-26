from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from photo_cleaner.embedder import Embedder
from photo_cleaner.index import Index, PhotoRecord
from photo_cleaner.worker import PhotoBundle, Worker


def _stub_embedder(seed: int) -> Embedder:
    v = np.arange(512, dtype="float32") + seed
    v = v / np.linalg.norm(v)
    return Embedder.from_callable(lambda img, _v=v: _v.copy())


def _seed_index(idx: Index, tmp_path) -> np.ndarray:
    """Seed one record so the worker has something to compare against."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(512).astype("float32")
    v = v / np.linalg.norm(v)
    idx.insert(
        PhotoRecord(
            path=tmp_path / "_done_old.jpg",
            original_name="old.jpg",
            sha256="0" * 64,
            embedding=v,
            detected_date="2010-01-01",
            edited_date="2010-01-01",
            date_action="keep",
            processed_at="2026-01-01T00:00:00",
        )
    )
    return v


def test_worker_emits_photo_ready(qtbot, tmp_path, make_jpeg):
    p = make_jpeg(tmp_path / "new.jpg", exif_dt=datetime(2010, 6, 12))
    idx = Index(tmp_path / "i.db")
    _seed_index(idx, tmp_path)
    embedder = _stub_embedder(1)

    worker = Worker(embedder=embedder, index=idx)
    worker.start()
    try:
        with qtbot.waitSignal(worker.photo_ready, timeout=5000) as blocker:
            worker.enqueue(p, threshold=0.92)
        bundle = blocker.args[0]
    finally:
        worker.stop()
        worker.wait(5000)

    assert isinstance(bundle, PhotoBundle)
    assert bundle.path == p
    assert bundle.date_info.earliest == datetime(2010, 6, 12)
    assert bundle.embedding.shape == (512,)
    assert bundle.duplicate is None  # different stub embedding from seed
    assert isinstance(bundle.similar, list)
    assert bundle.thumbnail_main_png and isinstance(bundle.thumbnail_main_png, bytes)
    assert bundle.sha256 and len(bundle.sha256) == 64
