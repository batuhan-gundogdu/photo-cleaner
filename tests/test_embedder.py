from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from photo_cleaner.embedder import Embedder


def _img():
    return Image.new("RGB", (32, 32), color=(10, 20, 30))


def test_stub_embedder_returns_expected_shape_and_norm():
    fixed = np.arange(512, dtype="float32") + 1.0
    fixed = fixed / np.linalg.norm(fixed)
    e = Embedder.from_callable(lambda img: fixed.copy())
    v = e.embed(_img())
    assert v.dtype == np.float32
    assert v.shape == (512,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_embed_normalizes_unnormalized_input():
    raw = np.ones(512, dtype="float32") * 3.0
    e = Embedder.from_callable(lambda img: raw.copy())
    v = e.embed(_img())
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


@pytest.mark.slow
def test_real_model_produces_unit_512_vector():
    e = Embedder()
    v1 = e.embed(_img())
    v2 = e.embed(_img())
    assert v1.shape == (512,)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-4
    # Same input -> identical output (model is deterministic in eval mode).
    assert np.allclose(v1, v2, atol=1e-5)
