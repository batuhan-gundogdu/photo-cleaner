from __future__ import annotations

from pathlib import Path

import numpy as np

from photo_cleaner.duplicate_finder import Match, find_duplicate, find_similar


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_find_duplicate_by_sha_takes_priority():
    matrix = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    shas = ["aa", "bb"]
    paths = [Path("/p/a.jpg"), Path("/p/b.jpg")]
    # Different embedding but matching sha -> still duplicate.
    q = _unit([0.5, 0.5, 0.5])
    m = find_duplicate(q, "bb", matrix, shas, paths)
    assert m is not None
    assert m.path == Path("/p/b.jpg")
    assert m.reason == "sha256"


def test_find_duplicate_by_cosine_when_near_one():
    matrix = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0])])
    shas = ["aa", "bb"]
    paths = [Path("/p/a.jpg"), Path("/p/b.jpg")]
    q = _unit([1, 0.0001, 0])
    m = find_duplicate(q, "zz", matrix, shas, paths)
    assert m is not None
    assert m.path == Path("/p/a.jpg")
    assert m.reason == "cosine"
    assert m.similarity > 0.999


def test_find_duplicate_returns_none_on_empty():
    matrix = np.zeros((0, 3), dtype="float32")
    assert find_duplicate(_unit([1, 0, 0]), "x", matrix, [], []) is None


def test_find_similar_returns_top_k_above_threshold():
    matrix = np.stack(
        [_unit([1, 0, 0]), _unit([0.9, 0.1, 0]), _unit([0, 1, 0]), _unit([0.7, 0.3, 0])]
    )
    paths = [Path(f"/p/{c}.jpg") for c in "abcd"]
    q = _unit([1, 0, 0])
    matches = find_similar(q, matrix, paths, threshold=0.95, k=10)
    # Expect "a" (cos=1), "b" (cos~0.994), skip "d" (cos~0.919), skip "c".
    got = [(m.path.name, round(m.similarity, 3)) for m in matches]
    assert got[0] == ("a.jpg", 1.0)
    assert got[1][0] == "b.jpg"
    assert len(matches) == 2


def test_find_similar_respects_k():
    matrix = np.stack([_unit([1, 0, 0]) for _ in range(20)])
    paths = [Path(f"/p/{i}.jpg") for i in range(20)]
    q = _unit([1, 0, 0])
    matches = find_similar(q, matrix, paths, threshold=0.0, k=5)
    assert len(matches) == 5
