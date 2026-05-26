from __future__ import annotations

from pathlib import Path

from photo_cleaner.scanner import iter_unprocessed


def test_iter_unprocessed_sorted_skips_done_and_unsupported(tmp_path, make_jpeg):
    # Mixed layout: two folders, varied filenames, one already-processed file,
    # one unsupported extension, one nested subfolder.
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "nested").mkdir()

    p_a2 = make_jpeg(tmp_path / "a" / "b.jpg")
    p_a1 = make_jpeg(tmp_path / "a" / "a.jpg")
    p_b1 = make_jpeg(tmp_path / "b" / "c.jpg")
    p_b_nested = make_jpeg(tmp_path / "b" / "nested" / "d.jpg")
    # These must be skipped:
    make_jpeg(tmp_path / "a" / "_done_old.jpg")
    (tmp_path / "a" / "readme.txt").write_text("ignore me")

    result = list(iter_unprocessed(tmp_path))

    # Sorted by (folder, filename):
    assert result == [p_a1, p_a2, p_b1, p_b_nested]


def test_iter_unprocessed_empty_root(tmp_path):
    assert list(iter_unprocessed(tmp_path)) == []
