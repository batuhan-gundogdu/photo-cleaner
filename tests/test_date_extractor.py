from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from photo_cleaner.date_extractor import (
    DateInfo,
    extract,
    parse_filename_date,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("IMG_20100612_134522.jpg", datetime(2010, 6, 12, 13, 45, 22)),
        ("IMG_20100612.jpg", datetime(2010, 6, 12)),
        ("2011-08-04.png", datetime(2011, 8, 4)),
        ("20120101_000000.heic", datetime(2012, 1, 1, 0, 0, 0)),
        ("vacation_pic.jpg", None),
        ("IMG_99999999.jpg", None),
    ],
)
def test_parse_filename_date(name, expected):
    assert parse_filename_date(name) == expected


def test_extract_uses_exif_when_present(tmp_path, make_jpeg):
    exif_dt = datetime(2010, 5, 15, 12, 0, 0)
    p = make_jpeg(tmp_path / "IMG_20200101.jpg", exif_dt=exif_dt)
    # mtime is "now"; filename says 2020; EXIF says 2010 -> earliest is 2010.
    info = extract(p)
    assert info.earliest == exif_dt
    assert info.earliest_source in {"exif_original", "exif_digitized"}
    assert "exif_original" in info.sources
    assert info.sources["filename"] == datetime(2020, 1, 1)


def test_extract_falls_back_when_no_exif(tmp_path, make_jpeg):
    p = make_jpeg(tmp_path / "vacation.jpg", exif_dt=None)
    info = extract(p)
    # No EXIF, no filename date -> earliest must come from mtime or birthtime.
    assert info.earliest_source in {"file_mtime", "file_birthtime"}
    assert "filename" not in info.sources


def test_extract_uses_filename_when_older_than_file_times(tmp_path, make_jpeg):
    p = make_jpeg(tmp_path / "IMG_20051231_235959.jpg", exif_dt=None)
    info = extract(p)
    assert info.earliest == datetime(2005, 12, 31, 23, 59, 59)
    assert info.earliest_source == "filename"


def test_extract_handles_image_without_exif_block(tmp_path):
    # A PNG has no EXIF infrastructure at all in our writer; extract must not raise.
    from PIL import Image
    p = tmp_path / "foo.png"
    Image.new("RGB", (8, 8)).save(p, format="PNG")
    info = extract(p)
    assert info.earliest_source in {"file_mtime", "file_birthtime"}
