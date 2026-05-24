"""Shared test fixtures."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import piexif
import pytest
from PIL import Image


def _make_jpeg(path: Path, exif_dt: datetime | None = None, color=(255, 0, 0)) -> Path:
    """Create a tiny JPEG, optionally with EXIF DateTimeOriginal+Digitized set."""
    img = Image.new("RGB", (32, 32), color=color)
    exif_bytes = b""
    if exif_dt is not None:
        dt_str = exif_dt.strftime("%Y:%m:%d %H:%M:%S").encode("ascii")
        exif = {
            "0th": {},
            "Exif": {
                piexif.ExifIFD.DateTimeOriginal: dt_str,
                piexif.ExifIFD.DateTimeDigitized: dt_str,
            },
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
        exif_bytes = piexif.dump(exif)
    img.save(path, format="JPEG", exif=exif_bytes)
    return path


@pytest.fixture
def make_jpeg():
    """Factory fixture: make_jpeg(path, exif_dt=None, color=(r,g,b)) -> Path."""
    return _make_jpeg


@pytest.fixture
def tmp_pictures(tmp_path):
    """A tmp dir with a couple of subfolders, mimicking the Pictures layout."""
    (tmp_path / "2010-2011").mkdir()
    (tmp_path / "2012-2013").mkdir()
    return tmp_path
