from __future__ import annotations

from datetime import datetime
from pathlib import Path

import piexif
from PIL import Image

from photo_cleaner.date_writer import WriteResult, write


def test_write_updates_jpeg_exif_and_mtime(tmp_path, make_jpeg):
    p = make_jpeg(tmp_path / "x.jpg", exif_dt=datetime(2020, 1, 1))
    new_dt = datetime(2010, 6, 12, 13, 45, 22)
    result = write(p, new_dt)
    assert isinstance(result, WriteResult)
    assert result.exif_written is True
    assert result.mtime_written is True
    assert result.warning is None
    # Verify on disk:
    exif = piexif.load(str(p))
    raw = exif["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("ascii")
    assert raw == "2010:06:12 13:45:22"
    # mtime should reflect new_dt (allow 1s tolerance):
    assert abs(p.stat().st_mtime - new_dt.timestamp()) < 1.0


def test_write_png_writes_exif_and_mtime(tmp_path):
    p = tmp_path / "y.png"
    Image.new("RGB", (8, 8)).save(p, format="PNG")
    new_dt = datetime(2015, 3, 10)
    result = write(p, new_dt)
    assert result.exif_written is True
    assert result.mtime_written is True
    assert result.warning is None
    img = Image.open(p)
    exif = piexif.load(img.info["exif"])
    raw = exif["Exif"][piexif.ExifIFD.DateTimeOriginal].decode("ascii")
    assert raw == "2015:03:10 00:00:00"
    assert abs(p.stat().st_mtime - new_dt.timestamp()) < 1.0
