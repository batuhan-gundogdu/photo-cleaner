"""Write a chosen date back to an image file (EXIF + filesystem mtime)."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import piexif

_JPEG_EXTS = frozenset({".jpg", ".jpeg"})


@dataclass(frozen=True)
class WriteResult:
    exif_written: bool
    mtime_written: bool
    warning: str | None


def _set_mtime(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def _write_jpeg_exif(path: Path, dt: datetime) -> None:
    dt_bytes = dt.strftime("%Y:%m:%d %H:%M:%S").encode("ascii")
    try:
        exif = piexif.load(str(path))
    except Exception:
        exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif.setdefault("Exif", {})
    exif["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt_bytes
    exif["Exif"][piexif.ExifIFD.DateTimeDigitized] = dt_bytes
    exif_bytes = piexif.dump(exif)
    # Atomic-ish: write to a sibling temp file, then replace.
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".tmp_", suffix=path.suffix, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        piexif.insert(exif_bytes, str(path), str(tmp_path))
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write(path: Path, dt: datetime) -> WriteResult:
    """Write ``dt`` as EXIF DateTimeOriginal+Digitized (JPEG only) and as file
    mtime. For non-JPEG formats only mtime is updated, with a warning in the
    result.
    """
    suffix = path.suffix.lower()
    exif_written = False
    warning: str | None = None
    if suffix in _JPEG_EXTS:
        try:
            _write_jpeg_exif(path, dt)
            exif_written = True
        except Exception as exc:
            warning = f"EXIF write failed for {path.name}: {exc}"
    else:
        warning = f"{suffix} does not support EXIF write; only mtime updated"
    _set_mtime(path, dt)
    return WriteResult(exif_written=exif_written, mtime_written=True, warning=warning)
