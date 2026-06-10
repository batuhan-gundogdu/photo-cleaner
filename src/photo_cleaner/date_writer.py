"""Write a chosen date back to an image file (EXIF + filesystem mtime)."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import piexif
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

_JPEG_EXTS = frozenset({".jpg", ".jpeg"})
_HEIC_EXTS = frozenset({".heic"})
_PNG_EXTS  = frozenset({".png"})


@dataclass(frozen=True)
class WriteResult:
    exif_written: bool
    mtime_written: bool
    warning: str | None


def _set_mtime(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def _make_exif_bytes(dt: datetime, existing_raw: bytes | None = None) -> bytes:
    """Build a piexif-serialisable EXIF dict with DateTimeOriginal set to *dt*.

    Loads *existing_raw* so other tags are preserved; falls back to a fresh
    dict if loading or dumping fails (e.g. malformed original EXIF).
    """
    dt_bytes = dt.strftime("%Y:%m:%d %H:%M:%S").encode("ascii")

    def _apply(d: dict) -> bytes:
        d.setdefault("0th",  {})[piexif.ImageIFD.DateTime]            = dt_bytes
        d.setdefault("Exif", {})[piexif.ExifIFD.DateTimeOriginal]     = dt_bytes
        d["Exif"][piexif.ExifIFD.DateTimeDigitized]                   = dt_bytes
        return piexif.dump(d)

    if existing_raw:
        try:
            return _apply(piexif.load(existing_raw))
        except Exception:
            pass  # malformed — fall through to fresh dict

    return _apply({"0th": {}, "Exif": {}, "GPS": {}, "1st": {}})


def _write_jpeg_exif(path: Path, dt: datetime) -> None:
    dt_bytes = dt.strftime("%Y:%m:%d %H:%M:%S").encode("ascii")

    try:
        exif_dict = piexif.load(str(path))
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    exif_dict.setdefault("0th",  {})[piexif.ImageIFD.DateTime]        = dt_bytes
    exif_dict.setdefault("Exif", {})[piexif.ExifIFD.DateTimeOriginal] = dt_bytes
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized]               = dt_bytes

    try:
        exif_bytes = piexif.dump(exif_dict)
    except Exception:
        # Malformed original EXIF — start fresh, preserving only our tags
        fresh = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
        fresh["0th"][piexif.ImageIFD.DateTime]             = dt_bytes
        fresh["Exif"][piexif.ExifIFD.DateTimeOriginal]     = dt_bytes
        fresh["Exif"][piexif.ExifIFD.DateTimeDigitized]    = dt_bytes
        exif_bytes = piexif.dump(fresh)

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


def _write_heic_exif(path: Path, dt: datetime) -> None:
    img = Image.open(path)
    raw = img.info.get("exif", b"") or b""
    exif_bytes = _make_exif_bytes(dt, raw if raw else None)
    ts = path.stat().st_mtime
    img.save(path, format="HEIF", exif=exif_bytes, quality=100)
    os.utime(path, (ts, ts))


def _write_png_exif(path: Path, dt: datetime) -> None:
    img = Image.open(path)
    raw = img.info.get("exif", b"") or b""
    exif_bytes = _make_exif_bytes(dt, raw if raw else None)
    ts = path.stat().st_mtime
    img.save(path, format="PNG", exif=exif_bytes)
    os.utime(path, (ts, ts))


def write(path: Path, dt: datetime) -> WriteResult:
    """Write *dt* as EXIF DateTimeOriginal/Digitized/DateTime and as file mtime."""
    suffix = path.suffix.lower()
    exif_written = False
    warning: str | None = None

    try:
        if suffix in _JPEG_EXTS:
            _write_jpeg_exif(path, dt)
            exif_written = True
        elif suffix in _HEIC_EXTS:
            _write_heic_exif(path, dt)
            exif_written = True
        elif suffix in _PNG_EXTS:
            _write_png_exif(path, dt)
            exif_written = True
        else:
            warning = f"{suffix} does not support EXIF write; only mtime updated"
    except Exception as exc:
        warning = f"EXIF write failed for {path.name}: {exc}"

    _set_mtime(path, dt)
    return WriteResult(exif_written=exif_written, mtime_written=True, warning=warning)
