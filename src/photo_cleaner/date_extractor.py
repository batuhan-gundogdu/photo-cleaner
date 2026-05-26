"""Determine the earliest plausible creation date of a photo file.

Sources, in priority order for tie-breaking only (the chosen date is
always the *earliest* across all sources):
    exif_original, exif_digitized, file_mtime, file_birthtime, filename
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import piexif

# IMG_YYYYMMDD or IMG_YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS
_RE_IMG_DT = re.compile(r"(?:IMG_)?(\d{8})(?:_(\d{6}))?")
# YYYY-MM-DD
_RE_DASHED = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def parse_filename_date(name: str) -> datetime | None:
    """Best-effort parse of a date out of a filename. Returns None if no
    plausible date can be parsed.
    """
    stem = Path(name).stem
    m = _RE_DASHED.search(stem)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = _RE_IMG_DT.search(stem)
    if m:
        date_part = m.group(1)
        time_part = m.group(2)
        try:
            y, mo, d = int(date_part[0:4]), int(date_part[4:6]), int(date_part[6:8])
            if time_part:
                h, mi, s = int(time_part[0:2]), int(time_part[2:4]), int(time_part[4:6])
                return datetime(y, mo, d, h, mi, s)
            return datetime(y, mo, d)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class DateInfo:
    earliest: datetime
    earliest_source: str  # name of the source that produced ``earliest``
    sources: dict[str, datetime] = field(default_factory=dict)


def _read_exif_dates(path: Path) -> dict[str, datetime]:
    out: dict[str, datetime] = {}
    try:
        exif = piexif.load(str(path))
    except Exception:
        return out
    exif_ifd = exif.get("Exif", {}) or {}
    for key, name in (
        (piexif.ExifIFD.DateTimeOriginal, "exif_original"),
        (piexif.ExifIFD.DateTimeDigitized, "exif_digitized"),
    ):
        raw = exif_ifd.get(key)
        if not raw:
            continue
        try:
            text = raw.decode("ascii") if isinstance(raw, bytes) else str(raw)
            out[name] = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
        except (ValueError, UnicodeDecodeError):
            continue
    return out


def _read_file_times(path: Path) -> dict[str, datetime]:
    st = path.stat()
    out = {"file_mtime": datetime.fromtimestamp(st.st_mtime)}
    bt = getattr(st, "st_birthtime", None)
    if bt is not None:
        out["file_birthtime"] = datetime.fromtimestamp(bt)
    return out


def extract(path: Path) -> DateInfo:
    sources: dict[str, datetime] = {}
    sources.update(_read_exif_dates(path))
    sources.update(_read_file_times(path))
    fn_dt = parse_filename_date(path.name)
    if fn_dt is not None:
        sources["filename"] = fn_dt
    if not sources:  # truly impossible (file_mtime always works), but defensive
        raise RuntimeError(f"no date sources for {path}")
    earliest_source = min(sources, key=lambda k: sources[k])
    return DateInfo(
        earliest=sources[earliest_source],
        earliest_source=earliest_source,
        sources=sources,
    )
