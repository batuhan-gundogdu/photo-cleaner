#!/usr/bin/env python3
"""
One-time recovery script: writes edited_date from the photo-cleaner database
into the embedded EXIF (DateTimeOriginal) of every processed photo that is
missing that tag — covering JPEGs where date_action='keep' was never written,
as well as HEIC and PNG files which were always skipped.

Preferred path  : exiftool (no re-encoding — install from exiftool.org)
Fallback path   : pillow / pillow_heif (re-encodes HEIC, losslessly rewrites PNG,
                  patches JPEG in-place via piexif)

Usage:
    python fix_heic_png_dates.py                   # uses default PICTURES_ROOT
    python fix_heic_png_dates.py /path/to/folder   # custom root
    python fix_heic_png_dates.py --dry-run         # show what would change, no writes
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PICTURES_ROOT = Path.home() / "Google Drive" / "My Drive" / "Pictures"
DONE_PREFIX = "_done_"
JPEG_EXTS = frozenset({".jpg", ".jpeg"})
HEIC_EXTS = frozenset({".heic"})
PNG_EXTS  = frozenset({".png"})
ALL_EXTS  = JPEG_EXTS | HEIC_EXTS | PNG_EXTS

EXIFTOOL = shutil.which("exiftool")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_path(db_path_str: str) -> Path | None:
    original = Path(db_path_str)
    if original.exists():
        return original
    stripped = original.parent / original.name.removeprefix(DONE_PREFIX)
    if stripped.exists():
        return stripped
    return None


# ---------------------------------------------------------------------------
# EXIF reading
# ---------------------------------------------------------------------------

def has_datetime_original(path: Path) -> bool:
    """Return True if the file already has a non-empty DateTimeOriginal tag."""
    import piexif
    suffix = path.suffix.lower()
    if suffix in HEIC_EXTS:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            from PIL import Image
            img = Image.open(path)
            raw = img.info.get("exif", b"")
            if not raw:
                return False
            exif = piexif.load(raw)
            return bool(exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal))
        except Exception:
            return False
    try:
        exif = piexif.load(str(path))
        return bool(exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def fix_with_exiftool(path: Path, dt: datetime) -> None:
    dt_str = dt.strftime("%Y:%m:%d %H:%M:%S")
    subprocess.run(
        [
            EXIFTOOL,
            f"-DateTimeOriginal={dt_str}",
            f"-DateTimeDigitized={dt_str}",
            f"-DateTime={dt_str}",
            "-overwrite_original",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def fix_jpeg_piexif(path: Path, dt: datetime) -> None:
    import piexif
    import tempfile

    dt_bytes = dt.strftime("%Y:%m:%d %H:%M:%S").encode("ascii")
    try:
        exif = piexif.load(str(path))
    except Exception:
        exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    exif.setdefault("0th",  {})[piexif.ImageIFD.DateTime]           = dt_bytes
    exif.setdefault("Exif", {})[piexif.ExifIFD.DateTimeOriginal]    = dt_bytes
    exif["Exif"][piexif.ExifIFD.DateTimeDigitized]                  = dt_bytes

    try:
        exif_bytes = piexif.dump(exif)
    except Exception:
        # Malformed original EXIF — start fresh
        fresh = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
        fresh["0th"][piexif.ImageIFD.DateTime]        = dt_bytes
        fresh["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt_bytes
        fresh["Exif"][piexif.ExifIFD.DateTimeDigitized] = dt_bytes
        exif_bytes = piexif.dump(fresh)

    ts = path.stat().st_mtime
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
    os.utime(path, (ts, ts))


def fix_heic_or_png_pillow(path: Path, dt: datetime) -> None:
    import piexif
    from PIL import Image

    suffix = path.suffix.lower()
    if suffix in HEIC_EXTS:
        import pillow_heif
        pillow_heif.register_heif_opener()

    dt_str   = dt.strftime("%Y:%m:%d %H:%M:%S")
    dt_bytes = dt_str.encode("ascii")
    img = Image.open(path)

    try:
        raw = img.info.get("exif", b"")
        exif_dict = piexif.load(raw) if raw else {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

    exif_dict.setdefault("0th",  {})[piexif.ImageIFD.DateTime]           = dt_bytes
    exif_dict.setdefault("Exif", {})[piexif.ExifIFD.DateTimeOriginal]    = dt_bytes
    exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized]                  = dt_bytes
    exif_bytes = piexif.dump(exif_dict)

    ts = path.stat().st_mtime
    fmt = "HEIF" if suffix in HEIC_EXTS else "PNG"
    kwargs = {"quality": 100} if suffix in HEIC_EXTS else {}
    img.save(path, format=fmt, exif=exif_bytes, **kwargs)
    os.utime(path, (ts, ts))


def fix_file(path: Path, dt: datetime, use_exiftool: bool) -> None:
    if use_exiftool:
        fix_with_exiftool(path, dt)
    elif path.suffix.lower() in JPEG_EXTS:
        fix_jpeg_piexif(path, dt)
    else:
        fix_heic_or_png_pillow(path, dt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    root = Path(args[0]) if args else PICTURES_ROOT
    db_path = root / "index.db"

    if not db_path.exists():
        sys.exit(f"ERROR: database not found at {db_path}")

    use_exiftool = EXIFTOOL is not None
    method = "exiftool" if use_exiftool else "pillow/piexif"
    print(f"Root    : {root}")
    print(f"Method  : {method}")
    print(f"Dry run : {dry_run}")
    if not use_exiftool:
        print("TIP: install exiftool from exiftool.org for lossless HEIC metadata patching.\n")

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT path, edited_date FROM photos WHERE "
        + " OR ".join(f"lower(path) LIKE '%{ext}'" for ext in ALL_EXTS)
    ).fetchall()
    conn.close()

    print(f"Found {len(rows)} processed photo records in DB.\n")

    ok = skipped = fail = missing = suspicious = 0
    SUSPICIOUS_THRESHOLD = datetime(2025, 1, 1)

    for db_path_str, edited_date in rows:
        actual = resolve_path(db_path_str)
        if actual is None:
            missing += 1
            continue

        dt = datetime.fromisoformat(edited_date)

        # Warn about dates that look like they came from a file-system timestamp
        # rather than a real photo date (e.g., Facebook download dates).
        if dt >= SUSPICIOUS_THRESHOLD:
            print(f"WARN  {actual.name}  →  {edited_date}  (date looks wrong — skip and review manually)")
            suspicious += 1
            continue

        # Skip files that already have DateTimeOriginal set correctly.
        if not dry_run and has_datetime_original(actual):
            skipped += 1
            continue

        label = edited_date[:10]
        if dry_run:
            print(f"WOULD  {actual.name}  →  {label}")
            ok += 1
            continue

        try:
            fix_file(actual, dt, use_exiftool)
            print(f"OK     {actual.name}  →  {label}")
            ok += 1
        except Exception as exc:
            print(f"FAIL   {actual.name}  →  {exc}")
            fail += 1

    print(f"\nDone: {ok} {'would be ' if dry_run else ''}updated, "
          f"{skipped} already had EXIF, "
          f"{suspicious} suspicious dates (need manual review), "
          f"{fail} failed, "
          f"{missing} not on disk.")


if __name__ == "__main__":
    main()
