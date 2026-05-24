# photo-cleaner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the interactive PyQt6 desktop app described in `docs/superpowers/specs/2026-05-21-photo-cleaner-design.md` — walk `~/Google Drive/My Drive/Pictures`, let the user correct creation dates, surface near-duplicates and visually-similar photos via a CLIP embedding index, and mark processed files with a `_done_` prefix so reruns resume cleanly.

**Architecture:** Single Python process, PyQt6 UI on the main thread, one `QThread` worker that prefetches one photo ahead of the user (decode + EXIF + SHA256 + embed + index query). Persistent state lives in a SQLite DB plus an in-RAM `(N, 512) float32` matrix of embeddings for sub-millisecond similarity queries.

**Tech Stack:** Python 3.11+, PyQt6, Pillow + pillow-heif, piexif, open_clip_torch, torch (CPU), numpy, send2trash, pytest, pytest-qt.

---

## File Map

**Created:**
- `pyproject.toml`
- `.gitignore` (extend the existing one)
- `README.md`
- `src/photo_cleaner/__init__.py`
- `src/photo_cleaner/__main__.py`
- `src/photo_cleaner/config.py`
- `src/photo_cleaner/scanner.py`
- `src/photo_cleaner/date_extractor.py`
- `src/photo_cleaner/date_writer.py`
- `src/photo_cleaner/embedder.py`
- `src/photo_cleaner/index.py`
- `src/photo_cleaner/duplicate_finder.py`
- `src/photo_cleaner/thumbnailer.py`
- `src/photo_cleaner/worker.py`
- `src/photo_cleaner/recovery.py`
- `src/photo_cleaner/ui/__init__.py`
- `src/photo_cleaner/ui/date_panel.py`
- `src/photo_cleaner/ui/similar_grid.py`
- `src/photo_cleaner/ui/duplicate_strip.py`
- `src/photo_cleaner/ui/toast.py`
- `src/photo_cleaner/ui/main_window.py`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_scanner.py`
- `tests/test_date_extractor.py`
- `tests/test_date_writer.py`
- `tests/test_embedder.py`
- `tests/test_index.py`
- `tests/test_duplicate_finder.py`
- `tests/test_thumbnailer.py`
- `tests/test_worker.py`
- `tests/test_recovery.py`

**Modified:**
- `.gitignore` (add Python + venv + DB patterns)

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/photo_cleaner/__init__.py`
- Create: `src/photo_cleaner/ui/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Extend `.gitignore`**

Replace the contents of `.gitignore` with:

```
.superpowers/
__pycache__/
*.pyc
.venv/
.pytest_cache/
*.egg-info/
build/
dist/
~/.photo-cleaner/
errors.log
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "photo-cleaner"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "PyQt6>=6.6",
  "Pillow>=10.0",
  "pillow-heif>=0.16",
  "piexif>=1.1.3",
  "open_clip_torch>=2.24",
  "torch>=2.2",
  "numpy>=1.26",
  "send2trash>=1.8",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-qt>=4.4",
]

[project.scripts]
photo-cleaner = "photo_cleaner.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "slow: tests that load the real CLIP model (skipped by default)",
]
addopts = "-m 'not slow'"
```

- [ ] **Step 3: Create empty package files**

```python
# src/photo_cleaner/__init__.py
```

```python
# src/photo_cleaner/ui/__init__.py
```

```python
# tests/__init__.py
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 5: Write a minimal `README.md`**

```markdown
# photo-cleaner

Interactive desktop tool to clean up `~/Google Drive/My Drive/Pictures`:
fix creation dates, surface near-duplicates and visually-similar photos, and
mark processed files with a `_done_` prefix so reruns resume cleanly.

## Install

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

## Run

    python -m photo_cleaner

## Test

    pytest -q

## Manual smoke test

See `docs/superpowers/specs/2026-05-21-photo-cleaner-design.md`.
```

- [ ] **Step 6: Verify the scaffold installs**

Run:

```bash
cd ~/Desktop/photo-cleaner
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Expected: `no tests ran` (no tests yet) and pip install completes cleanly.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore README.md src tests
git commit -m "Scaffold photo-cleaner package + dev deps"
```

---

## Task 2: `config.py`

**Files:**
- Create: `src/photo_cleaner/config.py`

This is constants only — no behavior to test, used by every other module.

- [ ] **Step 1: Write `src/photo_cleaner/config.py`**

```python
"""App-wide constants and paths."""
from __future__ import annotations

from pathlib import Path

PICTURES_ROOT = Path.home() / "Google Drive" / "My Drive" / "Pictures"
APP_DATA = Path.home() / ".photo-cleaner"
DB_PATH = APP_DATA / "index.db"
ERROR_LOG = APP_DATA / "errors.log"

DONE_PREFIX = "_done_"
SUPPORTED_EXTS = frozenset({".jpg", ".jpeg", ".png", ".heic"})

EMBED_DIM = 512
DUP_THRESHOLD = 0.999
SIM_THRESHOLD_DEFAULT = 0.92
SIM_THRESHOLD_MIN = 0.80
SIM_THRESHOLD_MAX = 0.99
SIM_TOP_K = 12

THUMB_MAIN = 800
THUMB_GRID = 160

# open_clip model — small, CPU-friendly, 512-d embeddings.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_MODEL_PRETRAINED = "openai"


def ensure_app_data() -> None:
    """Create APP_DATA if missing. Called once at startup."""
    APP_DATA.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Commit**

```bash
git add src/photo_cleaner/config.py
git commit -m "Add config module (paths + constants)"
```

---

## Task 3: `scanner.py` — recursive walk that skips processed files

**Files:**
- Create: `src/photo_cleaner/scanner.py`
- Create: `tests/test_scanner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scanner.py
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
    assert result == [p_a1, p_a2, p_b_nested, p_b1]


def test_iter_unprocessed_empty_root(tmp_path):
    assert list(iter_unprocessed(tmp_path)) == []
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_scanner.py -v
```

Expected: `ModuleNotFoundError: No module named 'photo_cleaner.scanner'`.

- [ ] **Step 3: Implement `src/photo_cleaner/scanner.py`**

```python
"""Recursive walk of the Pictures root that yields unprocessed image files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from photo_cleaner.config import DONE_PREFIX, SUPPORTED_EXTS


def iter_unprocessed(root: Path) -> Iterator[Path]:
    """Yield paths under ``root`` (recursive) for supported images that are
    not already prefixed with ``DONE_PREFIX``. Order is deterministic:
    sorted by (folder-as-string, filename).
    """
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if p.name.startswith(DONE_PREFIX):
            continue
        candidates.append(p)
    candidates.sort(key=lambda p: (str(p.parent), p.name))
    yield from candidates
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_scanner.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/photo_cleaner/scanner.py tests/test_scanner.py
git commit -m "Add scanner: recursive walk, skip _done_-prefixed and unsupported files"
```

---

## Task 4: `date_extractor.py` — earliest plausible date from multiple sources

**Files:**
- Create: `src/photo_cleaner/date_extractor.py`
- Create: `tests/test_date_extractor.py`

The extractor returns the earliest date across: EXIF DateTimeOriginal, EXIF DateTimeDigitized, file mtime, file birthtime (macOS), and a parsed filename. We TDD this in three rounds: filename parsing, then EXIF, then full integration.

- [ ] **Step 1: Write the failing test (round 1: filename parsing only)**

```python
# tests/test_date_extractor.py
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_date_extractor.py::test_parse_filename_date -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `date_extractor.py` skeleton + filename parser**

```python
# src/photo_cleaner/date_extractor.py
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


def extract(path: Path) -> DateInfo:
    """Read all date sources for ``path`` and return the earliest."""
    raise NotImplementedError  # filled in next step
```

- [ ] **Step 4: Run filename test, verify pass**

```bash
pytest tests/test_date_extractor.py::test_parse_filename_date -v
```

Expected: all parametrized cases pass.

- [ ] **Step 5: Write the failing test (round 2: EXIF + integration)**

Append to `tests/test_date_extractor.py`:

```python
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
```

- [ ] **Step 6: Run tests, verify they fail with NotImplementedError**

```bash
pytest tests/test_date_extractor.py -v
```

Expected: the 4 new tests fail with `NotImplementedError`.

- [ ] **Step 7: Implement `extract()`**

Replace the `extract` stub in `date_extractor.py`:

```python
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
```

- [ ] **Step 8: Run all date-extractor tests, verify pass**

```bash
pytest tests/test_date_extractor.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/photo_cleaner/date_extractor.py tests/test_date_extractor.py
git commit -m "Add date_extractor: earliest of EXIF, file times, filename parse"
```

---

## Task 5: `date_writer.py` — write EXIF + mtime

**Files:**
- Create: `src/photo_cleaner/date_writer.py`
- Create: `tests/test_date_writer.py`

JPEG gets EXIF + mtime. HEIC/PNG fall back to mtime-only with a `warning` in the result. Use atomic temp-file swap for EXIF writes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_date_writer.py
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


def test_write_png_falls_back_to_mtime_only(tmp_path):
    p = tmp_path / "y.png"
    Image.new("RGB", (8, 8)).save(p, format="PNG")
    new_dt = datetime(2015, 3, 10)
    result = write(p, new_dt)
    assert result.exif_written is False
    assert result.mtime_written is True
    assert result.warning is not None
    assert abs(p.stat().st_mtime - new_dt.timestamp()) < 1.0
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_date_writer.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/photo_cleaner/date_writer.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_date_writer.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/photo_cleaner/date_writer.py tests/test_date_writer.py
git commit -m "Add date_writer: EXIF for JPEG, mtime always, warning for non-JPEG"
```

---

## Task 6: `embedder.py` — CLIP wrapper with a test stub

**Files:**
- Create: `src/photo_cleaner/embedder.py`
- Create: `tests/test_embedder.py`

The real model is heavy. Tests use `Embedder.from_callable(fn)` to inject a deterministic stub; a single `@pytest.mark.slow` test exercises the real model.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedder.py
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_embedder.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/photo_cleaner/embedder.py`**

```python
"""Image embedding via open_clip (lazy-loaded), with a test-friendly stub path."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image

from photo_cleaner.config import CLIP_MODEL_NAME, CLIP_MODEL_PRETRAINED, EMBED_DIM


class Embedder:
    """Encode a PIL image as an L2-normalized float32 vector of shape (EMBED_DIM,).

    Construct with no args for the real model (lazy-loaded on first ``embed``).
    Use :py:meth:`from_callable` in tests to inject a deterministic embedding.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        pretrained: str = CLIP_MODEL_PRETRAINED,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._torch = None
        self._embed_callable: Optional[Callable[[Image.Image], np.ndarray]] = None

    @classmethod
    def from_callable(cls, fn: Callable[[Image.Image], np.ndarray]) -> "Embedder":
        e = cls.__new__(cls)
        e._model_name = "<stub>"
        e._pretrained = "<stub>"
        e._model = None
        e._preprocess = None
        e._torch = None
        e._embed_callable = fn
        return e

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._embed_callable is not None:
            return
        import open_clip  # heavy; defer
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._torch = torch

    def embed(self, image: Image.Image) -> np.ndarray:
        if self._embed_callable is not None:
            raw = self._embed_callable(image).astype(np.float32, copy=False)
        else:
            self._ensure_loaded()
            assert self._model is not None and self._preprocess is not None
            torch = self._torch
            with torch.no_grad():
                t = self._preprocess(image).unsqueeze(0)
                feats = self._model.encode_image(t)
            raw = feats.squeeze(0).cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(raw))
        if norm == 0.0:
            raise ValueError("embedding has zero norm")
        v = raw / norm
        if v.shape != (EMBED_DIM,):
            raise ValueError(f"expected ({EMBED_DIM},), got {v.shape}")
        return v
```

- [ ] **Step 4: Run fast tests, verify pass**

```bash
pytest tests/test_embedder.py -v
```

Expected: 2 stub tests pass, slow test deselected.

- [ ] **Step 5: Commit**

```bash
git add src/photo_cleaner/embedder.py tests/test_embedder.py
git commit -m "Add embedder: open_clip wrapper + injectable stub"
```

---

## Task 7: `index.py` — SQLite + in-RAM embeddings matrix

**Files:**
- Create: `src/photo_cleaner/index.py`
- Create: `tests/test_index.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from photo_cleaner.index import Index, PhotoRecord


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype("float32")
    return v / np.linalg.norm(v)


def _record(path: Path, sha: str, emb, edited="2010-06-12") -> PhotoRecord:
    return PhotoRecord(
        path=path,
        original_name="orig.jpg",
        sha256=sha,
        embedding=emb,
        detected_date="2010-06-12",
        edited_date=edited,
        date_action="change",
        processed_at="2026-05-21T10:00:00",
    )


def test_insert_and_load_matrix(tmp_path):
    db = tmp_path / "i.db"
    idx = Index(db)
    e1, e2 = _emb(1), _emb(2)
    idx.insert(_record(tmp_path / "_done_a.jpg", "a" * 64, e1))
    idx.insert(_record(tmp_path / "_done_b.jpg", "b" * 64, e2))
    matrix, shas, paths = idx.snapshot()
    assert matrix.shape == (2, 512)
    assert matrix.dtype == np.float32
    assert set(shas) == {"a" * 64, "b" * 64}
    assert len(paths) == 2


def test_settings_round_trip(tmp_path):
    idx = Index(tmp_path / "i.db")
    assert idx.get_setting("last_edited_date") is None
    idx.set_setting("last_edited_date", "2010-06-12")
    assert idx.get_setting("last_edited_date") == "2010-06-12"
    idx.set_setting("last_edited_date", "2011-01-01")
    assert idx.get_setting("last_edited_date") == "2011-01-01"


def test_insert_duplicate_path_raises(tmp_path):
    idx = Index(tmp_path / "i.db")
    e = _emb(1)
    idx.insert(_record(tmp_path / "_done_a.jpg", "a" * 64, e))
    with pytest.raises(Exception):
        idx.insert(_record(tmp_path / "_done_a.jpg", "a" * 64, e))


def test_find_by_sha256(tmp_path):
    idx = Index(tmp_path / "i.db")
    e = _emb(7)
    rec = _record(tmp_path / "_done_a.jpg", "f" * 64, e)
    idx.insert(rec)
    found = idx.find_by_sha256("f" * 64)
    assert found is not None
    assert found.path == rec.path
    assert idx.find_by_sha256("0" * 64) is None
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_index.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/photo_cleaner/index.py`**

```python
"""SQLite-backed index of processed photos with an in-RAM embeddings matrix.

The connection is opened with ``check_same_thread=False`` because the UI
thread writes (insert / set_setting / find_by_sha256) and the worker thread
reads (snapshot). A ``threading.Lock`` serializes access to the connection
to satisfy Python's sqlite3 module — SQLite itself handles concurrency via
WAL.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from photo_cleaner.config import EMBED_DIM

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS photos (
  id            INTEGER PRIMARY KEY,
  path          TEXT NOT NULL UNIQUE,
  original_name TEXT NOT NULL,
  sha256        TEXT NOT NULL,
  embedding     BLOB NOT NULL,
  detected_date TEXT,
  edited_date   TEXT NOT NULL,
  date_action   TEXT NOT NULL CHECK (date_action IN ('keep','change')),
  processed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_sha256 ON photos(sha256);

CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class PhotoRecord:
    path: Path
    original_name: str
    sha256: str
    embedding: np.ndarray  # (EMBED_DIM,) float32, L2-normalized
    detected_date: str | None
    edited_date: str
    date_action: str  # 'keep' | 'change'
    processed_at: str


class Index:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----- photos -----

    def insert(self, rec: PhotoRecord) -> None:
        if rec.embedding.shape != (EMBED_DIM,) or rec.embedding.dtype != np.float32:
            raise ValueError("embedding must be float32 shape (EMBED_DIM,)")
        params = (
            str(rec.path),
            rec.original_name,
            rec.sha256,
            rec.embedding.tobytes(),
            rec.detected_date,
            rec.edited_date,
            rec.date_action,
            rec.processed_at,
        )
        self._with_retry(
            lambda: self._conn.execute(
                """INSERT INTO photos
                   (path, original_name, sha256, embedding, detected_date,
                    edited_date, date_action, processed_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                params,
            )
        )

    def find_by_sha256(self, sha: str) -> PhotoRecord | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT path, original_name, sha256, embedding, detected_date,
                          edited_date, date_action, processed_at
                   FROM photos WHERE sha256 = ? LIMIT 1""",
                (sha,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def snapshot(self) -> tuple[np.ndarray, list[str], list[Path]]:
        """Return (matrix (N,EMBED_DIM), sha256_list, path_list) sorted by id."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT embedding, sha256, path FROM photos ORDER BY id"
            ).fetchall()
        if not rows:
            return (
                np.zeros((0, EMBED_DIM), dtype=np.float32),
                [],
                [],
            )
        mat = np.stack(
            [np.frombuffer(r[0], dtype=np.float32) for r in rows], axis=0
        )
        shas = [r[1] for r in rows]
        paths = [Path(r[2]) for r in rows]
        return mat, shas, paths

    # ----- settings -----

    def get_setting(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set_setting(self, key: str, value: str) -> None:
        self._with_retry(
            lambda: self._conn.execute(
                """INSERT INTO settings(key, value) VALUES(?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )
        )

    # ----- internals -----

    def _with_retry(self, op) -> None:
        """Run a write op under the lock with exponential backoff on lock errors."""
        delays = (0.05, 0.2, 0.5)
        last_exc: Exception | None = None
        for attempt in range(len(delays) + 1):
            try:
                with self._lock:
                    op()
                    self._conn.commit()
                return
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if attempt == len(delays):
                    break
                time.sleep(delays[attempt])
        raise last_exc  # type: ignore[misc]


def _row_to_record(row) -> PhotoRecord:
    return PhotoRecord(
        path=Path(row[0]),
        original_name=row[1],
        sha256=row[2],
        embedding=np.frombuffer(row[3], dtype=np.float32),
        detected_date=row[4],
        edited_date=row[5],
        date_action=row[6],
        processed_at=row[7],
    )
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_index.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/photo_cleaner/index.py tests/test_index.py
git commit -m "Add index: SQLite store + in-RAM embeddings snapshot"
```

---

## Task 8: `duplicate_finder.py` — pure functions over the matrix

**Files:**
- Create: `src/photo_cleaner/duplicate_finder.py`
- Create: `tests/test_duplicate_finder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_duplicate_finder.py
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_duplicate_finder.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/photo_cleaner/duplicate_finder.py`**

```python
"""Pure functions for duplicate + similar queries over an embeddings matrix."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from photo_cleaner.config import DUP_THRESHOLD


@dataclass(frozen=True)
class Match:
    path: Path
    similarity: float
    reason: str  # 'sha256' | 'cosine'


def find_duplicate(
    query: np.ndarray,
    query_sha: str,
    matrix: np.ndarray,
    shas: list[str],
    paths: list[Path],
) -> Match | None:
    """Exact SHA256 match wins; otherwise the highest cosine ≥ DUP_THRESHOLD."""
    if len(shas) == 0:
        return None
    for i, s in enumerate(shas):
        if s == query_sha:
            sim = float(np.dot(matrix[i], query))
            return Match(path=paths[i], similarity=sim, reason="sha256")
    sims = matrix @ query  # both are L2-normalized -> cosine
    best = int(np.argmax(sims))
    if float(sims[best]) >= DUP_THRESHOLD:
        return Match(path=paths[best], similarity=float(sims[best]), reason="cosine")
    return None


def find_similar(
    query: np.ndarray,
    matrix: np.ndarray,
    paths: list[Path],
    threshold: float,
    k: int,
) -> list[Match]:
    """Return up to ``k`` matches with cosine ≥ ``threshold``, sorted desc."""
    if matrix.shape[0] == 0:
        return []
    sims = matrix @ query
    idxs = np.argsort(-sims)
    out: list[Match] = []
    for i in idxs[: max(k, 0)]:
        s = float(sims[i])
        if s < threshold:
            break
        out.append(Match(path=paths[i], similarity=s, reason="cosine"))
    return out
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_duplicate_finder.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/photo_cleaner/duplicate_finder.py tests/test_duplicate_finder.py
git commit -m "Add duplicate_finder: SHA-first then cosine, top-k similar"
```

---

## Task 9: `thumbnailer.py` — PIL → QPixmap

**Files:**
- Create: `src/photo_cleaner/thumbnailer.py`
- Create: `tests/test_thumbnailer.py`

QPixmap requires a QApplication. We use `pytest-qt`'s `qapp` fixture.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_thumbnailer.py
from __future__ import annotations

from PIL import Image

from photo_cleaner.thumbnailer import to_pil_thumbnail, to_qpixmap


def test_pil_thumbnail_bounds_max_side():
    img = Image.new("RGB", (1600, 900))
    out = to_pil_thumbnail(img, 800)
    w, h = out.size
    assert max(w, h) == 800
    # aspect preserved within 1 pixel rounding
    assert abs(w / h - 1600 / 900) < 0.01


def test_pil_thumbnail_does_not_upscale():
    img = Image.new("RGB", (200, 100))
    out = to_pil_thumbnail(img, 800)
    assert out.size == (200, 100)


def test_to_qpixmap_returns_pixmap_with_correct_size(qapp):
    img = Image.new("RGB", (300, 200), color=(123, 45, 67))
    pix = to_qpixmap(img, 160)
    assert not pix.isNull()
    assert max(pix.width(), pix.height()) == 160
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_thumbnailer.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/photo_cleaner/thumbnailer.py`**

```python
"""PIL → bounded thumbnail → QPixmap."""
from __future__ import annotations

import io

from PIL import Image
from PyQt6.QtGui import QPixmap


def to_pil_thumbnail(img: Image.Image, max_side: int) -> Image.Image:
    """Return a thumbnail of ``img`` with the longer side bounded by
    ``max_side``. Never upscales. Preserves aspect ratio. Returns a copy.
    """
    if max(img.size) <= max_side:
        return img.copy()
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def to_qpixmap(img: Image.Image, max_side: int) -> QPixmap:
    """Thumbnail ``img`` to ``max_side`` and convert to QPixmap via PNG bytes."""
    thumb = to_pil_thumbnail(img, max_side)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    pix = QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    return pix
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_thumbnailer.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/photo_cleaner/thumbnailer.py tests/test_thumbnailer.py
git commit -m "Add thumbnailer: PIL bounded thumbnail + QPixmap conversion"
```

---

## Task 10: `worker.py` — background `QThread` that processes one photo at a time

**Files:**
- Create: `src/photo_cleaner/worker.py`
- Create: `tests/test_worker.py`

Worker pre-fetches one photo at a time on a background thread. UI enqueues a `Path` (and the current similarity threshold); worker emits a `photo_ready(PhotoBundle)` signal when done.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worker.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from photo_cleaner.embedder import Embedder
from photo_cleaner.index import Index, PhotoRecord
from photo_cleaner.worker import PhotoBundle, Worker


def _stub_embedder(seed: int) -> Embedder:
    v = np.arange(512, dtype="float32") + seed
    v = v / np.linalg.norm(v)
    return Embedder.from_callable(lambda img, _v=v: _v.copy())


def _seed_index(idx: Index, tmp_path) -> np.ndarray:
    """Seed one record so the worker has something to compare against."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(512).astype("float32")
    v = v / np.linalg.norm(v)
    idx.insert(
        PhotoRecord(
            path=tmp_path / "_done_old.jpg",
            original_name="old.jpg",
            sha256="0" * 64,
            embedding=v,
            detected_date="2010-01-01",
            edited_date="2010-01-01",
            date_action="keep",
            processed_at="2026-01-01T00:00:00",
        )
    )
    return v


def test_worker_emits_photo_ready(qtbot, tmp_path, make_jpeg):
    p = make_jpeg(tmp_path / "new.jpg", exif_dt=datetime(2010, 6, 12))
    idx = Index(tmp_path / "i.db")
    _seed_index(idx, tmp_path)
    embedder = _stub_embedder(1)

    worker = Worker(embedder=embedder, index=idx)
    worker.start()
    try:
        with qtbot.waitSignal(worker.photo_ready, timeout=5000) as blocker:
            worker.enqueue(p, threshold=0.92)
        bundle = blocker.args[0]
    finally:
        worker.stop()
        worker.wait(5000)

    assert isinstance(bundle, PhotoBundle)
    assert bundle.path == p
    assert bundle.date_info.earliest == datetime(2010, 6, 12)
    assert bundle.embedding.shape == (512,)
    assert bundle.duplicate is None  # different stub embedding from seed
    assert isinstance(bundle.similar, list)
    assert bundle.thumbnail_main is not None
    assert bundle.sha256 and len(bundle.sha256) == 64
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_worker.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/photo_cleaner/worker.py`**

```python
"""Background QThread that processes one photo end-to-end and emits a bundle."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

import pillow_heif

from photo_cleaner.config import EMBED_DIM, SIM_TOP_K, THUMB_MAIN
from photo_cleaner.date_extractor import DateInfo, extract
from photo_cleaner.duplicate_finder import Match, find_duplicate, find_similar
from photo_cleaner.embedder import Embedder
from photo_cleaner.index import Index
from photo_cleaner.thumbnailer import to_qpixmap

pillow_heif.register_heif_opener()


@dataclass(frozen=True)
class PhotoBundle:
    path: Path
    date_info: DateInfo
    sha256: str
    embedding: "np.ndarray"  # noqa: F821 — kept loose to avoid import cycle
    thumbnail_main: object  # QPixmap
    duplicate: Match | None
    similar: list[Match]


class Worker(QThread):
    photo_ready = pyqtSignal(PhotoBundle)
    error = pyqtSignal(Path, str)

    def __init__(self, embedder: Embedder, index: Index, parent=None) -> None:
        super().__init__(parent)
        self._embedder = embedder
        self._index = index
        self._queue: "Queue[tuple[Path, float] | None]" = Queue()

    def enqueue(self, path: Path, threshold: float) -> None:
        self._queue.put((path, threshold))

    def stop(self) -> None:
        self._queue.put(None)

    def run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                continue
            if item is None:
                return
            path, threshold = item
            try:
                bundle = self._process(path, threshold)
            except Exception as exc:  # never let the thread die on a bad file
                self.error.emit(path, str(exc))
                continue
            self.photo_ready.emit(bundle)

    def _process(self, path: Path, threshold: float) -> PhotoBundle:
        date_info = extract(path)
        sha = _sha256_of_file(path)
        with Image.open(path) as im:
            im.load()
            embedding = self._embedder.embed(im.convert("RGB"))
            thumb = to_qpixmap(im.convert("RGB"), THUMB_MAIN)
        matrix, shas, paths = self._index.snapshot()
        dup = find_duplicate(embedding, sha, matrix, shas, paths)
        sim = find_similar(embedding, matrix, paths, threshold=threshold, k=SIM_TOP_K)
        if embedding.shape != (EMBED_DIM,):
            raise RuntimeError(f"bad embedding shape {embedding.shape}")
        return PhotoBundle(
            path=path,
            date_info=date_info,
            sha256=sha,
            embedding=embedding,
            thumbnail_main=thumb,
            duplicate=dup,
            similar=sim,
        )


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_worker.py -v
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add src/photo_cleaner/worker.py tests/test_worker.py
git commit -m "Add worker: QThread processes one photo (date+embed+sha+dup+similar)"
```

---

## Task 11: `recovery.py` — heal a crash between DB insert and rename

**Files:**
- Create: `src/photo_cleaner/recovery.py`
- Create: `tests/test_recovery.py`

Per the spec atomicity section: order is `write → insert → rename`. If we crashed between insert and rename, the next run sees an un-prefixed file whose SHA256 is already in the DB. Recovery walks unprocessed files and, for any whose SHA matches an existing row, applies the missing rename.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recovery.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from photo_cleaner.config import DONE_PREFIX
from photo_cleaner.index import Index, PhotoRecord
from photo_cleaner.recovery import fix_up_pending_renames


def test_fix_up_renames_when_db_has_row_but_file_lacks_prefix(tmp_path, make_jpeg):
    # Pretend a previous run inserted the row, but crashed before renaming.
    p = make_jpeg(tmp_path / "x.jpg")
    sha = "a" * 64
    idx = Index(tmp_path / "i.db")
    # NOTE: the path stored in the DB is the *post-rename* path that the
    # previous run intended to write.
    intended = tmp_path / f"{DONE_PREFIX}x.jpg"
    v = np.ones(512, dtype="float32")
    v /= np.linalg.norm(v)
    idx.insert(
        PhotoRecord(
            path=intended,
            original_name="x.jpg",
            sha256=sha,
            embedding=v,
            detected_date="2010-01-01",
            edited_date="2010-01-01",
            date_action="keep",
            processed_at="2026-01-01T00:00:00",
        )
    )
    # Override the sha lookup by stubbing _sha256_of_file via patching:
    import photo_cleaner.recovery as rec_mod

    rec_mod._sha256_of_file = lambda _p: sha  # type: ignore[attr-defined]

    healed = fix_up_pending_renames(tmp_path, idx)
    assert healed == [intended]
    assert intended.exists()
    assert not p.exists()


def test_fix_up_noop_when_no_pending(tmp_path, make_jpeg):
    make_jpeg(tmp_path / "new.jpg")
    idx = Index(tmp_path / "i.db")
    assert fix_up_pending_renames(tmp_path, idx) == []
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_recovery.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/photo_cleaner/recovery.py`**

```python
"""Heal partial state from a crash between DB insert and filesystem rename."""
from __future__ import annotations

import hashlib
from pathlib import Path

from photo_cleaner.config import DONE_PREFIX
from photo_cleaner.index import Index
from photo_cleaner.scanner import iter_unprocessed


def fix_up_pending_renames(root: Path, index: Index) -> list[Path]:
    """Walk ``root`` for unprocessed files; for each, if a DB row exists with
    a matching SHA256, perform the missing rename. Returns the list of
    intended (post-rename) paths that were healed.
    """
    healed: list[Path] = []
    for p in iter_unprocessed(root):
        sha = _sha256_of_file(p)
        rec = index.find_by_sha256(sha)
        if rec is None:
            continue
        intended = p.parent / f"{DONE_PREFIX}{p.name}"
        if rec.path != intended:
            # The DB has a row for this content but at a different intended
            # path (e.g., file moved between runs). Best to leave it alone
            # and let the user reprocess.
            continue
        p.rename(intended)
        healed.append(intended)
    return healed


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_recovery.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/photo_cleaner/recovery.py tests/test_recovery.py
git commit -m "Add recovery: heal pending renames via SHA256 lookup"
```

---

## Task 12: UI — `date_panel.py`

**Files:**
- Create: `src/photo_cleaner/ui/date_panel.py`

Per spec, UI widgets have no automated tests — they are exercised in the manual smoke checklist. Implement, eyeball-check via the main window in Task 16.

- [ ] **Step 1: Implement `src/photo_cleaner/ui/date_panel.py`**

```python
"""Right-pane top: detected date + editable date + Keep/Change buttons."""
from __future__ import annotations

from datetime import date, datetime

from PyQt6.QtCore import QDate, pyqtSignal
from PyQt6.QtWidgets import (
    QDateEdit,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DatePanel(QWidget):
    keep_clicked = pyqtSignal()
    change_clicked = pyqtSignal(datetime)  # the chosen date as midnight datetime

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._detected_label = QLabel("Detected: —")
        self._source_label = QLabel("source: —")
        self._source_label.setStyleSheet("color: #777; font-size: 11px;")
        self._date_edit = QDateEdit()
        self._date_edit.setDisplayFormat("yyyy-MM-dd")
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setDate(QDate.currentDate())
        self._keep_btn = QPushButton("Keep")
        self._change_btn = QPushButton("Change")
        self._keep_btn.clicked.connect(self.keep_clicked.emit)
        self._change_btn.clicked.connect(self._on_change)

        layout = QVBoxLayout(self)
        layout.addWidget(self._detected_label)
        layout.addWidget(self._source_label)
        layout.addSpacing(8)
        layout.addWidget(QLabel("Set date to:"))
        layout.addWidget(self._date_edit)
        layout.addWidget(QLabel("(default = last entry)"))
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._keep_btn)
        btn_row.addWidget(self._change_btn)
        layout.addLayout(btn_row)
        layout.addStretch(1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

    def set_state(
        self, detected: datetime, source: str, default_for_editor: datetime
    ) -> None:
        self._detected_label.setText(f"Detected: {detected.strftime('%Y-%m-%d')}")
        self._source_label.setText(f"source: {source}")
        d = default_for_editor
        self._date_edit.setDate(QDate(d.year, d.month, d.day))

    def _on_change(self) -> None:
        d: QDate = self._date_edit.date()
        self.change_clicked.emit(datetime(d.year(), d.month(), d.day()))
```

- [ ] **Step 2: Commit**

```bash
git add src/photo_cleaner/ui/date_panel.py
git commit -m "Add UI: date_panel widget (detected + editor + Keep/Change)"
```

---

## Task 13: UI — `similar_grid.py`

**Files:**
- Create: `src/photo_cleaner/ui/similar_grid.py`

- [ ] **Step 1: Implement `src/photo_cleaner/ui/similar_grid.py`**

```python
"""Right-pane bottom: threshold slider + scrollable grid of similar thumbs."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from photo_cleaner.config import (
    SIM_THRESHOLD_DEFAULT,
    SIM_THRESHOLD_MAX,
    SIM_THRESHOLD_MIN,
    THUMB_GRID,
)
from photo_cleaner.duplicate_finder import Match


class SimilarGrid(QWidget):
    threshold_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(SIM_THRESHOLD_MIN * 100))
        self._slider.setMaximum(int(SIM_THRESHOLD_MAX * 100))
        self._slider.setValue(int(SIM_THRESHOLD_DEFAULT * 100))
        self._slider.valueChanged.connect(self._on_slider)
        self._readout = QLabel(f"SIMILAR ABOVE: {SIM_THRESHOLD_DEFAULT:.2f}")
        self._count_label = QLabel("(0 matches)")

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setSpacing(4)
        scroll = QScrollArea()
        scroll.setWidget(self._grid_host)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self._readout)
        layout.addWidget(self._slider)
        layout.addWidget(self._count_label)
        layout.addWidget(scroll, 1)

    @property
    def threshold(self) -> float:
        return self._slider.value() / 100.0

    def set_matches(self, matches: list[Match], pixmaps: dict[Path, QPixmap]) -> None:
        # Clear existing grid
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        cols = 4
        for i, m in enumerate(matches):
            pix = pixmaps.get(m.path)
            lbl = QLabel()
            if pix is not None and not pix.isNull():
                lbl.setPixmap(pix.scaled(THUMB_GRID, THUMB_GRID,
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
            else:
                lbl.setText(m.path.name)
            lbl.setFixedSize(THUMB_GRID, THUMB_GRID)
            lbl.setToolTip(f"{m.path.name}\ncos={m.similarity:.4f}")
            self._grid.addWidget(lbl, i // cols, i % cols)
        self._count_label.setText(f"({len(matches)} matches ≥ {self.threshold:.2f})")

    def _on_slider(self, value: int) -> None:
        thr = value / 100.0
        self._readout.setText(f"SIMILAR ABOVE: {thr:.2f}")
        self.threshold_changed.emit(thr)
```

- [ ] **Step 2: Commit**

```bash
git add src/photo_cleaner/ui/similar_grid.py
git commit -m "Add UI: similar_grid widget (threshold slider + thumb grid)"
```

---

## Task 14: UI — `duplicate_strip.py`

**Files:**
- Create: `src/photo_cleaner/ui/duplicate_strip.py`

- [ ] **Step 1: Implement `src/photo_cleaner/ui/duplicate_strip.py`**

```python
"""Bottom strip shown when the current photo is a duplicate of an indexed one."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from photo_cleaner.config import THUMB_GRID


class DuplicateStrip(QWidget):
    keep_both_clicked = pyqtSignal()
    delete_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background-color: #fff3cd; padding: 8px;")
        self._title = QLabel("DUPLICATE FOUND")
        self._title.setStyleSheet("font-weight: bold;")
        self._new_thumb = QLabel()
        self._match_thumb = QLabel()
        for lbl in (self._new_thumb, self._match_thumb):
            lbl.setFixedSize(THUMB_GRID, THUMB_GRID)
        self._new_label = QLabel("new:")
        self._match_label = QLabel("match:")
        self._keep_both = QPushButton("Keep both")
        self._delete = QPushButton("Delete")
        self._delete.setStyleSheet("color: #b00;")
        self._keep_both.clicked.connect(self.keep_both_clicked.emit)
        self._delete.clicked.connect(self.delete_clicked.emit)

        row = QHBoxLayout()
        col_title = QVBoxLayout()
        col_title.addWidget(self._title)
        col_title.addStretch(1)
        row.addLayout(col_title)
        row.addWidget(self._new_label)
        row.addWidget(self._new_thumb)
        row.addWidget(self._match_label)
        row.addWidget(self._match_thumb)
        row.addStretch(1)
        row.addWidget(self._keep_both)
        row.addWidget(self._delete)
        self.setLayout(row)
        self.setVisible(False)

    def show_pair(
        self, similarity: float, new_pix: QPixmap, match_pix: QPixmap, match_name: str
    ) -> None:
        self._title.setText(f"DUPLICATE FOUND (cos={similarity:.3f})")
        for lbl, pix in ((self._new_thumb, new_pix), (self._match_thumb, match_pix)):
            if pix is None or pix.isNull():
                lbl.setText("—")
            else:
                lbl.setPixmap(
                    pix.scaled(
                        THUMB_GRID, THUMB_GRID,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        self._match_label.setText(f"match: {match_name}")
        self.setVisible(True)

    def hide_strip(self) -> None:
        self.setVisible(False)
```

- [ ] **Step 2: Commit**

```bash
git add src/photo_cleaner/ui/duplicate_strip.py
git commit -m "Add UI: duplicate_strip widget (side-by-side + Keep both/Delete)"
```

---

## Task 15: UI — `toast.py`

**Files:**
- Create: `src/photo_cleaner/ui/toast.py`

- [ ] **Step 1: Implement `src/photo_cleaner/ui/toast.py`**

```python
"""Tiny non-blocking notification at bottom-right."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QLabel, QWidget


class Toast(QLabel):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "background: rgba(40,40,40,0.92); color: white;"
            "padding: 8px 12px; border-radius: 6px;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setVisible(False)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, ms: int = 3500) -> None:
        self.setText(text)
        self.adjustSize()
        parent = self.parentWidget()
        if parent is not None:
            geo = parent.geometry()
            self.move(geo.width() - self.width() - 16, geo.height() - self.height() - 16)
        self.setVisible(True)
        self.raise_()
        self._timer.start(ms)
```

- [ ] **Step 2: Commit**

```bash
git add src/photo_cleaner/ui/toast.py
git commit -m "Add UI: toast widget"
```

---

## Task 16: UI — `main_window.py`

**Files:**
- Create: `src/photo_cleaner/ui/main_window.py`

This is the glue. It owns the worker, the index, the scanner state; reacts to button clicks by calling `date_writer`, doing the rename, inserting into the index, asking the worker for the next photo.

- [ ] **Step 1: Implement `src/photo_cleaner/ui/main_window.py`**

```python
"""Main window: assembles widgets, drives the processing loop."""
from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import send2trash
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from photo_cleaner.config import (
    DONE_PREFIX,
    DUP_THRESHOLD,
    PICTURES_ROOT,
    SIM_THRESHOLD_DEFAULT,
    SIM_TOP_K,
    THUMB_GRID,
    THUMB_MAIN,
)
from photo_cleaner.date_writer import write as write_date
from photo_cleaner.duplicate_finder import find_similar
from photo_cleaner.embedder import Embedder
from photo_cleaner.index import Index, PhotoRecord
from photo_cleaner.scanner import iter_unprocessed
from photo_cleaner.thumbnailer import to_qpixmap
from photo_cleaner.ui.date_panel import DatePanel
from photo_cleaner.ui.duplicate_strip import DuplicateStrip
from photo_cleaner.ui.similar_grid import SimilarGrid
from photo_cleaner.ui.toast import Toast
from photo_cleaner.worker import PhotoBundle, Worker


class MainWindow(QMainWindow):
    def __init__(
        self,
        index: Index,
        embedder: Embedder,
        root: Path = PICTURES_ROOT,
    ) -> None:
        super().__init__()
        self.setWindowTitle("photo-cleaner")
        self.resize(1200, 800)

        self._index = index
        self._root = root
        self._scanner: Iterator[Path] = iter_unprocessed(root)
        self._total = sum(1 for _ in iter_unprocessed(root))
        self._processed_count = 0
        self._current: PhotoBundle | None = None
        self._ready_queue: "deque[PhotoBundle]" = deque()
        self._pixmap_cache: dict[Path, QPixmap] = {}

        # Widgets
        self._main_thumb = QLabel()
        self._main_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main_thumb.setMinimumSize(700, 500)
        self._filename_label = QLabel("")
        self._date_panel = DatePanel()
        self._similar_grid = SimilarGrid()
        self._dup_strip = DuplicateStrip()
        self._toast = Toast(self)

        # Layout
        left = QVBoxLayout()
        left.addWidget(self._main_thumb, 1)
        left.addWidget(self._filename_label)
        right = QVBoxLayout()
        right.addWidget(self._date_panel)
        right.addWidget(self._similar_grid, 1)
        top = QHBoxLayout()
        top.addLayout(left, 2)
        top.addLayout(right, 1)
        root_layout = QVBoxLayout()
        root_layout.addLayout(top, 1)
        root_layout.addWidget(self._dup_strip)
        host = QWidget()
        host.setLayout(root_layout)
        self.setCentralWidget(host)
        self.setStatusBar(QStatusBar())

        # Worker
        self._worker = Worker(embedder=embedder, index=index)
        self._worker.photo_ready.connect(self._on_photo_ready)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

        # Signals
        self._date_panel.keep_clicked.connect(self._on_keep)
        self._date_panel.change_clicked.connect(self._on_change)
        self._dup_strip.keep_both_clicked.connect(self._on_keep)  # same as Keep
        self._dup_strip.delete_clicked.connect(self._on_delete_duplicate)
        self._similar_grid.threshold_changed.connect(self._on_threshold_changed)

        # Prime the pump: enqueue first two photos.
        self._enqueue_next()
        self._enqueue_next()

    # ----- pipeline -----

    def _enqueue_next(self) -> None:
        try:
            p = next(self._scanner)
        except StopIteration:
            return
        self._worker.enqueue(p, threshold=self._similar_grid.threshold)

    def _on_photo_ready(self, bundle: PhotoBundle) -> None:
        if self._current is None:
            self._present(bundle)
        else:
            self._ready_queue.append(bundle)

    def _on_worker_error(self, path: Path, msg: str) -> None:
        self._toast.show_message(f"Could not process {path.name}: {msg}")
        self._enqueue_next()

    def _present(self, bundle: PhotoBundle) -> None:
        self._current = bundle
        # Main thumb
        self._main_thumb.setPixmap(
            bundle.thumbnail_main.scaled(
                self._main_thumb.width(),
                self._main_thumb.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        rel = bundle.path.relative_to(self._root) if bundle.path.is_relative_to(self._root) else bundle.path
        self._filename_label.setText(str(rel))

        # Date panel: pre-fill with last_edited_date if present, else detected
        last = self._index.get_setting("last_edited_date")
        default = (
            datetime.fromisoformat(last)
            if last
            else bundle.date_info.earliest
        )
        self._date_panel.set_state(
            detected=bundle.date_info.earliest,
            source=bundle.date_info.earliest_source,
            default_for_editor=default,
        )

        # Similar grid
        self._refresh_similar(bundle, threshold=self._similar_grid.threshold)

        # Duplicate strip
        if bundle.duplicate is not None:
            match_pix = self._get_thumb_for_indexed(bundle.duplicate.path)
            self._dup_strip.show_pair(
                similarity=bundle.duplicate.similarity,
                new_pix=bundle.thumbnail_main,
                match_pix=match_pix,
                match_name=bundle.duplicate.path.name,
            )
        else:
            self._dup_strip.hide_strip()

        self.statusBar().showMessage(
            f"{bundle.path.parent.name} · photo {self._processed_count + 1}/{self._total}"
        )

    def _refresh_similar(self, bundle: PhotoBundle, threshold: float) -> None:
        matrix, _shas, paths = self._index.snapshot()
        matches = find_similar(
            bundle.embedding, matrix, paths, threshold=threshold, k=SIM_TOP_K
        )
        pixmaps = {m.path: self._get_thumb_for_indexed(m.path) for m in matches}
        self._similar_grid.set_matches(matches, pixmaps)

    def _get_thumb_for_indexed(self, path: Path) -> QPixmap:
        if path in self._pixmap_cache:
            return self._pixmap_cache[path]
        try:
            with Image.open(path) as im:
                pix = to_qpixmap(im.convert("RGB"), THUMB_GRID)
        except Exception:
            pix = QPixmap()
        self._pixmap_cache[path] = pix
        return pix

    # ----- button handlers -----

    def _on_keep(self) -> None:
        cur = self._current
        if cur is None:
            return
        edited = cur.date_info.earliest
        self._finalize(cur, edited=edited, action="keep", wrote_disk=False)

    def _on_change(self, dt: datetime) -> None:
        cur = self._current
        if cur is None:
            return
        result = write_date(cur.path, dt)
        if result.warning:
            self._toast.show_message(result.warning)
        self._finalize(cur, edited=dt, action="change", wrote_disk=True)

    def _on_delete_duplicate(self) -> None:
        cur = self._current
        if cur is None or cur.duplicate is None:
            return
        try:
            send2trash.send2trash(str(cur.path))
        except Exception as exc:
            self._toast.show_message(f"Trash failed: {exc}")
            return
        self._advance()

    def _on_threshold_changed(self, thr: float) -> None:
        cur = self._current
        if cur is not None:
            self._refresh_similar(cur, threshold=thr)

    # ----- finalize -----

    def _finalize(
        self, cur: PhotoBundle, edited: datetime, action: str, wrote_disk: bool
    ) -> None:
        intended = cur.path.parent / f"{DONE_PREFIX}{cur.path.name}"
        # 1) insert DB row (path stored is the intended post-rename path)
        rec = PhotoRecord(
            path=intended,
            original_name=cur.path.name,
            sha256=cur.sha256,
            embedding=cur.embedding.astype(np.float32),
            detected_date=cur.date_info.earliest.isoformat(),
            edited_date=edited.isoformat(),
            date_action=action,
            processed_at=datetime.now().isoformat(timespec="seconds"),
        )
        try:
            self._index.insert(rec)
        except Exception as exc:
            self._toast.show_message(f"DB insert failed: {exc}")
            return
        # 2) rename file
        try:
            cur.path.rename(intended)
        except Exception as exc:
            self._toast.show_message(
                f"Rename failed (DB row kept, recovery will fix): {exc}"
            )
        # 3) update last_edited_date
        self._index.set_setting("last_edited_date", edited.isoformat())
        self._advance()

    def _advance(self) -> None:
        self._processed_count += 1
        self._current = None
        # Show next from queue if ready; else wait for worker signal.
        if self._ready_queue:
            self._present(self._ready_queue.popleft())
        # Pre-fetch the photo after that.
        self._enqueue_next()

    # ----- lifecycle -----

    def closeEvent(self, event) -> None:
        self._worker.stop()
        self._worker.wait(5000)
        self._index.close()
        super().closeEvent(event)
```

- [ ] **Step 2: Commit**

```bash
git add src/photo_cleaner/ui/main_window.py
git commit -m "Add UI: main_window — glue + processing loop"
```

---

## Task 17: `__main__.py` — entry point with recovery on startup

**Files:**
- Create: `src/photo_cleaner/__main__.py`

- [ ] **Step 1: Implement `src/photo_cleaner/__main__.py`**

```python
"""Entry point: ``python -m photo_cleaner``."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from photo_cleaner.config import DB_PATH, PICTURES_ROOT, ensure_app_data
from photo_cleaner.embedder import Embedder
from photo_cleaner.index import Index
from photo_cleaner.recovery import fix_up_pending_renames
from photo_cleaner.ui.main_window import MainWindow


def main() -> int:
    ensure_app_data()
    app = QApplication(sys.argv)
    if not PICTURES_ROOT.exists():
        QMessageBox.critical(
            None,
            "photo-cleaner",
            f"Pictures root not found:\n{PICTURES_ROOT}",
        )
        return 1
    try:
        embedder = Embedder()
    except Exception as exc:
        QMessageBox.critical(
            None,
            "photo-cleaner",
            f"Failed to load embedding model: {exc}\n\n"
            "Try: pip install open_clip_torch torch",
        )
        return 2
    index = Index(DB_PATH)
    healed = fix_up_pending_renames(PICTURES_ROOT, index)
    win = MainWindow(index=index, embedder=embedder)
    if healed:
        win._toast.show_message(f"Healed {len(healed)} pending rename(s) from a prior crash.")
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the full test suite, verify everything still passes**

```bash
pytest -q
```

Expected: all fast tests pass; slow test deselected.

- [ ] **Step 3: Commit**

```bash
git add src/photo_cleaner/__main__.py
git commit -m "Add entry point: ensure app data, recover, launch UI"
```

---

## Task 18: README — manual smoke checklist + final verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` with the full content**

```markdown
# photo-cleaner

Interactive desktop tool to clean up `~/Google Drive/My Drive/Pictures`:
fix creation dates, surface near-duplicates and visually-similar photos, and
mark processed files with a `_done_` prefix so reruns resume cleanly.

See `docs/superpowers/specs/2026-05-21-photo-cleaner-design.md` for the full design.

## Install

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

The first run downloads the CLIP model (~350 MB) into `~/.cache/clip/` — takes a minute on a normal connection.

## Run

    python -m photo_cleaner

## Test

    pytest -q                # fast tests
    pytest -m slow           # the real-model embedder test

## Where state lives

- DB + error log: `~/.photo-cleaner/`
- Processed files: renamed in place with a `_done_` prefix.
- Last-confirmed date: `settings` table inside the DB.

## Manual smoke checklist

Run on a small test folder first (copy a few photos into a temporary
`~/Google Drive/My Drive/Pictures/_smoketest/` directory or temporarily point
`PICTURES_ROOT` at one).

- [ ] App launches, main window appears within ~10 s (model load on first run).
- [ ] Left pane shows a thumbnail of the first photo.
- [ ] Right pane "Detected" date is sensible; "source" shows the chosen source.
- [ ] Editor pre-fills (with detected date on a fresh DB).
- [ ] **Keep** advances; next photo loads instantly (worker pre-fetched it).
- [ ] **Change** with a new date: re-open the photo in Preview → Inspector → EXIF; DateTimeOriginal matches.
- [ ] Finder "Date Modified" matches the entered date after Change.
- [ ] Default date for the *next* photo is now the date just entered.
- [ ] Similarity slider moves; grid updates in real time.
- [ ] Drop a copy of a processed photo back into the watched folder, restart app: duplicate strip appears with both thumbs.
- [ ] **Delete** on duplicate moves the file to Trash (visible in Finder Trash).
- [ ] **Keep both** treats the duplicate as a normal Keep.
- [ ] Quit mid-session; relaunch — already-processed files (`_done_` prefix) are skipped; resumes at the next file.
- [ ] Drop a corrupt `.jpg` (e.g., `echo nope > x.jpg`) in the folder; app shows a toast and skips it; file is NOT renamed.
```

- [ ] **Step 2: Run the full suite one more time**

```bash
pytest -q
```

Expected: all fast tests green.

- [ ] **Step 3: Smoke-launch the app**

```bash
python -m photo_cleaner
```

Expected: window opens; if the Pictures root has files, the first photo appears within ~10 s on a fresh install.

Stop after visually confirming. (UI behavior is otherwise covered by the manual checklist above — work through it once before considering the project done.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Add README with install, run, test, manual smoke checklist"
```

---

## Done criteria

- All `pytest -q` tests green.
- App launches against `~/Google Drive/My Drive/Pictures` and steps through real photos without crashing.
- Every item in the manual smoke checklist has been walked through once.
