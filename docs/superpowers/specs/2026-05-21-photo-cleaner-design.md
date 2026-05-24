# photo-cleaner — Design

**Date:** 2026-05-21
**Status:** Approved (design)
**Owner:** batuhangundogdu

## Purpose

Interactive desktop tool to clean up a long-neglected `~/Google Drive/My Drive/Pictures` library: correct creation dates, surface near-duplicates and visually-similar photos against everything previously processed, and mark each file as processed so reruns continue where the last session stopped.

## User flow

1. App launches, loads the embedding model, opens the local index DB, and scans `~/Google Drive/My Drive/Pictures` recursively for supported files whose name does NOT start with the processed prefix `_done_`.
2. For each unprocessed photo (in deterministic `(folder, filename)` order):
   - A thumbnail is shown in the left pane.
   - The right pane shows the detected creation date (the earliest plausible date across EXIF, file times, and filename parsing) and an editable date field pre-filled with the most recent date the user has confirmed (persisted across sessions; see "Default date logic").
   - The right pane shows a similarity threshold slider (default 0.92) and a grid of thumbnails of previously processed photos with cosine similarity ≥ threshold.
   - If a previously processed photo has cosine similarity ≥ 0.999 OR matching SHA256, a duplicate confirmation strip appears at the bottom showing the new photo and the matched photo side by side.
3. The user clicks one of:
   - **Keep** — accept the detected date unchanged.
   - **Change** — write the date from the editor field.
   - **Delete** (only on duplicates) — send the new photo to Trash and advance without indexing it.
   - **Keep both** (only on duplicates) — treat as a normal Keep.
4. The app writes any date changes, renames the file with the `_done_` prefix, inserts a row into the index DB, updates the session's "last entered date", and advances. While the user is reviewing photo N, the background worker is already thumbnailing + embedding + querying photo N+1.
5. App can be quit any time; next launch resumes by skipping `_done_`-prefixed files.

## Scope

**In scope:** `.jpg`, `.jpeg`, `.png`, `.heic` files under `~/Google Drive/My Drive/Pictures/**`. Recursive walk. macOS only.

**Out of scope:** RAW formats, videos, files outside `Pictures`, batch/non-interactive mode, multi-user concurrency, cloud-side dedup against Google Photos.

## Tech stack

- Python 3.11+
- PyQt6 — UI
- Pillow + pillow-heif — image decode + thumbnailing
- piexif — JPEG EXIF read/write; pyexiv2 fallback for HEIC EXIF writes if needed
- open_clip_torch (MobileCLIP-S0 if available on PyPI; otherwise `ViT-B-32` with the smallest pretrained weights) — embeddings
- torch (CPU) — model runtime
- numpy — similarity math (single matmul over a small in-RAM matrix)
- send2trash — safe move-to-Trash for duplicate deletion
- pytest — tests

## UI layout

Single `QMainWindow`, ~1200×800, two columns:

- **Left pane:** large bounded thumbnail (~700×500 max) of the current photo, with the filename and the relative folder path beneath it.
- **Right pane top — date panel:**
  - "Detected: YYYY-MM-DD" + small subtext showing which source produced it (e.g., "source: filename IMG_20100612_134522").
  - Editable date field labelled "Set date to:", pre-filled with `settings.last_edited_date` if present, else the detected date.
  - Two buttons: **Keep** and **Change**.
- **Right pane middle — similarity controls:**
  - Threshold slider, range 0.80–0.99, step 0.01, default 0.92, with numeric readout.
- **Right pane bottom — similar grid:**
  - Up to ~12 thumbnails (160×160) of previously processed photos with cosine similarity ≥ threshold, sorted descending. Scrollable if more.
- **Bottom strip (only when a duplicate is detected):**
  - "DUPLICATE FOUND (cos=1.000)" label + new thumbnail + matched thumbnail side by side + **Keep both** / **Delete** buttons.
- **Title bar / status:** app name · current folder · "photo N/M" · ☰ menu · ⏸ pause button.

A toast widget at the bottom-right surfaces non-blocking warnings (EXIF write fell back to mtime, file could not be opened, etc.).

## Architecture

Single Python process. PyQt6 event loop on the main thread; one `QThread` worker for heavy per-photo work (decode, EXIF read, embed, SHA256, index query). UI never blocks on I/O or compute. Worker pre-fetches one photo ahead of the UI; only one prefetch in flight to bound memory and avoid wasted work after Delete.

```
+---------------------+      enqueue path       +-----------------+
|     MainWindow      | ----------------------> |  Worker (QThread)|
|  (PyQt event loop)  |                         |                 |
|                     | <-- photo_ready signal -|  date_extractor |
|  - date_panel       |                         |  embedder       |
|  - similar_grid     |                         |  thumbnailer    |
|  - duplicate_strip  |                         |  duplicate_finder|
+---------------------+                         +-----------------+
        |                                              |
        | writes                                       | reads
        v                                              v
   date_writer, os.rename, send2trash      +--------------------+
        |                                  |   index (SQLite)   |
        +--------------------------------> |  + in-RAM embeddings|
                                           +--------------------+
```

## Modules

`src/photo_cleaner/`:

- `config.py` — paths (`PICTURES_ROOT = ~/Google Drive/My Drive/Pictures`, `APP_DATA = ~/.photo-cleaner`), constants (`DONE_PREFIX = "_done_"`, `SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".heic"}`, `DUP_THRESHOLD = 0.999`, `SIM_THRESHOLD_DEFAULT = 0.92`, `THUMB_MAIN = 800`, `THUMB_GRID = 160`).
- `scanner.py` — `iter_unprocessed(root: Path) -> Iterator[Path]`: recursive walk, supported extension, name does not start with `DONE_PREFIX`, sorted by `(folder, filename)`.
- `date_extractor.py` — `extract(path: Path) -> DateInfo` where `DateInfo = dataclass(earliest: datetime, sources: dict[str, datetime])`. Sources: EXIF DateTimeOriginal, EXIF DateTimeDigitized, file mtime, file ctime, filename parse (`IMG_YYYYMMDD`, `YYYYMMDD_HHMMSS`, `YYYY-MM-DD`).
- `date_writer.py` — `write(path: Path, new_date: datetime) -> WriteResult` where `WriteResult = dataclass(exif_written: bool, mtime_written: bool, warning: str | None)`. Writes EXIF DateTimeOriginal+Digitized via piexif (JPEG); attempts HEIC via pyexiv2; always sets mtime via `os.utime`. Atomic temp-file swap for EXIF write.
- `embedder.py` — `Embedder` class, lazy-loads the model on first call. `embed(image: PIL.Image) -> np.ndarray (512,) float32 L2-normalized`.
- `index.py` — `Index` class wrapping SQLite + an in-RAM `(N, 512) float32` matrix. Methods: `insert(record)`, `find(query_embedding, query_sha256, threshold) -> (duplicate_or_None, similar)`, `get_setting(key)`, `set_setting(key, value)`. Reloads the matrix when a row is inserted.
- `duplicate_finder.py` — pure functions on the matrix: `find_duplicate(emb, sha, matrix, shas) -> Match | None` (SHA exact-match first, then cosine ≥ `DUP_THRESHOLD`), `find_similar(emb, matrix, threshold, k=12) -> list[Match]`.
- `thumbnailer.py` — `to_qpixmap(image: PIL.Image, size: int) -> QPixmap`.
- `worker.py` — `Worker(QThread)`. Receives path via slot, processes, emits `photo_ready(PhotoBundle)` signal. `PhotoBundle` carries: path, date_info, sha256, embedding, main_thumb, dup_match, similar.
- `ui/main_window.py` — assembles widgets, owns the worker, handles button slots: Keep / Change / Delete / Keep-both → calls into `date_writer`, `index`, `os.rename`, `send2trash`, advances to next path.
- `ui/date_panel.py`, `ui/similar_grid.py`, `ui/duplicate_strip.py`, `ui/toast.py` — leaf widgets.
- `__main__.py` — `python -m photo_cleaner` entry point.

## Data model

SQLite database at `~/.photo-cleaner/index.db`, WAL mode:

```sql
PRAGMA user_version = 1;

CREATE TABLE photos (
  id            INTEGER PRIMARY KEY,
  path          TEXT NOT NULL UNIQUE,        -- post-rename path (with _done_ prefix)
  original_name TEXT NOT NULL,                -- name before rename, for audit
  sha256        TEXT NOT NULL,
  embedding     BLOB NOT NULL,                -- 512 float32 = 2048 bytes
  detected_date TEXT,                         -- ISO 8601, what we computed
  edited_date   TEXT NOT NULL,                -- ISO 8601, what got written
  date_action   TEXT NOT NULL CHECK (date_action IN ('keep', 'change')),
  processed_at  TEXT NOT NULL
);
CREATE INDEX idx_photos_sha256 ON photos(sha256);

CREATE TABLE settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- known keys: 'last_edited_date' (ISO 8601 date)
```

Only the UI thread writes to the DB. The worker reads — it holds a reference to the in-RAM embeddings matrix and the parallel SHA256 list, which the UI thread atomically swaps after each insert.

## Processing flow (per photo N, with prefetch of N+1)

1. UI asks scanner for the next path → enqueues it to the worker.
2. Worker:
   1. Opens with PIL (`pillow-heif` registers HEIC).
   2. Makes the 800px main thumbnail.
   3. Runs `date_extractor.extract`.
   4. Computes SHA256 of the file bytes.
   5. Runs the thumbnail through `embedder.embed`.
   6. Calls `duplicate_finder.find_duplicate` and `find_similar(threshold=current_slider)`.
   7. Emits `photo_ready(PhotoBundle)`.
3. UI receives the signal, swaps in the new thumbnail, date panel, similar grid, and (if applicable) the duplicate strip. Pre-fills the date editor with `settings.last_edited_date` (else `date_info.earliest`).
4. User clicks (order of operations matches the Atomicity section below: write → insert → rename):
   - **Keep** → `edited_date = detected_date`; no EXIF/mtime write; insert row with `date_action='keep'`; rename to `_done_` prefix; update `settings.last_edited_date = detected_date`.
   - **Change** → take date from editor; `date_writer.write(path, new_date)`; insert row with `date_action='change'`; rename; update `settings.last_edited_date = new_date`.
   - **Delete** (duplicate only) → `send2trash(path)`; do NOT rename or insert; advance.
   - **Keep both** (duplicate only) → identical to Keep.
5. UI calls `scanner.next()` and enqueues N+2 to the worker (which may already be working on it from speculative prefetch).

Threshold slider drag: UI re-queries `duplicate_finder.find_similar(current_emb, threshold)` on every slider value change — it's one matmul on the in-RAM matrix, sub-millisecond at the scale of personal photo libraries.

## Default date logic

When the date editor is populated for photo N:
- If `settings.last_edited_date` exists, use it. This value persists across app restarts, so the very first photo of a new session inherits the date the user confirmed at the end of the previous session — useful because photos typically come in chronological clusters and the user often resumes inside the same era.
- Else (no prior session has ever confirmed a date), use `date_info.earliest`.

Both Keep and Change update `settings.last_edited_date` (Keep sets it to the detected date; Change sets it to the entered date). Delete does not.

## Error handling

- **PIL cannot open the file** (corrupt / unsupported): log to `~/.photo-cleaner/errors.log`, toast `"Could not open <name>: <reason>"`, present Skip / Open in Finder buttons. File is NOT renamed → next run will retry.
- **EXIF write fails** (HEIC without writable EXIF, piexif rejection): fall back to mtime-only write, toast `"EXIF write failed for <name>, only mtime updated"`. Still rename + insert row.
- **Rename fails** (permissions, file vanished): if EXIF/mtime were changed, attempt rollback (re-write old values from `DateInfo`); toast error; do NOT insert row; advance.
- **send2trash fails**: toast error; do NOT advance; leave the file for the user to handle.
- **DB lock / IO error**: retry with exponential backoff (3 attempts); if still failing, blocking error dialog — index integrity is non-negotiable.
- **Embedder load fails at startup**: blocking dialog with the install command for the missing dependency; app exits cleanly.

**Atomicity of "process this photo"** — sequence: EXIF write → mtime → DB insert → rename. Crash recovery:
- Crash after EXIF/mtime but before insert: photo's date is updated on disk but no DB row; next run reprocesses; user sees the already-correct detected date and clicks Keep. Acceptable.
- Crash after insert but before rename: next run sees an unprefixed file that's already in the DB (lookup by SHA256). Scanner skips it after a fix-up step that performs the missing rename.
- Crash before any writes: clean — next run reprocesses normally.

## Testing

- `tests/test_date_extractor.py` — synthesize EXIF with piexif into temp JPEGs; test all source kinds and the "earliest of" reduction; test the three filename patterns.
- `tests/test_date_writer.py` — round-trip: write date, read back via independent `exifread`, verify both EXIF tags + mtime.
- `tests/test_scanner.py` — temp dir with mixed extensions, prefixed/unprefixed, nested folders; assert sort order and skipping.
- `tests/test_embedder.py` — same input → identical output; different inputs → cosine < 1; shape == (512,); unit norm.
- `tests/test_index.py` — insert N rows, query similarity, threshold filtering, SHA256 dedup.
- `tests/test_duplicate_finder.py` — synthetic embeddings; assert cosine=1 routes to duplicate, near-1 to similar.
- `tests/test_recovery.py` — simulate crash between insert and rename; verify next-run fix-up.
- **UI tests skipped** intentionally — PyQt6 tests are low ROI for a personal tool. README contains a manual smoke checklist.
- Fixtures: 4–6 small JPEGs + 1 HEIC + 1 PNG in `tests/fixtures/`, total ~50KB.
- Target: `pytest -q` runs in < 10s, no network required (model embedding tests use a tiny stubbed embedder, real-model tests are gated behind `@pytest.mark.slow`).

## Open questions

None at design time. Implementation may surface a question about HEIC EXIF writability — the fallback (mtime-only + toast) is already specified, so this won't block.

## Out of scope (explicit non-goals)

- Editing photo content / cropping / color.
- Batch CLI mode.
- Cross-library dedup or cloud-side actions on Google Photos.
- RAW or video processing.
- Multi-user / concurrent runs against the same DB.
- Auto-categorization beyond duplicate/similar surfacing.
