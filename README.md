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
