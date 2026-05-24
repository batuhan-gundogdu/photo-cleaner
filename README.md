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
