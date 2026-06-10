"""Main window: assembles widgets, drives the processing loop."""
from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import send2trash
from PIL import Image, ImageOps
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from photo_cleaner.config import (
    DONE_PREFIX,
    PICTURES_ROOT,
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
        self._current_snapshot: tuple | None = None

        # Widgets
        self._main_thumb = QLabel()
        self._main_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main_thumb.setMinimumSize(700, 500)
        self._progress_label = QLabel("")
        self._filename_label = QLabel("")
        self._date_panel = DatePanel()
        self._similar_grid = SimilarGrid()
        self._toast = Toast(self)
        self._save_exit_btn = QPushButton("Save && Exit")
        self._save_exit_btn.setFixedWidth(120)

        # Layout
        left = QVBoxLayout()
        left.addWidget(self._progress_label)
        left.addWidget(self._main_thumb, 1)
        left.addWidget(self._filename_label)
        right = QVBoxLayout()
        right.addWidget(self._date_panel)
        right.addWidget(self._similar_grid, 1)
        top = QHBoxLayout()
        top.addLayout(left, 2)
        top.addLayout(right, 1)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self._save_exit_btn)
        root_layout = QVBoxLayout()
        root_layout.addLayout(top, 1)
        root_layout.addLayout(footer)
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
        self._date_panel.skip_clicked.connect(self._advance)
        self._date_panel.delete_clicked.connect(self._on_delete_current_photo)
        self._similar_grid.threshold_changed.connect(self._on_threshold_changed)
        self._similar_grid.delete_similar_clicked.connect(self._on_delete_similar)
        self._similar_grid.delete_current_for_similar.connect(self._on_delete_current_photo)
        self._similar_grid.copy_date_clicked.connect(self._date_panel.set_date)
        self._save_exit_btn.clicked.connect(self.close)

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
        # Main thumb — decode PNG bytes on the GUI thread
        pix = QPixmap()
        pix.loadFromData(bundle.thumbnail_main_png, "PNG")
        self._main_thumb.setPixmap(
            pix.scaled(
                self._main_thumb.width(),
                self._main_thumb.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        rel = bundle.path.relative_to(self._root) if bundle.path.is_relative_to(self._root) else bundle.path.name
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

        # Similar grid — snapshot the index once per photo (slider drag re-uses it)
        self._current_snapshot = self._index.snapshot()
        self._refresh_similar(bundle, threshold=self._similar_grid.threshold)

        folder = bundle.path.parent.name
        n = self._processed_count + 1
        self._progress_label.setText(f"{folder}  ·  {n}/{self._total}")
        self.statusBar().showMessage(f"{folder}  ·  {n}/{self._total}")

    def _refresh_similar(self, bundle: PhotoBundle, threshold: float) -> None:
        if self._current_snapshot is None:
            return
        matrix, _shas, paths = self._current_snapshot
        matches = find_similar(
            bundle.embedding, matrix, paths, threshold=threshold, k=SIM_TOP_K
        )
        pixmaps = {m.path: self._get_thumb_for_indexed(m.path) for m in matches}
        dates: dict[Path, datetime | None] = {}
        for m in matches:
            rec = self._index.find_by_path(m.path)
            dates[m.path] = datetime.fromisoformat(rec.edited_date) if rec else None
        self._similar_grid.set_matches(matches, pixmaps, dates)

    def _get_thumb_for_indexed(self, path: Path) -> QPixmap:
        if path in self._pixmap_cache:
            return self._pixmap_cache[path]
        try:
            with Image.open(path) as im:
                pix = to_qpixmap(ImageOps.exif_transpose(im).convert("RGB"), THUMB_GRID)
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
        result = write_date(cur.path, edited)
        if result.warning:
            self._toast.show_message(result.warning)
        self._finalize(cur, edited=edited, action="keep", wrote_disk=result.exif_written)

    def _on_change(self, dt: datetime) -> None:
        cur = self._current
        if cur is None:
            return
        result = write_date(cur.path, dt)
        if result.warning:
            self._toast.show_message(result.warning)
        self._finalize(cur, edited=dt, action="change", wrote_disk=True)

    def _on_threshold_changed(self, thr: float) -> None:
        cur = self._current
        if cur is not None:
            self._refresh_similar(cur, threshold=thr)

    def _on_delete_current_photo(self) -> None:
        cur = self._current
        if cur is None:
            return
        try:
            send2trash.send2trash(str(cur.path))
        except Exception as exc:
            self._toast.show_message(f"Trash failed: {exc}")
            return
        self._advance()

    def _on_delete_similar(self, path: Path) -> None:
        try:
            send2trash.send2trash(str(path))
        except Exception as exc:
            self._toast.show_message(f"Trash failed: {exc}")
            return
        try:
            self._index.delete_by_path(path)
        except Exception as exc:
            self._toast.show_message(f"DB delete failed: {exc}")
        cur = self._current
        if cur is not None:
            self._current_snapshot = self._index.snapshot()
            self._refresh_similar(cur, threshold=self._similar_grid.threshold)

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
        self._current_snapshot = None
        # Show next from queue if ready; else wait for worker signal.
        if self._ready_queue:
            self._present(self._ready_queue.popleft())
        # Pre-fetch the photo after that.
        self._enqueue_next()

    # ----- public API -----

    def show_toast(self, message: str) -> None:
        self._toast.show_message(message)

    # ----- lifecycle -----

    def closeEvent(self, event) -> None:
        self._worker.stop()
        self._worker.wait(5000)
        self._index.close()
        super().closeEvent(event)
