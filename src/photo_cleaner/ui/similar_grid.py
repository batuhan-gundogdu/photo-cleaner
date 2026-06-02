"""Right-pane bottom: threshold slider + scrollable grid of similar thumbs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
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
    delete_similar_clicked = pyqtSignal(Path)
    delete_current_for_similar = pyqtSignal()
    copy_date_clicked = pyqtSignal(datetime)

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

    def set_matches(
        self,
        matches: list[Match],
        pixmaps: dict[Path, QPixmap],
        dates: dict[Path, datetime | None] | None = None,
    ) -> None:
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
            dt = (dates or {}).get(m.path)
            date_btn = QPushButton(dt.strftime("%Y-%m-%d") if dt else "—")
            date_btn.setFixedWidth(THUMB_GRID)
            date_btn.setToolTip("Click to use this date")
            date_btn.setEnabled(dt is not None)
            if dt is not None:
                date_btn.clicked.connect(lambda checked, d=dt: self.copy_date_clicked.emit(d))
            del_old_btn = QPushButton("Delete old")
            del_old_btn.setFixedWidth(THUMB_GRID)
            del_old_btn.setStyleSheet("color: #b00;")
            del_old_btn.clicked.connect(lambda checked, p=m.path: self.delete_similar_clicked.emit(p))
            del_cur_btn = QPushButton("Delete current")
            del_cur_btn.setFixedWidth(THUMB_GRID)
            del_cur_btn.setStyleSheet("color: #b00;")
            del_cur_btn.clicked.connect(self.delete_current_for_similar.emit)
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            cell_layout.addWidget(lbl)
            cell_layout.addWidget(date_btn)
            cell_layout.addWidget(del_old_btn)
            cell_layout.addWidget(del_cur_btn)
            self._grid.addWidget(cell, i // cols, i % cols)
        self._count_label.setText(f"({len(matches)} matches ≥ {self.threshold:.2f})")

    def _on_slider(self, value: int) -> None:
        thr = value / 100.0
        self._readout.setText(f"SIMILAR ABOVE: {thr:.2f}")
        self.threshold_changed.emit(thr)
