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
