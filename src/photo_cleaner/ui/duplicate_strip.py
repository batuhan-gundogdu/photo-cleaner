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
