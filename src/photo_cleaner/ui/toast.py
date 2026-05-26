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
