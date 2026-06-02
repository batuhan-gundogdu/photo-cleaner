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
    skip_clicked = pyqtSignal()
    delete_clicked = pyqtSignal()

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
        self._skip_btn = QPushButton("Skip")
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setStyleSheet("color: #b00;")
        self._keep_btn.clicked.connect(self.keep_clicked.emit)
        self._change_btn.clicked.connect(self._on_change)
        self._skip_btn.clicked.connect(self.skip_clicked.emit)
        self._delete_btn.clicked.connect(self.delete_clicked.emit)

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
        btn_row.addWidget(self._skip_btn)
        btn_row.addWidget(self._delete_btn)
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

    def set_date(self, dt: datetime) -> None:
        self._date_edit.setDate(QDate(dt.year, dt.month, dt.day))

    def _on_change(self) -> None:
        d: QDate = self._date_edit.date()
        self.change_clicked.emit(datetime(d.year(), d.month(), d.day()))
