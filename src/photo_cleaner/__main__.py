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
        embedder._ensure_loaded()
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
        win.show_toast(f"Healed {len(healed)} pending rename(s) from a prior crash.")
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
