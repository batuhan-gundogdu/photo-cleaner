"""Background QThread that processes one photo end-to-end and emits a bundle."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

import pillow_heif

from photo_cleaner.config import EMBED_DIM, SIM_TOP_K, THUMB_MAIN
from photo_cleaner.date_extractor import DateInfo, extract
from photo_cleaner.duplicate_finder import Match, find_duplicate, find_similar
from photo_cleaner.embedder import Embedder
from photo_cleaner.index import Index
from photo_cleaner.thumbnailer import to_pil_thumbnail

pillow_heif.register_heif_opener()


@dataclass(frozen=True)
class PhotoBundle:
    path: Path
    date_info: DateInfo
    sha256: str
    embedding: "np.ndarray"  # noqa: F821 — kept loose to avoid import cycle
    thumbnail_main_png: bytes  # PNG-encoded bytes, GUI thread will turn into QPixmap
    duplicate: Match | None
    similar: list[Match]


class Worker(QThread):
    photo_ready = pyqtSignal(PhotoBundle)
    error = pyqtSignal(Path, str)

    def __init__(self, embedder: Embedder, index: Index, parent=None) -> None:
        super().__init__(parent)
        self._embedder = embedder
        self._index = index
        self._queue: "Queue[tuple[Path, float] | None]" = Queue()

    def enqueue(self, path: Path, threshold: float) -> None:
        self._queue.put((path, threshold))

    def stop(self) -> None:
        self._queue.put(None)

    def run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                continue
            if item is None:
                return
            path, threshold = item
            try:
                bundle = self._process(path, threshold)
            except Exception as exc:  # never let the thread die on a bad file
                self.error.emit(path, str(exc))
                continue
            self.photo_ready.emit(bundle)

    def _process(self, path: Path, threshold: float) -> PhotoBundle:
        date_info = extract(path)
        sha = _sha256_of_file(path)
        with Image.open(path) as im:
            im.load()
            embedding = self._embedder.embed(im.convert("RGB"))
            buf = BytesIO()
            to_pil_thumbnail(im.convert("RGB"), THUMB_MAIN).save(buf, format="PNG")
            thumb_png = buf.getvalue()
        matrix, shas, paths = self._index.snapshot()
        dup = find_duplicate(embedding, sha, matrix, shas, paths)
        sim = find_similar(embedding, matrix, paths, threshold=threshold, k=SIM_TOP_K)
        if embedding.shape != (EMBED_DIM,):
            raise RuntimeError(f"bad embedding shape {embedding.shape}")
        return PhotoBundle(
            path=path,
            date_info=date_info,
            sha256=sha,
            embedding=embedding,
            thumbnail_main_png=thumb_png,
            duplicate=dup,
            similar=sim,
        )


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
