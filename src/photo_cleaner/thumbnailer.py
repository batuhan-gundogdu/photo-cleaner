"""PIL → bounded thumbnail → QPixmap."""
from __future__ import annotations

import io

from PIL import Image
from PyQt6.QtGui import QPixmap


def to_pil_thumbnail(img: Image.Image, max_side: int) -> Image.Image:
    """Return a thumbnail of ``img`` with the longer side bounded by
    ``max_side``. Never upscales. Preserves aspect ratio. Returns a copy.
    """
    if max(img.size) <= max_side:
        return img.copy()
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def to_qpixmap(img: Image.Image, max_side: int) -> QPixmap:
    """Thumbnail ``img`` to ``max_side`` and convert to QPixmap via PNG bytes."""
    thumb = to_pil_thumbnail(img, max_side)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    pix = QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    return pix
