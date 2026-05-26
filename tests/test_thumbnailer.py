from __future__ import annotations

from PIL import Image

from photo_cleaner.thumbnailer import to_pil_thumbnail, to_qpixmap


def test_pil_thumbnail_bounds_max_side():
    img = Image.new("RGB", (1600, 900))
    out = to_pil_thumbnail(img, 800)
    w, h = out.size
    assert max(w, h) == 800
    # aspect preserved within 1 pixel rounding
    assert abs(w / h - 1600 / 900) < 0.01


def test_pil_thumbnail_does_not_upscale():
    img = Image.new("RGB", (200, 100))
    out = to_pil_thumbnail(img, 800)
    assert out.size == (200, 100)


def test_to_qpixmap_returns_pixmap_with_correct_size(qapp):
    img = Image.new("RGB", (300, 200), color=(123, 45, 67))
    pix = to_qpixmap(img, 160)
    assert not pix.isNull()
    assert max(pix.width(), pix.height()) == 160
