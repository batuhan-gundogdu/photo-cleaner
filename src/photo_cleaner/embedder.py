"""Image embedding via open_clip (lazy-loaded), with a test-friendly stub path."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PIL import Image

from photo_cleaner.config import CLIP_MODEL_NAME, CLIP_MODEL_PRETRAINED, EMBED_DIM


class Embedder:
    """Encode a PIL image as an L2-normalized float32 vector of shape (EMBED_DIM,).

    Construct with no args for the real model (lazy-loaded on first ``embed``).
    Use :py:meth:`from_callable` in tests to inject a deterministic embedding.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        pretrained: str = CLIP_MODEL_PRETRAINED,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._torch = None
        self._embed_callable: Optional[Callable[[Image.Image], np.ndarray]] = None

    @classmethod
    def from_callable(cls, fn: Callable[[Image.Image], np.ndarray]) -> "Embedder":
        e = cls.__new__(cls)
        e._model_name = "<stub>"
        e._pretrained = "<stub>"
        e._model = None
        e._preprocess = None
        e._torch = None
        e._embed_callable = fn
        return e

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._embed_callable is not None:
            return
        import open_clip  # heavy; defer
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._torch = torch

    def embed(self, image: Image.Image) -> np.ndarray:
        if self._embed_callable is not None:
            raw = self._embed_callable(image).astype(np.float32, copy=False)
        else:
            self._ensure_loaded()
            assert self._model is not None and self._preprocess is not None
            torch = self._torch
            with torch.no_grad():
                t = self._preprocess(image).unsqueeze(0)
                feats = self._model.encode_image(t)
            raw = feats.squeeze(0).cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(raw))
        if norm == 0.0:
            raise ValueError("embedding has zero norm")
        v = raw / norm
        if v.shape != (EMBED_DIM,):
            raise ValueError(f"expected ({EMBED_DIM},), got {v.shape}")
        return v
