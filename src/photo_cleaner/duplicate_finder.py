"""Pure functions for duplicate + similar queries over an embeddings matrix."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from photo_cleaner.config import DUP_THRESHOLD


@dataclass(frozen=True)
class Match:
    path: Path
    similarity: float
    reason: str  # 'sha256' | 'cosine'


def find_duplicate(
    query: np.ndarray,
    query_sha: str,
    matrix: np.ndarray,
    shas: list[str],
    paths: list[Path],
) -> Match | None:
    """Exact SHA256 match wins; otherwise the highest cosine >= DUP_THRESHOLD."""
    if len(shas) == 0:
        return None
    for i, s in enumerate(shas):
        if s == query_sha:
            sim = float(np.dot(matrix[i], query))
            return Match(path=paths[i], similarity=sim, reason="sha256")
    sims = matrix @ query  # both are L2-normalized -> cosine
    best = int(np.argmax(sims))
    if float(sims[best]) >= DUP_THRESHOLD:
        return Match(path=paths[best], similarity=float(sims[best]), reason="cosine")
    return None


def find_similar(
    query: np.ndarray,
    matrix: np.ndarray,
    paths: list[Path],
    threshold: float,
    k: int,
) -> list[Match]:
    """Return up to ``k`` matches with cosine >= ``threshold``, sorted desc."""
    if matrix.shape[0] == 0:
        return []
    sims = matrix @ query
    idxs = np.argsort(-sims)
    out: list[Match] = []
    for i in idxs[: max(k, 0)]:
        s = float(sims[i])
        if s < threshold:
            break
        out.append(Match(path=paths[i], similarity=s, reason="cosine"))
    return out
