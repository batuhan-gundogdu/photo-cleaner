"""App-wide constants and paths."""
from __future__ import annotations

from pathlib import Path

PICTURES_ROOT = Path.home() / "Google Drive" / "My Drive" / "Pictures"
APP_DATA = Path.home() / ".photo-cleaner"
DB_PATH = PICTURES_ROOT / "index.db"
ERROR_LOG = APP_DATA / "errors.log"

DONE_PREFIX = "_done_"
SUPPORTED_EXTS = frozenset({".jpg", ".jpeg", ".png", ".heic"})

EMBED_DIM = 512
DUP_THRESHOLD = 0.999
SIM_THRESHOLD_DEFAULT = 0.92
SIM_THRESHOLD_MIN = 0.80
SIM_THRESHOLD_MAX = 0.99
SIM_TOP_K = 12

THUMB_MAIN = 800
THUMB_GRID = 160

# open_clip model — small, CPU-friendly, 512-d embeddings.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_MODEL_PRETRAINED = "openai"


def ensure_app_data() -> None:
    """Create APP_DATA and PICTURES_ROOT if missing. Called once at startup."""
    APP_DATA.mkdir(parents=True, exist_ok=True)
    PICTURES_ROOT.mkdir(parents=True, exist_ok=True)
