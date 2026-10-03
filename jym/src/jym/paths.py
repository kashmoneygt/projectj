import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the jym folder


def models() -> Path:
    return Path(os.environ.get("JYM_MODELS", ROOT / "models"))
