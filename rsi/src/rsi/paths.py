import os
import re
from pathlib import Path

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def rsi_root() -> Path:
    override = os.environ.get("RSI_ROOT")
    return Path(override).resolve() if override else project_dir()


def project_dir() -> Path:
    """The rsi source folder, which RSI_ROOT doesn't move"""
    return Path(__file__).resolve().parents[2]


def runs_dir() -> Path:
    return rsi_root() / "runs"


def models_dir() -> Path:
    return rsi_root() / "models"


def logs_dir() -> Path:
    return rsi_root() / "logs"


def harnesses_dir() -> Path:
    return project_dir() / "harnesses"


def benchmarks_dir() -> Path:
    return project_dir() / "benchmarks"


def backends_dir() -> Path:
    return project_dir() / "backends"


def teachers_dir() -> Path:
    return project_dir() / "teachers"


def models_toml_path() -> Path:
    return rsi_root() / "models.toml"


def node_dir(name: str) -> Path:
    if not _SAFE_NAME.match(name):
        raise ValueError(f"unsafe name: {name!r}")
    return runs_dir() / name


def instances_dir(node_name: str) -> Path:
    return node_dir(name=node_name) / "instances"


def variants_dir() -> Path:
    return rsi_root() / "variants"


def tree_path() -> Path:
    return variants_dir() / "tree.json"
