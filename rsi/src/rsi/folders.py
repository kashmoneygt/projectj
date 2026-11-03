import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def names(root: Path, file: str) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / file).is_file())


def module(root: Path, name: str, file: str, kind: str = "module") -> ModuleType:
    """Load <root>/<name>/<file> once (dashes and dots in name become underscores)"""
    folder = name.replace("-", "_").replace(".", "_")
    path = root / folder / file
    if not path.is_file():
        raise FileNotFoundError(f"{kind} {name!r} not found at {path}; known: {names(root=root, file=file)}")
    key = f"rsi.folder.{folder}.{path.stem}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod  # dataclasses and pydantic look modules up here
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(key)
            raise
    return sys.modules[key]
