from pathlib import Path

import pytest

from rsi import folders


def test_names_and_module(tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "thing.py").write_text("THING = 1\n")
    (tmp_path / "beta").mkdir()
    assert folders.names(root=tmp_path, file="thing.py") == ["alpha"]
    mod = folders.module(root=tmp_path, name="alpha", file="thing.py")
    assert mod.THING == 1
    (tmp_path / "my_thing").mkdir()
    (tmp_path / "my_thing" / "thing.py").write_text("THING = 2\n")
    assert folders.module(root=tmp_path, name="my-thing", file="thing.py").THING == 2


def test_module_unknown_lists_known(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="known:"):
        folders.module(root=tmp_path, name="nope", file="thing.py", kind="thing")
