import pytest

from jym import core, laya


def test_laya_needs_its_extra(monkeypatch):
    monkeypatch.setattr(laya, "laya", None)
    with pytest.raises(core.Unavailable, match="uv sync --extra laya"):
        laya.LayaPolicy()
