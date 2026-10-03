import dataclasses
import functools
import tomllib
from collections.abc import Callable
from pathlib import Path

from jym import core, environments, jev, laya, llama, logits, paths

DEVICES = ("cpu", "gpu")
# default deployed text for GGUF models
DEPLOYED = "Runs on llama.cpp ({device}) and answers with the most likely option letter for each question."
POLICIES = {"jev": jev.JevPolicy, "laya": laya.LayaPolicy}
READINESS = {"jev": jev.readiness, "laya": laya.readiness}


@dataclasses.dataclass(frozen=True)
class Player:
    id: str
    title: str
    make: Callable[[core.Environment | None], core.Policy]
    role: str | None = None  # shown beside its title
    trained: str = ""  # how it was trained
    deployed: str = ""  # how it runs here
    model: str | None = None  # model id, if known before loading
    per_run: bool = False  # created per run instead of loaded once and shared


@functools.cache
def models() -> dict:
    with (paths.ROOT / "models.toml").open("rb") as file:
        return tomllib.load(file)


def featured() -> list[str]:
    return models()["featured"]


def player(name: str, env: str) -> Player:
    """Look up a player by models.toml name or GGUF path, with an optional @cpu or @gpu"""
    if name == "reference":
        reference = environments.ENVIRONMENTS[env].reference
        return Player(id="reference", make=lambda environment: environment.reference(), per_run=True, **reference)
    base, at, device = name.rpartition("@")
    if not at or device not in DEVICES:
        base, device = name, ""
    if base.endswith(".gguf"):
        path = Path(base).resolve()
        if not path.is_file():
            raise ValueError(f"no GGUF file at {path}")
        return gguf(
            model=llama.Gguf(name=path.stem, file=str(path)), device=device or "cpu", entry={"title": path.stem}
        )
    entry = models()["models"].get(base)
    if entry is None or (device and entry["type"] != "gguf"):
        choices = ["reference", *models()["models"]]
        raise ValueError(f"unknown player {name!r}; choose from {choices} (@cpu or @gpu for GGUF) or a GGUF file")
    if entry["type"] == "gguf":
        return gguf(source(base, entry), device or "cpu", entry)
    make = POLICIES[entry["type"]]
    return Player(
        base,
        entry["title"],
        lambda environment: make(),
        entry.get("role"),
        entry.get("trained", ""),
        entry.get("deployed", ""),
    )


def source(name: str, entry: dict) -> llama.Gguf:
    """Find a GGUF model's file from a source like "hf:owner/repo/file.gguf", or a path in the jym folder"""
    if entry["source"].startswith("hf:"):
        owner, repo, file = entry["source"].removeprefix("hf:").split("/", 2)
        return llama.Gguf(name=name, file=file, repo=f"{owner}/{repo}", revision=entry.get("revision", "main"))
    return llama.Gguf(name=name, file=str(paths.ROOT / entry["source"]))


def gguf(model: llama.Gguf, device: str, entry: dict) -> Player:
    name = f"{model.name}@{device}"
    return Player(
        id=name,
        title=entry["title"] + (" (GPU)" if device == "gpu" else ""),
        make=lambda environment: logits.LogitsPolicy(name, model, device),
        role=entry.get("role"),
        trained=entry.get("trained", ""),
        deployed=entry.get("deployed", DEPLOYED).format(device=device.upper()),
        model=model.identity() if model.repo else None,
    )


def known(env: str) -> dict[str, Player]:
    names = ["reference"]
    for name, entry in models()["models"].items():
        names += [f"{name}@{device}" for device in DEVICES] if entry["type"] == "gguf" else [name]
    return {found.id: found for found in (player(name, env) for name in names)}


def readiness(name: str, entry: dict) -> str:
    if entry["type"] != "gguf":
        return READINESS[entry["type"]]()
    model = source(name, entry)
    if model.path.is_file():
        return f"ready ({model.path})"
    return f"downloads to {model.path} on first use" if model.repo else f"no file at {model.path}"
