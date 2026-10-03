import importlib.metadata
import json
import os
import platform
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from jym import core

FORMAT = 2
RUN_ID = re.compile(r"[0-9a-f]{32}")
SECRET_VARIABLE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$")
# flush frequent events at most this often (network storage is slow)
FLUSH_SECONDS = 0.5
FREQUENT = frozenset({"scene", "controls"})


class RunLog:
    """A run's folder: manifest.json, events.jsonl and result.json"""

    def __init__(self, runs_dir: Path | str, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex
        self.path = Path(runs_dir) / self.run_id
        self.path.mkdir(parents=True)
        (self.path / "events.jsonl").touch()
        self.started = time.monotonic()
        self.secrets = {value for name, value in os.environ.items() if SECRET_VARIABLE.search(name) and len(value) >= 8}
        self.listener: Callable[[dict], Any] | None = None  # called with each logged event
        self.lock = threading.Lock()
        self.events: TextIO | None = None
        self.flushed = time.monotonic()

    def redact(self, data: Any) -> str:
        text = json.dumps(data, default=_plain, allow_nan=False)
        for secret in self.secrets:
            text = text.replace(json.dumps(secret)[1:-1], "[REDACTED]")  # as escaped in the json
        return text

    def write(self, name: str, data: Any) -> None:
        self.close()  # flush events before writing the manifest or result
        temporary = self.path / f".{name}.tmp"
        temporary.write_text(self.redact(data) + "\n", encoding="utf-8")
        temporary.replace(self.path / name)

    def event(self, kind: str, **data: Any) -> None:
        record = {"type": kind, "time": time.time(), "elapsed": round(time.monotonic() - self.started, 4), **data}
        with self.lock:
            line = self.redact(record)
            if self.events is None:
                self.events = (self.path / "events.jsonl").open("a", encoding="utf-8")
            self.events.write(line + "\n")
            if kind not in FREQUENT or time.monotonic() - self.flushed >= FLUSH_SECONDS:
                self.events.flush()
                self.flushed = time.monotonic()
        if self.listener is not None:
            self.listener(json.loads(line))

    def close(self) -> None:
        with self.lock:
            if self.events is not None:
                self.events.close()
                self.events = None


def manifest(
    run_id: str, environment: core.Environment, policy: str, metadata: dict, task: core.Task, seed: int, mode: str
) -> dict:
    return {
        "format": FORMAT,
        "run_id": run_id,
        "mode": mode,
        "created": time.time(),
        "task": task,
        "seed": seed,
        "policy": {"name": policy, "metadata": metadata},
        "environment": {"version": environment.version},
        "versions": {
            "python": platform.python_version(),
            "jym": importlib.metadata.version("jym"),
            "system": platform.platform(),
        },
    }


def load_run(run_id: str, runs_dir: Path | str) -> dict:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"Invalid run ID: {run_id!r}")
    path = Path(runs_dir) / run_id
    # drop the last line, which a live run may still be writing
    lines = (path / "events.jsonl").read_text(encoding="utf-8").split("\n")[:-1]
    result = path / "result.json"
    return {
        "manifest": json.loads((path / "manifest.json").read_text(encoding="utf-8")),
        "events": [json.loads(line) for line in lines if line],
        "result": json.loads(result.read_text(encoding="utf-8")) if result.is_file() else None,
    }


def recent_runs(runs_dir: Path | str) -> list[dict]:
    rows = []
    root = Path(runs_dir)
    for path in root.iterdir() if root.is_dir() else []:
        try:
            recorded = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not RUN_ID.fullmatch(path.name) or recorded.get("format") != FORMAT:
            continue
        result_path = path / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        task = recorded["task"]
        rows.append(
            {
                "run_id": path.name,
                "created": recorded["created"],
                "env": task["env"],
                "task": task["id"],
                "task_version": task["version"],
                "player": recorded["policy"]["name"],
                "seed": recorded["seed"],
                "status": result.get("status"),
                "score": result.get("score"),
            }
        )
    return sorted(rows, key=lambda row: row["created"], reverse=True)


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, set | frozenset):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot record {type(value).__name__} as JSON")
