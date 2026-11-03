import http.client
import json
import logging
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from rsi import core, folders, paths

log = logging.getLogger(__name__)


def registry() -> dict[str, core.ModelSpec]:
    return {spec.name: spec for spec in core.load_models(path=paths.models_toml_path())}


def get(name: str) -> core.ModelSpec:
    reg = registry()
    if name not in reg:
        raise KeyError(f"unknown model {name!r}; known: {sorted(reg)}")
    return reg[name]


def teacher(name: str) -> core.TeacherSpec:
    teachers = {spec.name: spec for spec in core.load_teachers(path=paths.models_toml_path())}
    if name not in teachers:
        raise KeyError(f"unknown teacher {name!r}; known: {sorted(teachers)}")
    return teachers[name]


def make_backend(spec: core.ModelSpec) -> core.ModelBackend:
    return folders.module(
        root=paths.backends_dir(), name=spec.backend, file="backend.py", kind="model backend"
    ).BACKEND(spec=spec)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LocalServer:
    """Shared base for model servers on the host machine"""

    program = ""
    ready_path = ""  # returns 200 once the model is loaded

    def __init__(self, spec: core.ModelSpec) -> None:
        self.spec = spec
        self.proc: subprocess.Popen | None = None  # set when this process started it

    @property
    def log_path(self) -> Path:
        return paths.models_dir() / "logs" / f"{self.spec.name}.log"

    @property
    def pid_path(self) -> Path:
        """Lets `models down` find the server from another process"""
        return paths.models_dir() / "pids" / f"{self.spec.name}.pid"

    def host_url(self) -> str:
        return f"http://127.0.0.1:{self.spec.port}/v1"

    def endpoint(self) -> core.ModelEndpoint:
        root = f"http://host.docker.internal:{self.spec.port}"
        return core.ModelEndpoint(model_id=self.spec.model_id, base_url=f"{root}/v1", root_url=root)

    def is_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.spec.port}{self.ready_path}", timeout=2) as resp:
                return resp.status == 200
        except OSError:
            return False

    def healthy(self, timeout_s: float) -> bool:
        """Check that the model answers a one-token request"""
        payload = {"model": self.spec.model_id, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
        request = urllib.request.Request(
            f"{self.host_url()}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                return bool(json.loads(resp.read()).get("choices"))
        except (OSError, ValueError, http.client.HTTPException):
            return False

    def wait_ready(self, timeout: float = 300.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return
            if self.proc is not None:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.program} exited early (code {self.proc.returncode}); see {self.log_path}"
                    )
            else:
                # started by an earlier `models up`, so only its pid is known
                pid = self._pid()
                if pid is None or not pid_alive(pid=pid):
                    raise RuntimeError(f"{self.program} for {self.spec.name} died while starting; see {self.log_path}")
            time.sleep(2)
        raise TimeoutError(f"{self.program} for {self.spec.name} not ready after {timeout:.0f}s")

    def _pid(self) -> int | None:
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text().strip())
        except ValueError:
            return None

    def _starting(self) -> bool:
        """Check if an earlier `models up` is still loading the model"""
        pid = self._pid()
        if pid is None or not pid_alive(pid=pid):
            return False
        log.info("%s already starting (pid %s); waiting for readiness", self.program, pid)
        return True

    def _launch(self, args: list[str], env: dict[str, str] | None = None, new_session: bool = False) -> None:
        log.info("%s args: %s", self.program, " ".join(args))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            args, stdout=self.log_path.open("ab"), stderr=subprocess.STDOUT, start_new_session=new_session, env=env
        )
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(self.proc.pid))
