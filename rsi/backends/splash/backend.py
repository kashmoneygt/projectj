import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request

from rsi import core, models

# per-request defaults that would override models.toml
REQUEST_DEFAULT_VARS = ("SPLASH_DEFAULT_REASONING_EFFORT", "SPLASH_THINKING_TOKEN_BUDGET", "SPLASH_THINKING_END_TEXT")


def find_splash_bin() -> str:
    env = os.environ.get("RSI_SPLASH_BIN")
    if env:
        return env
    found = shutil.which("splash")
    if not found:
        raise RuntimeError("splash not on PATH; install it (brew install incoai/tap/splash) or set RSI_SPLASH_BIN")
    return found


def splash_ref(source: str) -> str:
    # splash resolves OWNER/REPO[:VARIANT] on HF
    if not source.startswith("splash:"):
        raise ValueError(f"splash backend needs a 'splash:OWNER/REPO[:VARIANT]' source, got {source!r}")
    ref = source.removeprefix("splash:")
    if "/" not in ref:
        raise ValueError(f"splash source must be splash:OWNER/REPO[:VARIANT], got {source!r}")
    return ref


class SplashServer(models.LocalServer):
    """The Qwen 3.5/3.6/3.8 families on MLX, with speculative decoding"""

    program = "splash"
    ready_path = "/ready"

    def __init__(self, spec: core.ModelSpec) -> None:
        super().__init__(spec=spec)
        self.ref = splash_ref(source=spec.source)

    def _cmd(self) -> list[str]:
        return [
            find_splash_bin(),
            "serve",
            "--model",
            self.ref,
            # never fetch new weights, templates or drafts
            "--offline",
            "--port",
            str(self.spec.port),
            # the name containers call it by
            "--allowed-host",
            "host.docker.internal",
            "--max-context",
            str(self.spec.ctx),
            "--served-model-name",
            self.spec.model_id,
            "--announce-served-name",
            "--language-only",
            "--no-webui",
            "--idle-release",
            "off",
            *(["--max-memory", f"{self.spec.memory_gb}G"] if self.spec.memory_gb else []),
        ]

    def start(self) -> None:
        if self.is_running() or self._starting():
            return
        env = {key: value for key, value in os.environ.items() if key not in REQUEST_DEFAULT_VARS}
        # own session, so stop() can kill the whole process tree
        self._launch(args=self._cmd(), env=env, new_session=True)

    def wait_ready(self, timeout: float = 900.0) -> None:
        super().wait_ready(timeout=timeout)
        self._check_window()

    def _status(self, timeout: float) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.spec.port}/status", timeout=timeout) as resp:
            return json.loads(resp.read())

    def _check_window(self) -> None:
        # --max-context and --max-memory can only lower the window splash picks, so check it
        window = self._status(timeout=5).get("maximum_context_tokens")
        if window != self.spec.ctx:
            self.stop()
            raise RuntimeError(
                f"splash ran {self.spec.name} with a {window}-token window, not the {self.spec.ctx} in models.toml, "
                "so we stopped it. Lower ctx, or raise memory_gb, to what this machine can hold"
            )

    def stop(self) -> None:
        # the pid is the session leader, so killpg stops the whole tree
        pid = self._pid()
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return
            for _ in range(75):
                if not models.pid_alive(pid=pid):
                    break
                time.sleep(0.2)
            if models.pid_alive(pid=pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        self.pid_path.unlink(missing_ok=True)

    def status_lines(self) -> list[str]:
        binary = find_splash_bin()
        lines = [f"binary: {binary}", f"model: {self.ref}"]
        version = subprocess.run([binary, "--version"], capture_output=True, text=True, check=False)
        if version.stdout.strip():
            lines.append(f"  {version.stdout.strip().splitlines()[0]}")
        try:
            status = self._status(timeout=2)
            lines.append(
                f"  status: ready={status.get('ready')} memory={status.get('memory_pressure')} "
                f"thermal={status.get('thermal_state')} ctx={status.get('maximum_context_tokens')}"
            )
        except OSError:
            lines.append("  status: not running")
        return lines

    def identity(self) -> dict[str, str]:
        """Version, weights hash and engine build of the running server"""
        found = self._status(timeout=5)["identity"]
        version = subprocess.run([find_splash_bin(), "--version"], capture_output=True, text=True, check=True)
        return {
            "version": version.stdout.strip(),
            "source": self.spec.source,
            "target_model_sha256": found["kv"]["target_model_sha256"],
            "build_id": found["cache"]["build_id"],
        }


BACKEND = SplashServer
