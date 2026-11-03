import logging
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import huggingface_hub

from rsi import core, models, paths

log = logging.getLogger(__name__)

_LOG_CACHE_HIT = re.compile(r"cache hit: ([\d.]+)%")
_LOG_PROMPT = re.compile(r"prompt eval time = [\d.]+ ms / (\d+) tokens")
_LOG_EVAL = re.compile(r"eval time = ([\d.]+) ms / (\d+) new tokens")


def find_llama_bin() -> str:
    env = os.environ.get("RSI_LLAMA_BIN")
    if env:
        return env
    found = shutil.which("llama-server")
    if not found:
        raise RuntimeError(
            "llama-server not on PATH; install the Metal build (brew install llama.cpp) or set RSI_LLAMA_BIN"
        )
    return found


def parse_source(source: str) -> tuple[str, str] | Path:
    if source.startswith("hf:"):
        rest = source.removeprefix("hf:")
        if "/" not in rest:
            raise ValueError(f"hf source must be hf:<org>/<name>/<file>, got {source!r}")
        repo, filename = rest.rsplit("/", 1)
        return repo, filename
    if source.startswith("path:"):
        return Path(source.removeprefix("path:"))
    raise ValueError(f"unsupported model source: {source!r}")


def resolve_gguf(spec: core.ModelSpec) -> Path:
    parsed = parse_source(source=spec.source)
    if isinstance(parsed, Path):
        if not parsed.exists():
            raise FileNotFoundError(f"local GGUF not found: {parsed}")
        return parsed
    repo, filename = parsed
    dest_dir = paths.models_dir() / "gguf" / repo
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if not dest.exists():
        log.info("downloading %s/%s ...", repo, filename)
        dest = Path(huggingface_hub.hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest_dir)))
    return dest


class LlamaServer(models.LocalServer):
    program = "llama-server"
    ready_path = "/v1/models"

    def start(self) -> None:
        if self.is_running() or self._starting():
            return
        gguf = resolve_gguf(spec=self.spec)
        self._launch(
            args=[
                find_llama_bin(),
                "-m",
                str(gguf),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.spec.port),
                "--ctx-size",
                str(self.spec.ctx),
                "--parallel",
                str(self.spec.parallel),
                "--jinja",
                "--alias",
                self.spec.model_id,
                "--seed",
                str(self.spec.seed),
            ]
        )

    def _terminate(self, pid: int) -> None:
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not models.pid_alive(pid=pid):
                break
            time.sleep(0.2)
        if models.pid_alive(pid=pid):
            os.kill(pid, signal.SIGKILL)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        else:
            pid = self._pid()
            if pid is not None and models.pid_alive(pid=pid):
                self._terminate(pid=pid)
        self.pid_path.unlink(missing_ok=True)

    def status_lines(self) -> list[str]:
        return [f"binary: {find_llama_bin()}", f"  {self._version()}"]

    def _version(self) -> str:
        # llama-server prints its version on stderr
        done = subprocess.run([find_llama_bin(), "--version"], capture_output=True, text=True, check=True)
        return (done.stdout or done.stderr).strip().splitlines()[0]

    def identity(self) -> dict[str, str]:
        return {"version": self._version(), "source": self.spec.source}

    def log_stats(self) -> core.ServerStats | None:
        # server-wide numbers, since llama-server doesn't log them per request
        log_file = self.log_path
        if not log_file.exists():
            return None
        cache_hits: list[float] = []
        prompt_tokens = completion_tokens = 0
        gen_ms = 0.0
        for line in log_file.read_text(errors="replace").splitlines():
            m = _LOG_CACHE_HIT.search(line)
            if m:
                cache_hits.append(float(m.group(1)))
            m = _LOG_PROMPT.search(line)
            if m:
                prompt_tokens += int(m.group(1))
            m = _LOG_EVAL.search(line)
            if m:
                gen_ms += float(m.group(1))
                completion_tokens += int(m.group(2))
        if completion_tokens == 0:
            return None
        return core.ServerStats(
            requests=len(cache_hits),
            cache_hits=cache_hits,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tok_s=completion_tokens / (gen_ms / 1000.0),
        )


BACKEND = LlamaServer
