import json
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rsi import core, models


def _spec(
    source: str = "splash:unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
    port: int = 1234,
    ctx: int = 4096,
    model_id: str = "qwen-3.8-27b",
) -> core.ModelSpec:
    return core.ModelSpec(name="m", source=source, model_id=model_id, backend="splash", port=port, ctx=ctx)


_STATUS = {
    "ready": True,
    "memory_pressure": "normal",
    "thermal_state": "nominal",
    "maximum_context_tokens": 4096,
    "identity": {
        "cache": {"loaded_model_layout_sha256": "2ec9", "build_id": "src-f6a5"},
        "kv": {"target_model_sha256": "31b1", "format": "int8"},
    },
}


class _FakeSplash(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._json(status=200, body={"status": "ok"} if self.path == "/ready" else _STATUS)

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if request["model"] == "stuck":
            time.sleep(2)
        elif self.path != "/v1/chat/completions" or request["max_tokens"] != 1 or request["model"] == "broken":
            self._json(status=500, body={"error": "engine failure"})
        else:
            self._json(status=200, body={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # keep the fake server quiet
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_qwen_is_served_by_splash() -> None:
    assert type(models.make_backend(spec=models.get(name="qwen-3.8-27b"))).__name__ == "SplashServer"


def test_splash_requires_splash_source() -> None:
    for bad in ("hf:unsloth/Qwen3.8-27B-GGUF/x.gguf", "path:/tmp/x.gguf", "splash:onlyonepart"):
        with pytest.raises(ValueError):
            models.make_backend(spec=_spec(source=bad))


def test_splash_command_and_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    monkeypatch.setenv("RSI_SPLASH_BIN", "splash")
    server = models.make_backend(spec=_spec())
    assert server.host_url() == "http://127.0.0.1:1234/v1"
    assert server._cmd() == [
        "splash",
        "serve",
        "--model",
        "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
        "--offline",
        "--port",
        "1234",
        "--allowed-host",
        "host.docker.internal",
        "--max-context",
        "4096",
        "--served-model-name",
        "qwen-3.8-27b",
        "--announce-served-name",
        "--language-only",
        "--no-webui",
        "--idle-release",
        "off",
    ]
    assert server.endpoint() == core.ModelEndpoint(
        model_id="qwen-3.8-27b",
        base_url="http://host.docker.internal:1234/v1",
        root_url="http://host.docker.internal:1234",
    )


def test_splash_caps_its_memory_only_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    monkeypatch.setenv("RSI_SPLASH_BIN", "splash")
    capped = models.make_backend(spec=_spec().model_copy(update={"memory_gb": 24}))
    assert capped._cmd()[-2:] == ["--max-memory", "24G"]
    assert "--max-memory" not in models.make_backend(spec=_spec())._cmd()


def test_splash_ready_status_and_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    monkeypatch.setenv("RSI_SPLASH_BIN", "echo")
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _FakeSplash)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        server = models.make_backend(spec=_spec(port=port))
        assert server.is_running() is True
        server.proc = subprocess.Popen(["sleep", "30"])
        try:
            server.wait_ready(timeout=5)
            lines = server.status_lines()
            assert any("ready=True" in line for line in lines)
            assert any("memory=normal" in line for line in lines)
        finally:
            server.proc.kill()
    finally:
        httpd.shutdown()
    assert server.is_running() is False


def test_splash_start_leaves_out_request_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    monkeypatch.setenv("RSI_SPLASH_BIN", "splash")
    monkeypatch.setenv("SPLASH_THINKING_TOKEN_BUDGET", "300")
    monkeypatch.setenv("SPLASH_DEFAULT_REASONING_EFFORT", "low")
    monkeypatch.setenv("HF_HOME", "/tmp/hf")
    envs: list[dict[str, str]] = []

    class _Proc:
        pid = 4321

    def fake_popen(
        args: list[str], stdout: object, stderr: object, start_new_session: bool, env: dict[str, str]
    ) -> _Proc:
        envs.append(env)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    models.make_backend(spec=_spec(port=_free_port())).start()
    assert "SPLASH_THINKING_TOKEN_BUDGET" not in envs[0]
    assert "SPLASH_DEFAULT_REASONING_EFFORT" not in envs[0]
    assert envs[0]["HF_HOME"] == "/tmp/hf"


def test_splash_wait_ready_stops_a_server_with_a_smaller_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _FakeSplash)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    stopped: list[bool] = []
    try:
        server = models.make_backend(spec=_spec(port=port, ctx=8192))
        server.stop = lambda: stopped.append(True)
        with pytest.raises(RuntimeError, match="4096-token window, not the 8192"):
            server.wait_ready(timeout=5)
    finally:
        httpd.shutdown()
    assert stopped == [True]


def test_splash_stop_kills_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    server = models.make_backend(spec=_spec())
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    server.pid_path.parent.mkdir(parents=True, exist_ok=True)
    server.pid_path.write_text(str(proc.pid))
    try:
        server.stop()
        deadline = time.time() + 5
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
        assert proc.poll() is not None
        assert not server.pid_path.exists()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_splash_identity_and_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    splash = tmp_path / "splash"
    splash.write_text('#!/bin/sh\necho "Splash 1.3.2"\n')
    splash.chmod(0o755)
    monkeypatch.setenv("RSI_SPLASH_BIN", str(splash))
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _FakeSplash)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    server = models.make_backend(spec=_spec(port=port))
    try:
        assert server.identity() == {
            "version": "Splash 1.3.2",
            "source": "splash:unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
            "target_model_sha256": "31b1",
            "build_id": "src-f6a5",
        }
        assert server.healthy(timeout_s=5) is True
        for model_id in ("broken", "stuck"):
            assert models.make_backend(spec=_spec(port=port, model_id=model_id)).healthy(timeout_s=0.5) is False
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert server.healthy(timeout_s=1) is False
