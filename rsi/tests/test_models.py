import json
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rsi import core, folders, models, paths


def test_parse_source_hf_and_path() -> None:
    llama = folders.module(root=paths.backends_dir(), name="llama.cpp", file="backend.py")
    assert llama.parse_source(source="hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf") == (
        "unsloth/Qwen3.8-27B-GGUF",
        "Qwen3.8-27B-UD-Q4_K_M.gguf",
    )
    assert llama.parse_source(source="path:/tmp/m.gguf") == Path("/tmp/m.gguf")
    for bad in ("hf:onlyonepart", "ftp://x"):
        with pytest.raises(ValueError):
            llama.parse_source(source=bad)


def test_an_unknown_model_name_is_an_error() -> None:
    with pytest.raises(KeyError):
        models.get(name="nope")


def test_make_backend() -> None:
    spec = core.ModelSpec(name="m", source="path:/tmp/x.gguf", model_id="x")
    backend = models.make_backend(spec=spec)
    assert type(backend).__name__ == "LlamaServer"
    assert backend.spec is spec
    bad = core.ModelSpec(name="m", source="path:/tmp/x.gguf", model_id="x", backend="vllm")
    with pytest.raises(FileNotFoundError):
        models.make_backend(spec=bad)


def test_server_urls_and_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    spec = core.ModelSpec(name="m", source="path:/tmp/x.gguf", model_id="x", port=1234, ctx=4096)
    server = models.make_backend(spec=spec)
    assert server.host_url() == "http://127.0.0.1:1234/v1"
    assert server.is_running() is False
    assert server.endpoint() == core.ModelEndpoint(
        model_id="x", base_url="http://host.docker.internal:1234/v1", root_url="http://host.docker.internal:1234"
    )


def test_stop_kills_via_pid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # stop() works from the pid file, since `models down` runs in another process
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    spec = core.ModelSpec(name="m", source="path:/tmp/x.gguf", model_id="x")
    server = models.make_backend(spec=spec)
    proc = subprocess.Popen(["sleep", "30"])
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


def test_a_pid_from_before_a_reboot_is_left_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    server = models.make_backend(spec=core.ModelSpec(name="m", source="path:/tmp/x.gguf", model_id="x"))
    other = subprocess.Popen(["sleep", "30"])
    try:
        server.pid_path.parent.mkdir(parents=True, exist_ok=True)
        server.pid_path.write_text(str(other.pid))
        # saved an hour before this process started
        before = time.time() - 3600
        os.utime(server.pid_path, (before, before))
        assert server._starting() is False
        server.stop()
        assert other.poll() is None
        assert not server.pid_path.exists()
    finally:
        other.kill()


class _FakeLlama(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["content-length"])))
        ok = self.path == "/v1/chat/completions" and request["max_tokens"] == 1
        data = json.dumps({"choices": [{"message": {"content": "H"}}]} if ok else {"error": "no"}).encode()
        self.send_response(200 if ok else 500)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # keep the fake server quiet
        pass


def test_llama_identity_and_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    binary = tmp_path / "llama-server"
    binary.write_text(
        "#!/bin/sh\necho 'version: 0.7.0 (build 11669, commit 620d2225c)' >&2\necho 'built with clang' >&2\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("RSI_LLAMA_BIN", str(binary))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = models.make_backend(spec=core.ModelSpec(name="m", source="path:/tmp/x.gguf", model_id="x", port=port))
    assert server.identity() == {
        "version": "version: 0.7.0 (build 11669, commit 620d2225c)",
        "source": "path:/tmp/x.gguf",
    }
    assert server.status_lines() == [f"binary: {binary}", "  version: 0.7.0 (build 11669, commit 620d2225c)"]
    assert server.healthy(timeout_s=1) is False
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _FakeLlama)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert server.healthy(timeout_s=5) is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_llama_refuses_a_memory_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    spec = core.ModelSpec(name="m", source="path:/tmp/x.gguf", model_id="x", port=port, memory_gb=8)
    with pytest.raises(ValueError, match="memory_gb"):
        models.make_backend(spec=spec).start()
