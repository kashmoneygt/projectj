import asyncio
import collections
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi import core, folders, harness, paths

pi = folders.module(root=paths.harnesses_dir(), name="pi", file="harness.py", kind="harness")
forwarder = folders.module(root=paths.harnesses_dir(), name="pi", file="forwarder.py")


def _read(direction: str, data: bytes, methods: tuple[str, ...] = (), piece: int = 5) -> list[dict]:
    async def go() -> list[dict]:
        entries: list[dict] = []
        tap = forwarder.Tap(
            read=getattr(forwarder, direction),
            methods=collections.deque(methods),
            log=lambda entry: entries.append(entry),
        )
        for start in range(0, len(data), piece):
            tap.feed(data[start : start + piece])
            await asyncio.sleep(0)
        tap.end()
        await tap.done
        return entries

    return asyncio.run(go())


def _post(path: str, body: bytes) -> bytes:
    head = f"POST {path} HTTP/1.1\r\nHost: host.docker.internal:11501\r\nContent-Type: application/json\r\n"
    return f"{head}Content-Length: {len(body)}\r\n\r\n".encode() + body


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def _reply(body: bytes) -> bytes:
    return f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body


def test_forwarder_logs_each_requests_parameters_without_the_conversation() -> None:
    kwargs = {"enable_thinking": True, "reasoning_effort": "medium"}
    chat = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "fix it " * 2000}],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
        "stream": True,
        "temperature": 1,
        "chat_template_kwargs": kwargs,
    }
    tokenize = {"content": "x" * 3000}
    data = (
        _post(path="/v1/chat/completions", body=json.dumps(chat).encode())
        + _post(path="/tokenize", body=json.dumps(tokenize).encode())
        + b"GET /v1/models HTTP/1.1\r\nHost: host.docker.internal:11501\r\n\r\n"
    )
    assert _read(direction="read_requests", data=data, piece=7) == [
        {
            "method": "POST",
            "path": "/v1/chat/completions",
            "params": {"model": "qwen", "stream": True, "temperature": 1, "chat_template_kwargs": kwargs},
        },
        {"method": "POST", "path": "/tokenize", "params": {"content": "(3002 bytes)"}},
        {"method": "GET", "path": "/v1/models"},
    ]


def test_forwarder_says_what_it_couldnt_read() -> None:
    chunked = (
        b"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n" + _chunk(data=b"{}") + b"0\r\n\r\n"
    )
    data = chunked + _post(path="/v1/chat/completions", body=b"{}")
    assert _read(direction="read_requests", data=data) == [
        {"method": "POST", "path": "/v1/chat/completions", "unread": "chunked body"}
    ]
    for body in (b"temperature=0", b"[1, 2]"):
        assert _read(direction="read_requests", data=_post(path="/v1/chat/completions", body=body)) == [
            {"method": "POST", "path": "/v1/chat/completions", "unread": "body is not a JSON object"}
        ]
    (entry,) = _read(direction="read_requests", data=b"HELLO THERE\r\n\r\n")
    assert entry["unread"].startswith("read_requests: ValueError")
    assert _read(direction="read_requests", data=_post(path="/v1/chat/completions", body=b'{"model": "q"}')[:-3]) == []


def test_forwarder_finds_the_usage_servers_report() -> None:
    usage = {"prompt_tokens": 1200, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 1000}}
    splash = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b": splash-keepalive\n\n"
        b'data: {"choices":[],"usage":' + json.dumps(usage).encode() + b"}\n\ndata: [DONE]\n\n"
    )
    assert _read(direction="read_responses", data=splash, methods=("POST",)) == [{"usage": usage}]
    tail = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
    event = b"data: " + json.dumps(tail).encode() + b"\n\n"
    stream = (
        b"HTTP/1.1 100 Continue\r\n\r\n"
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
        + _chunk(data=b'data: {"choices":[{"delta":{}}],"usage":null}\n\n')
        + _chunk(data=event[:20])
        + _chunk(data=event[20:])
        + b"0\r\n\r\n"
    )
    head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 50\r\n\r\n"
    tokens = _reply(body=json.dumps({"tokens": [1, 2, 3]}).encode())
    whole = _reply(body=json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": usage}).encode())
    data = stream + head + tokens + whole
    entries = _read(direction="read_responses", data=data, methods=("POST", "HEAD", "POST", "POST"), piece=3)
    assert entries == [{"usage": tail["usage"]}, {"usage": usage}]


def test_forwarder_passes_every_byte_on_and_logs_who_sent_it(tmp_path: Path) -> None:
    log_file = tmp_path / "requests-11501.jsonl"
    request = _post(path="/v1/chat/completions", body=json.dumps({"model": "q", "messages": []}).encode())
    reply = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
        b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\ndata: [DONE]\n\n'
    )
    received: list[bytes] = []

    async def model(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        received.append(await reader.readexactly(len(request)))
        writer.write(reply)
        await writer.drain()
        writer.close()

    async def go() -> bytes:
        server = await asyncio.start_server(model, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        upstream = forwarder.Upstream(host="127.0.0.1", port=port, log_path=str(log_file))
        proxy = await asyncio.start_server(upstream.handle, "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.sockets[0].getsockname()[1])
        writer.write(request)
        await writer.drain()
        answer = await reader.read()
        writer.close()
        for _ in range(200):
            if log_file.exists() and len(log_file.read_text().splitlines()) == 2:
                break
            await asyncio.sleep(0.01)
        proxy.close()
        server.close()
        return answer

    assert asyncio.run(go()) == reply
    assert received == [request]
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [line["ip"] for line in lines] == ["127.0.0.1", "127.0.0.1"]
    assert lines[0]["path"] == "/v1/chat/completions" and lines[0]["params"] == {"model": "q"}
    assert lines[1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 1}
    assert all(isinstance(line["time"], float) for line in lines)


class _Proxy:
    def __init__(self, labels: dict[str, str], removed: list[str]) -> None:
        self.id = f"proxy-{len(removed)}"
        self.labels = labels
        self.status = "running"
        self.removed = removed

    def remove(self, force: bool) -> None:
        self.removed.append(self.id)


class _Containers:
    def __init__(self, existing: _Proxy | None, removed: list[str]) -> None:
        self.existing = existing
        self.removed = removed
        self.runs: list[dict] = []

    def get(self, name: str) -> _Proxy:
        if self.existing is None or self.existing.id in self.removed:
            raise pi.docker.errors.NotFound(name)
        return self.existing

    def run(
        self,
        image: str,
        command: list[str],
        name: str,
        labels: dict[str, str],
        volumes: dict[str, dict[str, str]],
        extra_hosts: dict[str, str],
        restart_policy: dict[str, str],
        detach: bool,
    ) -> _Proxy:
        self.runs.append({"image": image, "command": command, "name": name, "volumes": volumes})
        self.existing = _Proxy(labels=labels, removed=self.removed)
        return self.existing


class _Network:
    def __init__(self) -> None:
        self.containers: list[_Proxy] = []
        self.aliases: list[list[str]] = []

    def reload(self) -> None:
        pass

    def connect(self, container: _Proxy, aliases: list[str]) -> None:
        self.containers.append(container)
        self.aliases.append(aliases)


def test_sandbox_network_replaces_a_forwarder_from_other_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    removed: list[str] = []
    network = _Network()
    containers = _Containers(existing=_Proxy(labels={}, removed=removed), removed=removed)
    client = SimpleNamespace(
        networks=SimpleNamespace(get=lambda name: network),
        containers=containers,
        images=SimpleNamespace(get=lambda image: SimpleNamespace(attrs={"Id": "sha256:base"})),
    )
    endpoint = core.ModelEndpoint(model_id="q", base_url="http://host.docker.internal:11501/v1")
    assert pi.sandbox_network(client=client, endpoint=endpoint) == "rsi-sandbox-11501"
    assert removed == ["proxy-0"]
    (run,) = containers.runs
    assert run["name"] == "rsi-model-proxy-11501"
    assert run["command"] == [
        "python3",
        "/opt/rsi/forwarder.py",
        "host.docker.internal",
        "11501",
        "/logs/requests-11501.jsonl",
    ]
    assert run["volumes"] == {
        str(tmp_path / "models" / "logs"): {"bind": "/logs", "mode": "rw"},
        str(pi.FORWARDER_SCRIPT): {"bind": "/opt/rsi/forwarder.py", "mode": "ro"},
    }
    assert pi.request_log(port=11501) == tmp_path / "models" / "logs" / "requests-11501.jsonl"
    assert (tmp_path / "models" / "logs").is_dir()
    assert network.aliases == [["host.docker.internal"]]
    pi.sandbox_network(client=client, endpoint=endpoint)
    assert removed == ["proxy-0"] and len(containers.runs) == 1


def test_save_requests_keeps_the_containers_lines_since_it_started(tmp_path: Path) -> None:
    log_file = tmp_path / "requests-11501.jsonl"
    request = {"time": 2.0, "ip": "172.20.0.3", "method": "POST", "path": "/v1/chat/completions", "params": {}}
    usage = {"time": 3.0, "ip": "172.20.0.3", "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    log_file.write_text(json.dumps({**request, "time": 1.0}) + "\n")
    offset = log_file.stat().st_size
    with log_file.open("a") as fh:
        for line in (request, {**request, "ip": "172.20.0.9"}, usage):
            fh.write(json.dumps(line) + "\n")
        fh.write('{"time": 4.0, "ip": "172.20.0.3", "us')
    dest = tmp_path / "requests.jsonl"
    pi._save_requests(log_file=log_file, offset=offset, ip="172.20.0.3", dest=dest)
    assert [json.loads(line) for line in dest.read_text().splitlines()] == [request, usage]
    for log, ip in ((log_file, ""), (tmp_path / "missing.jsonl", "172.20.0.3")):
        pi._save_requests(log_file=log, offset=0, ip=ip, dest=tmp_path / "none.jsonl")
        assert not (tmp_path / "none.jsonl").exists()


def test_sandbox_ip() -> None:
    calls: list[str] = []
    networks = {"rsi-sandbox-11501": {"IPAddress": "172.20.0.3"}, "bridge": {"IPAddress": "172.17.0.2"}}
    container = SimpleNamespace(
        attrs={"NetworkSettings": {"Networks": networks}}, reload=lambda: calls.append("reload")
    )
    assert pi._sandbox_ip(container=container, network="rsi-sandbox-11501") == "172.20.0.3"
    assert calls == ["reload"]


class _TaskContainer:
    def __init__(self, out: Path, log_file: Path, sent: list[dict]) -> None:
        self.out = out
        self.log_file = log_file
        self.sent = sent
        self.short_id = "task"
        self.status = "running"
        self.attrs: dict = {}

    def reload(self) -> None:
        self.attrs = {"NetworkSettings": {"Networks": {"rsi-sandbox-11501": {"IPAddress": "172.20.0.3"}}}}

    def wait(self, timeout: int) -> dict[str, int]:
        message = {"role": "assistant", "stopReason": "stop", "usage": {"input": 30, "output": 5}}
        (self.out / "events.jsonl").write_text(json.dumps({"type": "message_end", "message": message}) + "\n")
        (self.out / "patch.diff").write_text("+fix\n")
        (self.out / pi.STAGING_FAILED).write_text("fatal: Unable to create '/testbed/.git/index.lock'\n")
        with self.log_file.open("a") as fh:
            fh.writelines(json.dumps(line) + "\n" for line in self.sent)
        self.status = "exited"
        return {"StatusCode": 0}

    def logs(self, tail: int) -> bytes:
        return b""

    def remove(self, force: bool) -> None:
        pass


class _Docker:
    def __init__(self, task: _TaskContainer) -> None:
        self.task = task
        self.proxy: _Proxy | None = None
        self.volumes: dict[str, dict[str, str]] = {}
        self.networks = SimpleNamespace(get=lambda name: _Network())
        self.images = SimpleNamespace(get=lambda image: SimpleNamespace(attrs={"Id": f"sha256:{image}"}))
        self.containers = self

    def get(self, name: str) -> _Proxy:
        if name.startswith("rsi-model-proxy-") and self.proxy is not None:
            return self.proxy
        raise pi.docker.errors.NotFound(name)

    def run(
        self,
        image: str,
        command: list[str],
        name: str,
        detach: bool,
        volumes: dict[str, dict[str, str]],
        labels: dict[str, str] | None = None,
        extra_hosts: dict[str, str] | None = None,
        restart_policy: dict[str, str] | None = None,
        environment: dict[str, str] | None = None,
        network: str = "",
        working_dir: str = "",
    ) -> _Proxy | _TaskContainer:
        if labels is not None:
            self.proxy = _Proxy(labels=labels, removed=[])
            return self.proxy
        self.volumes = volumes
        return self.task


def test_a_task_run_keeps_its_share_of_the_request_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    kwargs = {"enable_thinking": True}
    model = core.ModelSpec(
        name="q",
        source="s",
        model_id="qwen",
        port=11501,
        thinking=True,
        sampling={"temperature": 1.0},
        chat_template_kwargs=kwargs,
    )
    chat = {"model": "qwen", "stream": True, "temperature": 1, "chat_template_kwargs": kwargs}
    log_file = pi.request_log(port=11501)
    log_file.parent.mkdir(parents=True)
    log_file.write_text(json.dumps({"ip": "172.20.0.3", "method": "POST", "path": "/v1/chat/completions"}) + "\n")
    sent = [
        {"ip": "172.20.0.3", "method": "POST", "path": "/v1/chat/completions", "params": chat},
        {"ip": "172.20.0.9", "method": "POST", "path": "/v1/chat/completions", "params": {"model": "other"}},
        {"ip": "172.20.0.3", "usage": {"prompt_tokens": 30, "completion_tokens": 5}},
        {"ip": "172.20.0.3", "method": "POST", "path": "/tokenize", "params": {"content": "(2000 bytes)"}},
        {"ip": "172.20.0.3", "method": "POST", "path": "/v1/chat/completions", "params": {**chat, "temperature": 0}},
        {"ip": "172.20.0.3", "usage": {"prompt_tokens": 50, "completion_tokens": 7}},
    ]
    out = tmp_path / "runs" / "n" / "instances" / "a__b-1"
    client = _Docker(task=_TaskContainer(out=out, log_file=log_file, sent=sent))
    monkeypatch.setattr(pi.docker, "from_env", lambda: client)
    ctx = harness.RunContext(
        instance=core.Instance(instance_id="a__b-1", repo="a/b", image="img", workdir="/testbed", task="Fix it."),
        out_dir=out,
        model=model,
        endpoint=core.ModelEndpoint(
            model_id="qwen",
            base_url="http://host.docker.internal:11501/v1",
            root_url="http://host.docker.internal:11501",
        ),
        budget=core.Budget(),
    )
    with caplog.at_level(logging.WARNING):
        run = pi.HARNESS.run(ctx=ctx)
    assert run.status == "ok" and run.patch == "+fix\n"
    assert client.volumes[str(out)] == {"bind": "/out", "mode": "rw"}
    assert client.volumes[str(out / "piagent" / "extensions")] == {"bind": "/out/piagent/extensions", "mode": "ro"}
    assert client.volumes[str(out / "piagent" / "models.json")] == {"bind": "/out/piagent/models.json", "mode": "ro"}
    kept = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
    assert kept == [line for line in sent if line["ip"] == "172.20.0.3"]
    assert (run.metrics.requests, run.metrics.input_tokens, run.metrics.output_tokens) == (2, 80, 12)
    assert run.metrics.contract_violations == ["sampling {'temperature': 0}"]
    assert "git couldn't stage the changes" in caplog.text and "index.lock" in caplog.text
