import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import docker
import requests

from rsi import cards, core, folders, harness, paths

log = logging.getLogger(__name__)

PI_DIR = paths.harnesses_dir() / "pi"
# the same checks the teacher's container runs
check = folders.module(root=paths.harnesses_dir(), name="pi", file="check_variant.py", kind="pi variant check")
CHECK_SCRIPT = PI_DIR / "check_variant.py"
TRY_SCRIPT = PI_DIR / "try_variant.py"
CARDS_SCRIPT = Path(cards.__file__).resolve()
# runs the card and package checks, reporting both
CHECK_COMMAND = (
    "#!/bin/sh\n"
    'python3 /opt/rsi/cards.py "$@"; card=$?\n'
    'python3 /opt/rsi/check_variant.py --terms /opt/rsi/terms.txt "$@" || exit 1\n'
    'exit "$card"\n'
)
TRY_COMMAND = '#!/bin/sh\nexec python3 /opt/rsi/try_variant.py "$@"\n'

PI_VERSION = os.environ.get("RSI_PI_VERSION", "1.4.0")
PI_PKG = "@earendil-works/pi-coding-agent"
PI_REPO = "https://github.com/earendil-works/pi.git"
PI_BASE_IMAGE = f"rsi/pi-base:{PI_VERSION}"
PI_BASE_DOCKERFILE = (
    "FROM node:22-bookworm\n"
    f"ARG PI_VERSION={PI_VERSION}\n"
    f"RUN npm install -g --ignore-scripts {PI_PKG}@$PI_VERSION \\\n"
    "    && pi --version\n"
)
# also pi's source (the npm package is one bundled file) and pytest for test repos
TEACHER_DOCKERFILE = (
    f"FROM {PI_BASE_IMAGE}\n"
    f"RUN git clone --quiet --depth 1 --branch v{PI_VERSION} {PI_REPO} /opt/pi-src \\\n"
    "    && rm -rf /opt/pi-src/.git \\\n"
    "    && apt-get update && apt-get install -y --no-install-recommends python3-pytest \\\n"
    "    && rm -rf /var/lib/apt/lists/*\n"
)
# tagged by its Dockerfile's hash
TEACHER_IMAGE = f"rsi/pi-teacher:{PI_VERSION}-{hashlib.sha256(TEACHER_DOCKERFILE.encode()).hexdigest()[:10]}"
_live: list[docker.models.containers.Container] = []  # containers cleanup() kills on an interrupt


def instance_image(instance_id: str) -> str:
    # swebench names its amd64 images x86_64, even on arm64 hosts
    return f"sweb.eval.x86_64.{instance_id}:latest".lower()


def overlay_image(instance_id: str) -> str:
    return f"rsi/pi-swe/{instance_id}:{PI_VERSION}"


def overlay_dockerfile(instance: str) -> str:
    # COPY --from follows symlinks, so recreate pi's bin link
    return (
        f"FROM {instance}\n"
        f"COPY --from={PI_BASE_IMAGE} /usr/local/lib/node_modules /usr/local/lib/node_modules\n"
        f"COPY --from={PI_BASE_IMAGE} /usr/local/bin/node /usr/local/bin/node\n"
        f"RUN ln -s ../lib/node_modules/{PI_PKG}/dist/bundle/cli.js /usr/local/bin/pi \\\n"
        "    && pi --version\n"
    )


def build(dockerfile: str, tag: str) -> None:
    with tempfile.TemporaryDirectory(prefix="rsi-dockerfile-") as tmp:
        (Path(tmp) / "Dockerfile").write_text(dockerfile)
        # the task images are amd64, so ours are too
        subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", tag, tmp], check=True)


def _ensure_image(client: docker.DockerClient, tag: str, dockerfile: str) -> None:
    if not _image_exists(client=client, image=tag):
        build(dockerfile=dockerfile, tag=tag)


def _image_exists(client: docker.DockerClient, image: str) -> bool:
    return bool(_digest(client=client, image=image))


def _digest(client: docker.DockerClient, image: str) -> str:
    try:
        return client.images.get(image).attrs["Id"]
    except docker.errors.NotFound:
        return ""


# Containers can only reach the model: each model port gets an internal network and a forwarder that answers to the
# model URL's host name. It passes the stream through unchanged and logs every request and its usage. It runs on the pi
# image's python3 (3.11).
_FORWARDER = r"""
import asyncio, collections, json, sys, time

# request lines leave out the conversation, and log longer parameters by their size
LEFT_OUT = ("messages", "tools", "functions")
LONGEST = 1000


def request(method, path, body):
    line = {"method": method, "path": path}
    if not body:
        return line
    try:
        params = json.loads(body)
    except ValueError:
        params = None
    if not isinstance(params, dict):
        return {**line, "unread": "body is not a JSON object"}
    kept = {}
    for key, value in params.items():
        if key not in LEFT_OUT:
            size = len(json.dumps(value))
            kept[key] = value if size <= LONGEST else f"({size} bytes)"
    return {**line, "params": kept}


def usage(raw, log):
    # log the usage a response body or stream event reports
    if b'"usage"' not in raw:
        return
    try:
        data = json.loads(raw)
    except ValueError:
        return
    found = data.get("usage") if isinstance(data, dict) else None
    if isinstance(found, dict):
        log({"usage": found})


async def head(stream):
    # start line and headers, or None if the connection ends
    try:
        raw = await stream.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError as exc:
        if exc.partial.strip():
            raise
        return None
    start, *fields = raw.decode("latin-1").split("\r\n")
    headers = {}
    for field in fields:
        name, _, value = field.partition(":")
        headers[name.strip().lower()] = value.strip()
    return start.split(" ", 2), headers


async def body(stream, headers):
    # the body as it arrives: chunked, by length, or until the connection ends
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while size := int((await stream.readuntil(b"\r\n")).split(b";")[0], 16):
            yield await stream.readexactly(size)
            await stream.readexactly(2)
        while await stream.readuntil(b"\r\n") != b"\r\n":
            pass
    elif "content-length" in headers:
        yield await stream.readexactly(int(headers["content-length"]))
    else:
        while piece := await stream.read(65536):
            yield piece


async def read_requests(stream, methods, log):
    while (message := await head(stream)) is not None:
        (method, path, _), headers = message
        methods.append(method)
        if "chunked" in headers.get("transfer-encoding", "").lower():
            # the next request starts somewhere in chunks we don't read
            log({"method": method, "path": path, "unread": "chunked body"})
            return
        log(request(method, path, await stream.readexactly(int(headers.get("content-length", 0)))))


async def read_responses(stream, methods, log):
    while (message := await head(stream)) is not None:
        (_, status, *_), headers = message
        if status == "101":
            return  # the connection leaves HTTP
        if status.startswith("1"):
            continue  # an interim response, before the real one
        method = methods.popleft() if methods else ""
        if method == "HEAD" or status in ("204", "304"):
            continue
        events = headers.get("content-type", "").lower().startswith("text/event-stream")
        rest = b""
        async for piece in body(stream, headers):
            rest += piece
            if events:
                *lines, rest = rest.split(b"\n")
                for line in lines:
                    usage(line.removeprefix(b"data:"), log)
        usage(rest.removeprefix(b"data:"), log)


class Tap:
    # reads a copy of one direction of a connection without holding up its bytes
    def __init__(self, read, methods, log):
        self.stream = asyncio.StreamReader()
        self.done = asyncio.create_task(self.watch(read, methods, log))

    async def watch(self, read, methods, log):
        try:
            await read(self.stream, methods, log)
        except asyncio.IncompleteReadError:
            pass  # the connection ended inside a message
        except Exception as exc:
            log({"unread": f"{read.__name__}: {type(exc).__name__}: {exc}"})
        finally:
            self.stream = None

    def feed(self, data):
        if self.stream is not None:
            self.stream.feed_data(data)

    def end(self):
        if self.stream is not None:
            self.stream.feed_eof()


async def pipe(reader, writer, tap):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            tap.feed(data)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        tap.end()
        writer.close()


def logger(ip):
    def log(entry):
        # reopened per line, so a moved or deleted log starts again
        with open(LOG, "a") as fh:
            fh.write(json.dumps({"time": round(time.time(), 3), "ip": ip, **entry}) + "\n")

    return log


async def handle(reader, writer):
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(HOST, PORT)
    except OSError:
        writer.close()
        return
    log = logger(writer.get_extra_info("peername")[0])
    methods = collections.deque()
    sent, received = Tap(read_requests, methods, log), Tap(read_responses, methods, log)
    await asyncio.gather(pipe(reader, upstream_writer, sent), pipe(upstream_reader, writer, received))
    await asyncio.gather(sent.done, received.done)


async def main():
    server = await asyncio.start_server(handle, "0.0.0.0", PORT)
    await server.serve_forever()


if __name__ == "__main__":
    HOST, PORT, LOG = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    open(LOG, "a").close()  # there from the start, so a run can tell this forwarder from one that logs nothing
    asyncio.run(main())
"""
_PROXY_LABEL = "rsi.forwarder"
_sandbox_lock = threading.Lock()


def model_address(endpoint: core.ModelEndpoint) -> tuple[str, int]:
    url = urllib.parse.urlsplit(endpoint.base_url)
    return url.hostname or "", url.port or (443 if url.scheme == "https" else 80)


def request_log(port: int) -> Path:
    """The forwarder's request and usage log for this model port"""
    return paths.models_dir() / "logs" / f"requests-{port}.jsonl"


def sandbox_network(client: docker.DockerClient, endpoint: core.ModelEndpoint) -> str:
    """Get or create this model port's network and start its forwarder"""
    host, port = model_address(endpoint=endpoint)
    name = f"rsi-sandbox-{port}"
    proxy_name = f"rsi-model-proxy-{port}"
    logs = request_log(port=port)
    command = ["python3", "-c", _FORWARDER, host, str(port), f"/logs/{logs.name}"]
    # so a forwarder started by other code gets replaced
    label = hashlib.sha256("\0".join([*command, str(logs.parent)]).encode()).hexdigest()[:12]
    with _sandbox_lock:
        try:
            network = client.networks.get(name)
        except docker.errors.NotFound:
            network = client.networks.create(name, driver="bridge", internal=True)
        _ensure_image(client=client, tag=PI_BASE_IMAGE, dockerfile=PI_BASE_DOCKERFILE)
        try:
            proxy = client.containers.get(proxy_name)
        except docker.errors.NotFound:
            proxy = None
        if proxy is not None and proxy.labels.get(_PROXY_LABEL) != label:
            log.info("replacing %s, whose forwarder is from other code", proxy_name)
            proxy.remove(force=True)
            proxy = None
        if proxy is None:
            logs.parent.mkdir(parents=True, exist_ok=True)
            proxy = client.containers.run(
                PI_BASE_IMAGE,
                command=command,
                name=proxy_name,
                labels={_PROXY_LABEL: label},
                volumes={str(logs.parent): {"bind": "/logs", "mode": "rw"}},
                extra_hosts={"host.docker.internal": "host-gateway"},
                restart_policy={"Name": "unless-stopped"},
                detach=True,
            )
        if proxy.status != "running":
            proxy.start()
        network.reload()
        if proxy.id not in {c.id for c in network.containers}:
            network.connect(proxy, aliases=[host])
    return name


def _sandbox_ip(container: docker.models.containers.Container, network: str) -> str:
    """The container's IP on the sandbox network (how the forwarder log tells containers apart)"""
    container.reload()
    return container.attrs["NetworkSettings"]["Networks"][network]["IPAddress"]


def _save_requests(log_file: Path, offset: int, ip: str, dest: Path) -> None:
    """Copy this container's lines from the forwarder log"""
    if not ip:
        log.warning("the container's address is unknown, so its model requests weren't audited")
        return
    if not log_file.exists():
        log.warning("no request log at %s, so the run's model requests weren't audited", log_file)
        return
    kept: list[bytes] = []
    with log_file.open("rb") as fh:
        fh.seek(offset)
        for line in fh:
            try:
                if json.loads(line).get("ip") == ip:
                    kept.append(line)
            except ValueError:
                continue  # a line the forwarder is still writing
    dest.write_bytes(b"".join(kept))


def pi_models_json(model: core.ModelSpec, endpoint: core.ModelEndpoint, thinking_budgets: bool) -> dict:
    """pi's models.json (the thinking budget field only for packages that set budgets)"""
    entry: dict = {"id": model.model_id, "contextWindow": model.ctx, "maxTokens": model.output_cap}
    if model.sampling:
        entry["samplingParams"] = dict(model.sampling)
    if model.thinking:
        entry["reasoning"] = True
    compat: dict = {}
    if model.chat_template_kwargs:
        compat |= {"thinkingFormat": "chat-template", "chatTemplateKwargs": dict(model.chat_template_kwargs)}
    if thinking_budgets and model.thinking_budget_field:
        compat["thinkingTokenBudgetField"] = model.thinking_budget_field
    if compat:
        entry["compat"] = compat
    return {
        "providers": {
            "local": {"baseUrl": endpoint.base_url, "api": "openai-completions", "apiKey": "local", "models": [entry]}
        }
    }


def pi_model_ref(model: core.ModelSpec) -> str:
    """The model at pi's default thinking level, like a plain `pi --model`"""
    return f"local/{model.model_id}" if model.thinking else f"local/{model.model_id}:off"


def _install_extensions(source_dir: Path, agent_dir: Path) -> None:
    # in the agent dir, outside the variant package
    dst = agent_dir / "extensions"
    dst.mkdir(parents=True, exist_ok=True)
    for ext in sorted(source_dir.glob("*.ts")):
        (dst / ext.name).write_bytes(ext.read_bytes())


# the run's starting commit, so a timed-out run still gets the right diff
ORIG_HEAD_FILE = "/out/orig-head"
STAGING_FAILED = "staging-failed.txt"  # git's error when staging failed (a partial patch)
# every change since that commit, new and binary files included. Hooks, fsmonitor, external diff and textconv are off,
# since the agent could set them. Falls back to the tracked files' diff if git can't stage
CAPTURE_PATCH = (
    f'cd /testbed && start="$(cat {ORIG_HEAD_FILE})"\n'
    'g() { git -c core.fsmonitor=false -c core.hooksPath=/dev/null "$@"; }\n'
    'if staged="$(g add -A 2>&1)"; then g diff --cached --binary --no-ext-diff --no-textconv "$start"\n'
    f'else echo "$staged" > /out/{STAGING_FAILED}; g diff --binary --no-ext-diff --no-textconv "$start"\n'
    "fi > /out/patch.diff\n"
    "g status --porcelain > /out/status.txt || true\n"
)


def _flush_partial_patch(container: docker.models.containers.Container) -> None:
    # exec_run takes no timeout, but git diff is quick
    try:
        container.exec_run(["bash", "-lc", CAPTURE_PATCH])
    except Exception as exc:  # noqa: BLE001
        log.warning("partial patch capture failed: %s", exc)


def _remove_stale_container(client: docker.DockerClient, name: str) -> None:
    """Remove a container left by a dead run, which would block the create"""
    try:
        client.containers.get(name).remove(force=True)
    except docker.errors.NotFound:
        pass


def _finish_container(container: docker.models.containers.Container | None, log_path: Path) -> None:
    """Save the container's log and remove it"""
    if container is None:
        return
    if container in _live:
        _live.remove(container)
    try:
        log_path.write_bytes(container.logs(tail=2000))
    except Exception as exc:  # noqa: BLE001
        log.warning("container log capture failed: %s", exc)
    try:
        container.remove(force=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("container remove failed: %s", exc)


def _wait_exited(container: docker.models.containers.Container, timeout_s: float = 120.0) -> None:
    """Wait for a killed container to exit, since it can keep calling the model"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        container.reload()
        if container.status == "exited":
            return
        time.sleep(2.0)
    log.warning("container %s not exited within %ss after kill", container.short_id, timeout_s)


def _timed_out(exc: Exception) -> bool:
    """Check if a docker wait timed out (a ReadTimeout, or a ConnectionError over a unix socket)"""
    return isinstance(exc, requests.exceptions.ReadTimeout) or "Read timed out" in str(exc)


@dataclasses.dataclass
class TaskEvents:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    # input + cacheRead + cacheWrite of the largest prompt
    peak_context_tokens: int = 0
    peak_output_tokens: int = 0  # the largest single response
    capped_requests: int = 0  # responses cut at the output cap
    last_stop: str = ""  # the last response's stop reason
    tool_ms: int = 0
    tool_stats: dict[str, dict] = dataclasses.field(default_factory=dict)
    final_event: str = ""
    acted: int = 0  # times the variant logged that it acted


# what a variant appends with pi.appendEntry when it acts (named in HARNESS.md)
ACTED_ENTRY = "acted"


# stream updates, which the final message repeats
_DELTA_PREFIXES = ('{"type":"message_update"', '{"type":"tool_execution_update"')


def _records(path: Path) -> Iterator[tuple[str, dict | None]]:
    """Yield (type, record) per line, with record None for stream updates"""
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(_DELTA_PREFIXES):
                yield line[9 : line.index('"', 9)], None
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield str(record.get("type", "")), record


def _count_call(ev: TaskEvents, name: str, ms: int, error: bool) -> None:
    stat = ev.tool_stats.setdefault(name, {"count": 0, "errors": 0, "ms": 0})
    stat["count"] += 1
    stat["errors"] += int(error)
    stat["ms"] += ms
    ev.tool_ms += ms


def parse_events(events: Path, seconds: float = 0.0) -> TaskEvents:
    """Sum up a run's events. A tool call still running at the end counts until the run ended"""
    ev = TaskEvents()
    running: dict[str, str] = {}
    first_ms = asked_ms = 0
    for record_type, record in _records(path=events):
        if record_type:
            ev.final_event = record_type
        if record is None:
            continue
        if record_type == "message_end":
            message = record.get("message", {})
            first_ms = first_ms or message.get("timestamp") or 0
            if message.get("role") != "assistant":
                continue
            asked_ms = message.get("timestamp") or asked_ms
            usage = message.get("usage") or {}
            ev.requests += 1
            ev.input_tokens += usage.get("input", 0)
            ev.output_tokens += usage.get("output", 0)
            ev.cache_read_tokens += usage.get("cacheRead", 0)
            ev.reasoning_tokens += usage.get("reasoning", 0) or 0
            prompt = usage.get("input", 0) + (usage.get("cacheRead") or 0) + (usage.get("cacheWrite") or 0)
            ev.peak_context_tokens = max(ev.peak_context_tokens, prompt)
            ev.peak_output_tokens = max(ev.peak_output_tokens, usage.get("output", 0))
            ev.capped_requests += message.get("stopReason") == "length"
            ev.last_stop = str(message.get("stopReason") or "")
        elif record_type == "tool_execution_start":
            running[str(record.get("toolCallId", ""))] = record.get("toolName", "unknown")
        elif record_type == "tool_execution_end":
            running.pop(str(record.get("toolCallId", "")), None)
            _count_call(
                ev=ev,
                name=record.get("toolName", "unknown"),
                ms=record.get("durationMs", 0) or 0,
                error=bool(record.get("isError")),
            )
        elif record_type == "entry_appended":
            entry = record.get("entry") or {}
            ev.acted += entry.get("type") == "custom" and entry.get("customType") == ACTED_ENTRY
    if seconds and first_ms and asked_ms:
        for name in running.values():
            _count_call(ev=ev, name=name, ms=max(0, round(seconds * 1000) - (asked_ms - first_ms)), error=False)
    return ev


def has_model_output(events: Path) -> bool:
    """Check if the model ever answered"""
    if not events.exists():
        return False
    with events.open(encoding="utf-8", errors="replace") as fh:
        return any('"message_end"' in line and '"assistant"' in line for line in fh)


def model_error(events: Path) -> str | None:
    """The last model request's error, if it failed"""
    failed: str | None = None
    for record_type, record in _records(path=events):
        if record is None or record_type != "message_end":
            continue
        msg = record.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        failed = (msg.get("errorMessage") or "model request failed") if msg.get("stopReason") == "error" else None
    return failed


def _json_lines(path: Path) -> list[dict]:
    return [record for _, record in _records(path=path) if record is not None]


# other keys pi sends with each request
_REQUEST_KEYS = ("model", "stream", "stream_options", "store", "max_tokens", "max_completion_tokens")
_CHAT_PATH = "/chat/completions"  # pi's openai-completions endpoint
_TOKENIZE_PATH = "/tokenize"  # the metrics extension counts tokens with it


def _off_spec(params: dict, model: core.ModelSpec) -> set[str]:
    """How one request strays from the pinned model settings"""
    found: set[str] = set()
    if params.get("model", model.model_id) != model.model_id:
        found.add(f"model {params.get('model')}")
    for key in ("max_tokens", "max_completion_tokens"):
        cap = params.get(key)
        if isinstance(cap, int) and cap > model.output_cap:
            found.add(f"output cap {cap} > {model.output_cap}")
    sent = {key: params[key] for key in model.sampling if key in params}
    if sent != model.sampling:
        found.add(f"sampling {sent}")
    kwargs = params.get("chat_template_kwargs")
    if (kwargs or {}) != model.chat_template_kwargs:
        found.add(f"chat_template_kwargs {kwargs}")
    expected = {*_REQUEST_KEYS, "chat_template_kwargs", *model.sampling, model.thinking_budget_field}
    found |= {f"unexpected key {key}" for key in params.keys() - expected}
    return found


def contract_violations(requests: Path, metrics: Path, model: core.ModelSpec) -> list[str]:
    """Requests off the pinned model settings, from the forwarder log (older runs: metrics.jsonl)"""
    if not requests.exists():
        params = [line["params"] for line in _json_lines(path=metrics) if isinstance(line.get("params"), dict)]
        return sorted(set().union(*(_off_spec(params=p, model=model) for p in params)))
    found: set[str] = set()
    for line in _json_lines(path=requests):
        path = str(line.get("path", "")).split("?")[0]
        if "unread" in line:
            found.add(f"unread {path or 'traffic'}: {line['unread']}")
        elif "path" not in line or path == _TOKENIZE_PATH:
            continue
        elif not path.endswith(_CHAT_PATH):
            found.add(f"request to {path}")
        else:
            found |= _off_spec(params=line.get("params") or {}, model=model)
    return sorted(found)


@dataclasses.dataclass
class ModelCalls:
    requests: int = 0
    input_tokens: int = 0  # uncached prompt tokens, as pi counts them
    output_tokens: int = 0


def model_calls(requests: Path) -> ModelCalls:
    """Model requests and token usage the forwarder logged for one run"""
    calls = ModelCalls()
    for line in _json_lines(path=requests):
        if "path" in line and str(line["path"]).split("?")[0] != _TOKENIZE_PATH:
            calls.requests += 1
        usage = line.get("usage")
        if isinstance(usage, dict):
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            calls.input_tokens += max(0, (usage.get("prompt_tokens") or 0) - cached)
            calls.output_tokens += usage.get("completion_tokens") or 0
    return calls


def parse_breakdown(metrics: Path) -> tuple[int, int, int]:
    """System prompt, tool definition and message tokens of the largest request"""
    lines = [m for m in _json_lines(path=metrics) if "system_tokens" in m]
    largest = max(
        lines,
        key=lambda m: m.get("system_tokens", 0) + m.get("tool_def_tokens", 0) + m.get("message_tokens", 0),
        default={},
    )
    return largest.get("system_tokens", 0), largest.get("tool_def_tokens", 0), largest.get("message_tokens", 0)


def parse_timing(metrics: Path) -> tuple[float, float, float, int, int, str]:
    """ttft_first, ttft_avg, ttft_max, total generation ms, total thinking ms, final stop reason"""
    lines = [m for m in _json_lines(path=metrics) if m.get("type") == "timing"]
    if not lines:
        return 0.0, 0.0, 0.0, 0, 0, ""
    ttfts = [m.get("ttft_ms", 0) or 0 for m in lines]
    return (
        float(ttfts[0]),
        round(sum(ttfts) / len(ttfts), 1),
        float(max(ttfts)),
        int(sum(m.get("gen_ms", 0) or 0 for m in lines)),
        int(sum(m.get("think_ms", 0) or 0 for m in lines)),
        str(lines[-1].get("stop_reason", "")),
    )


# compact transcripts fit one read_file call; full ones keep everything
TRANSCRIPT_MAX_CHARS = 12_000
_KEEP_FIRST_ENTRIES = 6
FULL_OUTPUT_CHARS = 2_000  # per tool call input and output in the full transcript
TASK_GLIMPSE_CHARS = 800  # of the task message in the compact transcript
UNFINISHED = "still running when the run ended"


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _excerpt(text: str, head: int, tail: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= head + tail + 3:
        return flat
    return f"{flat[:head]} … {flat[-tail:]}"


def _tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _block(text: str, limit: int) -> str:
    """Keep multi-line text, or its start and end if it's too long"""
    text = text.strip("\n")
    if len(text) <= limit:
        return text
    head = limit // 4
    return f"{text[:head]}\n… ({len(text) - limit} characters omitted) …\n{text[-(limit - head) :]}"


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict) and c.get("type") == "text")


def _output_of(record: dict) -> str:
    result = record.get("result")
    return _text_of(content=result.get("content") if isinstance(result, dict) else result)


def _stop_note(message: dict, error_chars: int) -> str:
    stop = message.get("stopReason") or "-"
    if stop == "length":
        return "stop length (cut at the output cap)"
    if stop == "error":
        return f"stop error: {_clip(text=str(message.get('errorMessage') or ''), limit=error_chars)}"
    return "" if stop == "toolUse" else f"stop {stop}"


def _call_label(name: str, args: object) -> str:
    if not isinstance(args, dict):
        return f"{name} {_clip(text=str(args), limit=120)}"
    if name == "bash":
        lines = str(args.get("command", "")).strip().splitlines() or [""]
        more = f" (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
        return f"bash: {_clip(text=lines[0], limit=160)}{more}"
    if name == "read":
        offset = f" from line {args['offset']}" if args.get("offset") else ""
        return f"read {args.get('path', '?')}{offset}"
    if name == "edit":
        edits = args.get("edits")
        count = f" ({len(edits)} edits)" if isinstance(edits, list) and len(edits) != 1 else ""
        return f"edit {args.get('path', '?')}{count}"
    if name == "write":
        return f"write {args.get('path', '?')} ({len(str(args.get('content', '')).splitlines())} lines)"
    return f"{name} {_clip(text=json.dumps(args), limit=120)}"


def _call_outcome(name: str, record: dict | None) -> str:
    if record is None:
        return UNFINISHED
    lines = [line for line in _output_of(record=record).splitlines() if line.strip()]
    if record.get("isError"):
        return f"error: {_clip(text=' '.join(lines[-3:]), limit=240)}" if lines else "error"
    if name in ("edit", "write") or not lines:
        return "ok"
    if name == "read":
        return f"ok, {len(lines)} lines"
    return f"ok, {len(lines)} lines, last: {_clip(text=lines[-1], limit=160)}"


def _took(record: dict | None) -> str:
    """A tool call's duration, if it took at least a second"""
    ms = (record or {}).get("durationMs") or 0
    return f" ({ms / 1000:.0f} s)" if ms >= 1000 else ""


def _call_input(name: str, args: object) -> str:
    if not isinstance(args, dict):
        return _block(text=str(args), limit=FULL_OUTPUT_CHARS)
    if name == "bash":
        return "$ " + _block(text=str(args.get("command", "")), limit=FULL_OUTPUT_CHARS)
    if name == "read":
        window = {key: args[key] for key in ("offset", "limit") if args.get(key)}
        return f"{args.get('path', '?')} {json.dumps(window) if window else ''}".rstrip()
    if name == "write":
        return f"{args.get('path', '?')}\n" + _block(text=str(args.get("content", "")), limit=FULL_OUTPUT_CHARS)
    rest = {key: value for key, value in args.items() if key != "path"}
    head = f"{args['path']}\n" if "path" in args else ""
    return head + _block(text=json.dumps(rest, indent=1), limit=FULL_OUTPUT_CHARS)


@dataclasses.dataclass
class _Entry:
    lines: list[str]
    calls: Counter[str] = dataclasses.field(default_factory=Counter)  # calls and errors sum up what _fit drops
    errors: int = 0


def _turn_entry(index: int, message: dict, results: dict[str, dict], last: bool) -> _Entry:
    usage = message.get("usage") or {}
    parts = [p for p in message.get("content") or [] if isinstance(p, dict)]
    thinking = "\n".join(str(p.get("thinking", "")) for p in parts if p.get("type") == "thinking")
    text = _text_of(content=parts)
    head = f"[{index}] {_tokens(n=usage.get('output') or 0)} tok out"
    note = _stop_note(message=message, error_chars=160)
    entry = _Entry(lines=[f"{head}, {note}" if note else head])
    if thinking.strip():
        reasoning = usage.get("reasoning") or 0
        size = f" ({_tokens(n=reasoning)} tok)" if reasoning else ""
        excerpt = _excerpt(text=thinking, head=160, tail=1200) if last else _excerpt(text=thinking, head=120, tail=200)
        entry.lines.append(f"    thinking{size}: {excerpt}")
    if text.strip():
        entry.lines.append(f"    text: {_excerpt(text=text, head=240, tail=600 if last else 120)}")
    for call in (p for p in parts if p.get("type") == "toolCall"):
        name = str(call.get("name", "?"))
        record = results.get(str(call.get("id", "")))
        label = _call_label(name=name, args=call.get("arguments"))
        entry.lines.append(f"    {label} → {_call_outcome(name=name, record=record)}{_took(record=record)}")
        entry.calls[name] += 1
        entry.errors += int(bool(record and record.get("isError")))
    return entry


def _full_entry(index: int, message: dict, results: dict[str, dict]) -> _Entry:
    usage = message.get("usage") or {}
    parts = [p for p in message.get("content") or [] if isinstance(p, dict)]
    thinking = "\n".join(str(p.get("thinking", "")) for p in parts if p.get("type") == "thinking").strip()
    text = _text_of(content=parts).strip()
    head = f"## turn {index} · {_tokens(n=usage.get('output') or 0)} tok out"
    if usage.get("reasoning"):
        head += f" (thinking {_tokens(n=usage['reasoning'])})"
    note = _stop_note(message=message, error_chars=300)
    entry = _Entry(lines=["", f"{head} · {note}" if note else head])
    if thinking:
        entry.lines += ["### thinking", thinking]
    if text:
        entry.lines += ["### text", text]
    for call in (p for p in parts if p.get("type") == "toolCall"):
        name = str(call.get("name", "?"))
        record = results.get(str(call.get("id", "")))
        verdict = UNFINISHED if record is None else "error" if record.get("isError") else "ok"
        entry.lines += [
            f"### {name}",
            _call_input(name=name, args=call.get("arguments")),
            f"→ {verdict}{_took(record=record)}",
        ]
        output = _output_of(record=record or {})
        if output.strip():
            entry.lines.append(_block(text=output, limit=FULL_OUTPUT_CHARS))
    return entry


def transcript(events: Path, full: bool = False, max_chars: int = TRANSCRIPT_MAX_CHARS) -> str:
    # assistant messages (dict) and one-line notes (str), in order
    items: list[dict | str] = []
    results: dict[str, dict] = {}
    task: str | None = None
    for record_type, record in _records(path=events):
        if record is None:
            continue
        if record_type == "message_end":
            message = record.get("message") or {}
            role = message.get("role")
            body = _clip(text=_text_of(content=message.get("content")), limit=4_000 if full else 400)
            if role == "assistant":
                items.append(message)
            elif role == "user":
                if task is None:
                    task = _text_of(content=message.get("content")).strip()
                else:
                    items.append(f"user message: {body}")
            elif role not in ("toolResult", "system"):
                items.append(f"{message.get('customType') or role} message: {body}")
        elif record_type == "tool_execution_end":
            results[str(record.get("toolCallId", ""))] = record
        elif record_type == "compaction_end":
            items.append(f"context compacted ({record.get('reason', '?')})")
        elif record_type == "auto_retry_start":
            reason = _clip(text=str(record.get("errorMessage")), limit=160)
            items.append(f"request retry {record.get('attempt', '?')}: {reason}")
    last_turn = max((i for i, item in enumerate(items) if isinstance(item, dict)), default=-1)
    if last_turn < 0:
        return "(no model output)\n"
    entries: list[_Entry] = []
    turn = 0
    for i, item in enumerate(items):
        if isinstance(item, dict):
            turn += 1
            if full:
                entries.append(_full_entry(index=turn, message=item, results=results))
            else:
                entries.append(_turn_entry(index=turn, message=item, results=results, last=i == last_turn))
        else:
            entries.append(_Entry(lines=[f"    [{item}]"]))
    kept = entries if full else _fit(entries=entries, max_chars=max_chars)
    head = ""
    if task:
        head = f"## task message\n{task}\n" if full else f"task: {_clip(text=task, limit=TASK_GLIMPSE_CHARS)}\n\n"
    return head + "\n".join(line for entry in kept for line in entry.lines) + "\n"


def _fit(entries: list[_Entry], max_chars: int) -> list[_Entry]:
    """Keep the first entries and as many last ones as fit, summing up the middle"""
    size = [sum(len(line) + 1 for line in entry.lines) for entry in entries]
    if sum(size) <= max_chars:
        return entries
    budget = max_chars - sum(size[:_KEEP_FIRST_ENTRIES])
    start = len(entries)
    while start > _KEEP_FIRST_ENTRIES and budget >= size[start - 1]:
        start -= 1
        budget -= size[start]
    middle = entries[_KEEP_FIRST_ENTRIES:start]
    if not middle:
        return entries
    calls: Counter[str] = Counter()
    for entry in middle:
        calls.update(entry.calls)
    used = ", ".join(f"{name} {count}" for name, count in calls.most_common()) or "none"
    errors = sum(entry.errors for entry in middle)
    note = _Entry(lines=[f"    … {len(middle)} entries omitted (tool calls: {used}; {errors} errors) …"])
    return [*entries[:_KEEP_FIRST_ENTRIES], note, *entries[start:]]


@dataclasses.dataclass
class PiHarness:
    name: str = "pi"
    version: str = PI_VERSION
    package_dir: Path = PI_DIR / "package"
    doc: Path = PI_DIR / "HARNESS.md"  # the teacher reads it as harness.md
    # our metrics extension, which only reads requests
    eval_dir: Path = PI_DIR / "eval"
    task_rules: Path = PI_DIR / "prompts" / "task-rules.md"
    # task-rule words a variant can't use
    protocol_terms: tuple[str, ...] = ("hidden tests",)

    def prepare(self, instances: list[core.Instance]) -> None:
        client = docker.from_env()
        _ensure_image(client=client, tag=PI_BASE_IMAGE, dockerfile=PI_BASE_DOCKERFILE)
        missing: list[tuple[str, str]] = []
        for instance in instances:
            tag = overlay_image(instance_id=instance.instance_id)
            if _image_exists(client=client, image=tag):
                continue
            base = instance_image(instance_id=instance.instance_id)
            if not _image_exists(client=client, image=base):
                raise RuntimeError(f"instance image {base} not present; run `rsi images build` first")
            missing.append((base, tag))
        if not missing:
            return
        # overlay builds are mostly I/O, so run several at once
        self._build_overlays(client=client, items=missing)

    def _build_overlays(self, client: docker.DockerClient, items: list[tuple[str, str]]) -> None:
        first: RuntimeError | None = None
        try:
            # waits for every build before raising the first failure
            with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
                list(pool.map(self._build_overlay, items))
        except RuntimeError as exc:
            first = exc
        still_missing = [item for item in items if not _image_exists(client=client, image=item[1])]
        if not still_missing:
            return
        # retry once, one at a time, so a docker hiccup doesn't stop the run
        log.warning(
            "overlay builds failed (%s), retrying serially: %s",
            first if first is not None else "incomplete",
            [tag for _, tag in still_missing],
        )
        for item in still_missing:
            self._build_overlay(item=item)
        still_missing = [item for item in items if not _image_exists(client=client, image=item[1])]
        if still_missing:
            raise RuntimeError(f"overlay builds failed: {[tag for _, tag in still_missing]}")

    @staticmethod
    def _build_overlay(item: tuple[str, str]) -> None:
        base, tag = item
        try:
            build(dockerfile=overlay_dockerfile(instance=base), tag=tag)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"overlay build failed for {tag}") from exc

    def check_model(self, model: core.ModelSpec) -> str | None:
        # pi only sends chat_template_kwargs for thinking models
        if model.chat_template_kwargs and not model.thinking:
            return f"{model.name}: pi can't send chat_template_kwargs for a model with thinking off"
        return None

    def run(self, ctx: harness.RunContext) -> core.TaskRun:
        problem = self.check_model(model=ctx.model)
        if problem:
            return core.TaskRun(instance_id=ctx.instance.instance_id, status="error", error=problem)
        client = docker.from_env()
        package = ctx.variant if ctx.variant is not None else self.package_dir
        if not (package / "package.json").exists():
            return core.TaskRun(
                instance_id=ctx.instance.instance_id, status="error", error=f"variant {package} has no package.json"
            )
        # start clean, or pi resumes the old session with old files
        shutil.rmtree(ctx.out_dir, ignore_errors=True)
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        (ctx.out_dir / "piagent").mkdir(parents=True, exist_ok=True)
        models = pi_models_json(
            model=ctx.model, endpoint=ctx.endpoint, thinking_budgets=check.sets_thinking_budgets(package=package)
        )
        (ctx.out_dir / "piagent" / "models.json").write_text(json.dumps(models, indent=2))
        check.copy_agent_resources(package=package, agent_dir=ctx.out_dir / "piagent")
        _install_extensions(source_dir=self.eval_dir, agent_dir=ctx.out_dir / "piagent")
        (ctx.out_dir / "problem.txt").write_text(self.task_prompt(instance=ctx.instance))
        runtime_pkg = check.stage_runtime_package(package=package, dest=ctx.out_dir / "harness-pkg")

        tag = overlay_image(instance_id=ctx.instance.instance_id)
        if not _image_exists(client=client, image=tag):
            self.prepare(instances=[ctx.instance])

        # no --tools: variants change tools through settings or extensions
        pi_args = [
            "pi",
            "--print",
            "-e",
            "/opt/harness",
            "--model",
            pi_model_ref(model=ctx.model),
            "--session-dir",
            "/out/session",
            "--session-id",
            ctx.instance.instance_id.replace("__", "_1776_"),
            "--mode",
            "json",
        ]
        # diff against the starting HEAD, not the dataset's base commit (the image re-records every file mode)
        script = (
            "set -u\n"
            "cd /testbed\n"
            f"git rev-parse HEAD > {ORIG_HEAD_FILE}\n"
            f"cat /out/problem.txt | {' '.join(pi_args)} > /out/events.jsonl 2> /out/pi.err\n"
            f"{CAPTURE_PATCH}"
            "exit 0\n"
        )
        start = time.time()
        status = "error"
        error: str | None = None
        container = None
        container_name = f"rsi-{ctx.instance.instance_id.replace('__', '_1776_')}"
        requests_log = request_log(port=model_address(endpoint=ctx.endpoint)[1])
        offset = 0
        ip = ""
        try:
            _remove_stale_container(client=client, name=container_name)
            network = sandbox_network(client=client, endpoint=ctx.endpoint)
            # before the container starts, so none of its requests are missed
            offset = requests_log.stat().st_size if requests_log.exists() else 0
            container = client.containers.run(
                tag,
                command=["bash", "-lc", script],
                environment={
                    "PI_CODING_AGENT_DIR": "/out/piagent",
                    "RSI_METRICS_PATH": "/out/metrics.jsonl",
                    "RSI_MODEL_ROOT": ctx.endpoint.root_url,
                    "HOME": "/root",
                },
                volumes={
                    # staged without the card
                    str(runtime_pkg): {"bind": "/opt/harness", "mode": "ro"},
                    str(ctx.out_dir): {"bind": "/out", "mode": "rw"},
                    # read-only, while pi writes its own files next to them
                    str(ctx.out_dir / "piagent" / "extensions"): {"bind": "/out/piagent/extensions", "mode": "ro"},
                    str(ctx.out_dir / "piagent" / "models.json"): {"bind": "/out/piagent/models.json", "mode": "ro"},
                },
                network=network,
                working_dir="/testbed",
                name=container_name,
                detach=True,
            )
            _live.append(container)
            ip = _sandbox_ip(container=container, network=network)
            try:
                container.wait(timeout=ctx.budget.timeout_s)
                events = ctx.out_dir / "events.jsonl"
                if not has_model_output(events=events):
                    # an error, so a resume retries it
                    status = "error"
                    error = "no model output in events (pi died before the first assistant message)"
                else:
                    status = "ok"
                    failed = model_error(events=events)
                    if failed:
                        status = "error"
                        error = f"model request failed: {failed}"
            except Exception as exc:  # noqa: BLE001
                if _timed_out(exc=exc):
                    # out of time: score the diff so far
                    _flush_partial_patch(container=container)
                    status = "timeout"
                    error = f"timed out after {ctx.budget.timeout_s}s"
                else:
                    error = f"wait failed: {type(exc).__name__}: {exc}"
                container.kill()
                _wait_exited(container=container)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        _finish_container(container=container, log_path=ctx.out_dir / "run.log")
        if container is not None:
            _save_requests(log_file=requests_log, offset=offset, ip=ip, dest=ctx.out_dir / "requests.jsonl")
        if (ctx.out_dir / STAGING_FAILED).exists():
            log.warning(
                "%s: git couldn't stage the changes, so the patch only has the tracked files' diff: %s",
                ctx.instance.instance_id,
                (ctx.out_dir / STAGING_FAILED).read_text().strip()[-300:],
            )

        seconds = round(time.time() - start, 1)
        base_img = instance_image(instance_id=ctx.instance.instance_id)
        images = {tag: _digest(client=client, image=tag), base_img: _digest(client=client, image=base_img)}
        patch = (ctx.out_dir / "patch.diff").read_text() if (ctx.out_dir / "patch.diff").exists() else ""
        if status == "error" and error is None:
            err_file = ctx.out_dir / "pi.err"
            if err_file.exists():
                error = err_file.read_text()[-2000:]
        return core.TaskRun(
            instance_id=ctx.instance.instance_id,
            status=status,
            patch=patch,
            trajectory=str(ctx.out_dir / "events.jsonl"),
            seconds=seconds,
            metrics=self.task_metrics(instance_dir=ctx.out_dir, model=ctx.model, seconds=seconds, patch=patch),
            images=images,
            error=error,
        )

    @contextlib.contextmanager
    def teacher_container(
        self,
        *,
        workspace: Path,
        out_dir: Path,
        model: core.ModelSpec,
        endpoint: core.ModelEndpoint,
        name: str,
        terms: tuple[str, ...],
    ) -> Iterator[str]:
        problem = self.check_model(model=model)
        if problem:
            raise RuntimeError(problem)
        client = docker.from_env()
        _ensure_image(client=client, tag=PI_BASE_IMAGE, dockerfile=PI_BASE_DOCKERFILE)
        _ensure_image(client=client, tag=TEACHER_IMAGE, dockerfile=TEACHER_DOCKERFILE)
        (out_dir / "variants").mkdir(parents=True, exist_ok=True)
        (out_dir / "piagent").mkdir(exist_ok=True)
        # read-only mounts besides our scripts, removed with the container
        with tempfile.TemporaryDirectory(prefix=".teacher-", dir=out_dir.parent) as tmp:
            tools = Path(tmp)
            (tools / "models").mkdir()
            for file, budgets in (("plain.json", False), ("budget.json", True)):
                models = pi_models_json(model=model, endpoint=endpoint, thinking_budgets=budgets)
                (tools / "models" / file).write_text(json.dumps(models, indent=2))
            (tools / "models" / "ref.txt").write_text(pi_model_ref(model=model) + "\n")
            shutil.copy2(tools / "models" / "plain.json", out_dir / "piagent" / "models.json")
            (tools / "terms.txt").write_text("".join(f"{term}\n" for term in terms))
            runtime_pkg = check.stage_runtime_package(package=self.package_dir, dest=tools / "harness-pkg")
            script = (
                "set -eu\n"
                f"cat > /usr/local/bin/check-variant <<'EOF'\n{CHECK_COMMAND}EOF\n"
                f"cat > /usr/local/bin/try-variant <<'EOF'\n{TRY_COMMAND}EOF\n"
                "chmod +x /usr/local/bin/check-variant /usr/local/bin/try-variant\n"
                "touch /tmp/rsi-ready\n"
                "exec sleep infinity\n"
            )
            container_name = f"rsi-teacher-{name}"
            _remove_stale_container(client=client, name=container_name)
            container = client.containers.run(
                TEACHER_IMAGE,
                command=["bash", "-lc", script],
                environment={"PI_CODING_AGENT_DIR": "/out/piagent", "HOME": "/root"},
                volumes={
                    str(runtime_pkg): {"bind": "/opt/harness", "mode": "ro"},
                    str(CHECK_SCRIPT): {"bind": "/opt/rsi/check_variant.py", "mode": "ro"},
                    str(TRY_SCRIPT): {"bind": "/opt/rsi/try_variant.py", "mode": "ro"},
                    str(CARDS_SCRIPT): {"bind": "/opt/rsi/cards.py", "mode": "ro"},
                    str(self.task_rules): {"bind": "/opt/rsi/task-rules.md", "mode": "ro"},
                    str(tools / "models"): {"bind": "/opt/rsi/models", "mode": "ro"},
                    str(tools / "terms.txt"): {"bind": "/opt/rsi/terms.txt", "mode": "ro"},
                    str(workspace): {"bind": "/work", "mode": "ro"},
                    str(out_dir): {"bind": "/out", "mode": "rw"},
                },
                network=sandbox_network(client=client, endpoint=endpoint),
                working_dir="/work",
                name=container_name,
                detach=True,
            )
            _live.append(container)
            try:
                for _ in range(60):
                    if container.exec_run(["test", "-f", "/tmp/rsi-ready"]).exit_code == 0:
                        break
                    time.sleep(0.5)
                else:
                    raise RuntimeError(f"{container_name} didn't start: {container.logs()[-500:]!r}")
                yield container_name
            finally:
                container.remove(force=True)
                _live.remove(container)

    def validate_variant(self, *, variant: Path, out_dir: Path, terms: tuple[str, ...]) -> list[str]:
        """Run check-variant in a fresh container"""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "terms.txt").write_text("".join(f"{term}\n" for term in terms))
        client = docker.from_env()
        _ensure_image(client=client, tag=PI_BASE_IMAGE, dockerfile=PI_BASE_DOCKERFILE)
        container = None
        output = ""
        exit_code = -1
        try:
            _remove_stale_container(client=client, name="rsi-validate")
            container = client.containers.run(
                PI_BASE_IMAGE,
                command=["python3", "/opt/rsi/check_variant.py", "--terms", "/opt/rsi/terms.txt", "/variant"],
                environment={"HOME": "/root"},
                volumes={
                    str(variant.resolve()): {"bind": "/variant", "mode": "ro"},
                    str(CHECK_SCRIPT): {"bind": "/opt/rsi/check_variant.py", "mode": "ro"},
                    str(out_dir / "terms.txt"): {"bind": "/opt/rsi/terms.txt", "mode": "ro"},
                },
                name="rsi-validate",
                network_mode="none",
                detach=True,
            )
            _live.append(container)
            exit_code = int(container.wait(timeout=240).get("StatusCode", -1))
            output = container.logs().decode(errors="replace")
        except Exception as exc:  # noqa: BLE001
            if _timed_out(exc=exc) and container is not None:
                container.kill()
                _wait_exited(container=container)
                output = "problem: check-variant did not finish within 240 s"
            else:
                output = f"problem: docker: {type(exc).__name__}: {exc}"
        _finish_container(container=container, log_path=out_dir / "check.log")
        problems = [line.removeprefix("problem: ") for line in output.splitlines() if line.startswith("problem: ")]
        if exit_code != 0 and not problems:
            problems.append(f"check-variant exited {exit_code}: {output.strip()[-300:]}")
        return problems

    def cleanup(self) -> None:
        for container in list(_live):
            try:
                container.kill()
                _wait_exited(container=container, timeout_s=60.0)
                container.remove(force=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("cleanup failed for %s: %s", container.short_id, exc)
            if container in _live:
                _live.remove(container)

    def contract(self) -> str:
        """Hash of pi's version, the task rules, the metrics extension and the baseline package"""
        files = {"task-rules.md": self.task_rules}
        for name, root in (("eval", self.eval_dir), ("package", self.package_dir)):
            for path in root.rglob("*"):
                rel = path.relative_to(root)
                if path.is_file() and not any(part.startswith(".") for part in rel.parts):
                    files[f"{name}/{rel.as_posix()}"] = path
        digest = hashlib.sha256(self.version.encode())
        for rel in sorted(files):
            data = files[rel].read_bytes()
            digest.update(f"\0{rel}\0{len(data)}\0".encode() + data)
        return digest.hexdigest()[:12]

    def task_metrics(
        self, *, instance_dir: Path, model: core.ModelSpec, seconds: float, patch: str
    ) -> core.TaskMetrics:
        events = parse_events(events=instance_dir / "events.jsonl", seconds=seconds)
        requests = instance_dir / "requests.jsonl"
        # every model call from the forwarder log (older runs: the agent's own, from its events)
        calls = (
            model_calls(requests=requests)
            if requests.exists()
            else ModelCalls(
                requests=events.requests, input_tokens=events.input_tokens, output_tokens=events.output_tokens
            )
        )
        system_tokens, tool_def_tokens, message_tokens = parse_breakdown(metrics=instance_dir / "metrics.jsonl")
        ttft_first, ttft_avg, ttft_max, model_ms, think_ms, stop_reason = parse_timing(
            metrics=instance_dir / "metrics.jsonl"
        )
        gen_s = max(seconds - events.tool_ms / 1000.0, 1.0)
        return core.TaskMetrics(
            requests=calls.requests,
            input_tokens=calls.input_tokens,
            output_tokens=calls.output_tokens,
            cache_read_tokens=events.cache_read_tokens,
            reasoning_tokens=events.reasoning_tokens,
            tool_ms=events.tool_ms,
            tool_stats={name: core.ToolStat(**stat) for name, stat in events.tool_stats.items()},
            final_event=events.final_event,
            tok_s=round(events.output_tokens / gen_s, 2),
            context_pct=round(100.0 * events.peak_context_tokens / model.ctx, 1),
            system_tokens=system_tokens,
            tool_def_tokens=tool_def_tokens,
            message_tokens=message_tokens,
            ttft_first_ms=ttft_first,
            ttft_avg_ms=ttft_avg,
            ttft_max_ms=ttft_max,
            model_ms=model_ms,
            think_ms=think_ms,
            stop_reason=stop_reason or events.last_stop,  # metrics.ts writes nothing without the model's root URL
            patch_lines=len([line for line in patch.splitlines() if line.strip()]),
            peak_output_tokens=events.peak_output_tokens,
            capped_requests=events.capped_requests,
            acted=events.acted,
            contract_violations=contract_violations(
                requests=requests, metrics=instance_dir / "metrics.jsonl", model=model
            ),
        )

    def transcript(self, *, instance_dir: Path, full: bool) -> str:
        return transcript(events=instance_dir / "events.jsonl", full=full)

    def system_context(self, *, instance_dir: Path) -> str:
        """The system prompt and tool definitions from the first request (saved by metrics.ts)"""
        lines = _json_lines(path=instance_dir / "metrics.jsonl")
        context = next((line for line in lines if line.get("type") == "context"), None)
        if context is None:
            return ""
        tools = json.dumps(context.get("tools"), indent=1, ensure_ascii=False)
        return f"## System prompt\n\n{context.get('system', '')}\n\n## Tool definitions\n\n```json\n{tools}\n```\n"

    def task_prompt(self, instance: core.Instance) -> str:
        """The issue, then the fixed task rules"""
        return f"{instance.problem_statement.strip()}\n\n{self.task_rules.read_text()}"


HARNESS = PiHarness()
