import atexit
import contextlib
import dataclasses
import hashlib
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import docker
import huggingface_hub
import requests

from jym import core, paths

LLAMA_IMAGES = {
    "gpu": "ghcr.io/ggml-org/llama.cpp@sha256:0192ab2545efcbe79c240645e34abd8fffbe4813aedcef5a0e3a886ef6d6d82f",
    "cpu": "ghcr.io/ggml-org/llama.cpp@sha256:6257697a7f5d034b8fb499ddb07af3e250506352f94102054252a23f3b85e0af",
}
# load the whole model into memory and cache prompt starts in RAM
ARGUMENTS = ["-c", "2048", "-np", "1", "--no-webui", "--load-mode", "none", "--cache-ram", "1024"]
READY_SECONDS = 300  # CUDA can take 3 minutes to start in a new container on WSL
TOP_LOGPROBS = 1000
SAMPLING = {"n_predict": 1, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0}
# CPU servers take turns instead of sharing the cores
CPU_TURN = threading.Lock()
# one thread starts servers at a time
STARTING = threading.Lock()
DIGESTS: dict[tuple, str] = {}
CHILDREN: dict[tuple[str, str], tuple[subprocess.Popen, str]] = {}  # child-process servers and their URLs


@dataclasses.dataclass(frozen=True)
class Gguf:
    name: str  # names its server on each device
    file: str  # path in the repo, or a local path
    repo: str | None = None
    revision: str | None = None

    @property
    def path(self) -> Path:
        return paths.models() / self.repo / self.file if self.repo else Path(self.file)

    def identity(self) -> str:
        """Return the model's name in runs"""
        if self.repo:
            return f"{self.repo}@{self.revision[:8]}/{self.file}"
        return f"{self.path.name}@{digest(self.path)[:12]}"

    def fetch(self) -> Path:
        if self.repo and not self.path.is_file():
            huggingface_hub.hf_hub_download(
                self.repo, self.file, revision=self.revision, local_dir=paths.models() / self.repo
            )
        return self.path


def turn(device: str) -> contextlib.AbstractContextManager:
    return CPU_TURN if device == "cpu" else contextlib.nullcontext()


def arguments(device: str) -> list[str]:
    if device == "gpu":
        return [*ARGUMENTS, "-ngl", "99"]
    return [*ARGUMENTS, "-ngl", "0", "-t", os.environ.get("LLAMA_THREADS", str(os.cpu_count()))]


def serve(model: Gguf, device: str) -> str:
    binary = os.environ.get("LLAMA_SERVER")
    return child(binary, model, device) if binary else container(model, device)


def container(model: Gguf, device: str) -> str:
    """Serve the model in a local-only Docker container, reusing a running one"""
    client = docker.from_env()
    name = "jym-" + re.sub(r"[^A-Za-z0-9_.-]", "-", f"{model.name}-{device}")
    identity = model.identity()
    server = None
    for old in client.containers.list(all=True, filters={"name": f"^/?{name}$"}):
        if old.status == "running" and old.labels.get("jym.model") == identity:
            server = old
        else:
            old.remove(force=True)
    if server is None:
        path = model.fetch()
        server = client.containers.run(
            LLAMA_IMAGES[device],
            name=name,
            labels={"jym.model": identity},
            entrypoint="/app/llama-server",
            command=["-m", f"/models/{path.name}", "--host", "0.0.0.0", "--port", "8080", *arguments(device)],
            ports={"8080/tcp": ("127.0.0.1", None)},  # any free port, read back below
            volumes={str(path.parent): {"bind": "/models", "mode": "ro"}},
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if device == "gpu" else None,
            mem_limit="8g",
            oom_score_adj=1000,
            read_only=True,
            tmpfs={"/tmp": "size=16m"},
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            restart_policy={"Name": "unless-stopped"},
            detach=True,
        )
    server.reload()
    url = f"http://127.0.0.1:{server.ports['8080/tcp'][0]['HostPort']}"
    wait(url=url, hint=f"see docker logs {name}")
    return url


def child(binary: str, model: Gguf, device: str) -> str:
    running = CHILDREN.get((model.name, device))
    if running and running[0].poll() is None:
        return running[1]
    path = model.fetch()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [binary, "-m", str(path), "--host", "127.0.0.1", "--port", str(port), *arguments(device)]
    )
    atexit.register(process.terminate)
    url = f"http://127.0.0.1:{port}"
    CHILDREN[model.name, device] = process, url
    wait(url=url, hint=f"see the output of {binary}", process=process)
    return url


def wait(url: str, hint: str, process: subprocess.Popen | None = None) -> None:
    deadline = time.monotonic() + READY_SECONDS
    while time.monotonic() < deadline and (process is None or process.poll() is None):
        try:
            if requests.get(url + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise core.Unavailable(f"llama.cpp did not become ready at {url}; {hint}")


def digest(path: Path) -> str:
    """Return a file's SHA-256, cached by size and modified time"""
    stat = path.stat()
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key not in DIGESTS:
        with open(path, "rb") as file:
            DIGESTS[key] = hashlib.file_digest(file, "sha256").hexdigest()
    return DIGESTS[key]


class LlamaServer:
    def __init__(self, model: Gguf, device: str, session: requests.Session | None = None):
        self.model, self.device = model, device
        self.session = session or requests.Session()
        self.url = ""
        self.props: dict = {}
        self.model_name = model.identity()

    def start(self) -> None:
        with STARTING:
            self.url = serve(self.model, self.device)

    def ensure(self) -> None:
        self.start()
        self.props = self.call(method="GET", path="/props")

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        try:
            response = self.session.request(method, self.url + path, json=body, timeout=120)
        except requests.ConnectionError:  # stopped, or restarted by Docker on another port
            self.start()
            response = self.session.request(method, self.url + path, json=body, timeout=120)
        response.raise_for_status()
        return response.json()

    def tokenize(self, text: str) -> list[int]:
        return self.call(
            method="POST", path="/tokenize", body={"content": text, "add_special": False, "parse_special": True}
        )["tokens"]

    def next_token_logprobs(self, prompt: list[int], candidates: list[int]) -> tuple[list[float], dict]:
        """Return each candidate's log-probability as the next token, and the server's timings"""
        body = {**SAMPLING, "n_probs": TOP_LOGPROBS, "cache_prompt": True, "prompt": prompt}
        raw = self.call(method="POST", path="/completion", body=body)
        if not raw.get("completion_probabilities"):
            # a fully cached prompt can return no probabilities, so retry without the cache
            raw = self.call(method="POST", path="/completion", body={**body, "cache_prompt": False})
        if not raw.get("completion_probabilities"):
            raise core.InvalidAnswer(f"llama.cpp replied without probabilities: {sorted(raw)} {raw.get('error')}")
        top = {item["id"]: item["logprob"] for item in raw["completion_probabilities"][0]["top_logprobs"]}
        if not any(token in top for token in candidates):
            raise core.InvalidAnswer(f"option tokens {candidates} are outside the top {TOP_LOGPROBS} tokens")
        # a confident model can push the other options out of the top tokens
        lowest = min(top.values()) - 5.0
        return [top.get(token, lowest) for token in candidates], raw.get("timings") or {}
