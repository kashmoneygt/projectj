import contextlib
import dataclasses
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Protocol

import docker
import requests

from rsi import cards, core, folders, harness, paths, variants, views

log = logging.getLogger(__name__)

PI_DIR = paths.harnesses_dir() / "pi"
# questions about small repos that need no change, asked of every variant the teacher hands in. The teacher's
# container only gets the JavaScript one to practice on
QUESTIONS_DIR = PI_DIR / "questions"
PRACTICE_QUESTION = QUESTIONS_DIR / "javascript"
QUESTION_TRIES = 2  # a problem only counts if it shows up on both tries
QUESTION_TIMEOUT_S = 540  # time limit per try
# check_variant.py, which the teacher's container also runs as check-variant
check = folders.module(root=paths.harnesses_dir(), name="pi", file="check_variant.py", kind="pi variant check")
# events.py reads the files a run leaves, i.e. pi's events, the metrics.ts log and the forwarder's request log
events = folders.module(root=paths.harnesses_dir(), name="pi", file="events.py", kind="pi run events")
CHECK_SCRIPT = PI_DIR / "check_variant.py"
TRY_SCRIPT = PI_DIR / "try_variant.py"
FORWARDER_SCRIPT = PI_DIR / "forwarder.py"  # logging proxy between containers and the model server
CARDS_SCRIPT = Path(cards.__file__).resolve()
# check-variant runs both the card and package checks, so the teacher sees both reports even if one fails
CHECK_COMMAND = (
    "#!/bin/sh\n"
    'python3 /opt/rsi/cards.py "$@"; card=$?\n'
    'python3 /opt/rsi/check_variant.py --terms /opt/rsi/terms.txt --manifest /opt/harness/package.json "$@" || exit 1\n'
    'exit "$card"\n'
)
TRY_COMMAND = '#!/bin/sh\nexec python3 /opt/rsi/try_variant.py "$@"\n'

PI_VERSION = "1.5.2"
PI_PKG = "@earendil-works/pi-coding-agent"
PI_REPO = "https://github.com/earendil-works/pi.git"
PI_BASE_IMAGE = f"rsi/pi-base:{PI_VERSION}"
PI_BASE_DOCKERFILE = (
    "FROM node:22-bookworm\n"
    f"ARG PI_VERSION={PI_VERSION}\n"
    f"RUN npm install -g --ignore-scripts {PI_PKG}@$PI_VERSION \\\n"
    "    && pi --version\n"
)
# the pi image plus pi's source (the npm package is one bundled file), and pytest for the teacher's test repos
TEACHER_DOCKERFILE = (
    f"FROM {PI_BASE_IMAGE}\n"
    f"RUN git clone --quiet --depth 1 --branch v{PI_VERSION} {PI_REPO} /opt/pi-src \\\n"
    "    && rm -rf /opt/pi-src/.git \\\n"
    "    && apt-get update && apt-get install -y --no-install-recommends python3-pytest \\\n"
    "    && rm -rf /var/lib/apt/lists/*\n"
)
# tag with a hash of the Dockerfile, so changing the Dockerfile builds a new image
TEACHER_IMAGE = f"rsi/pi-teacher:{PI_VERSION}-{hashlib.sha256(TEACHER_DOCKERFILE.encode()).hexdigest()[:10]}"
_live: list[docker.models.containers.Container] = []  # containers that cleanup() kills on an interrupt
# HARNESS.md asks a variant to call pi.appendEntry("acted", ...) when it kicks in
ACTED_CALL = re.compile(r"""appendEntry\(\s*["']acted["']""")
CODE_SUFFIXES = (".ts", ".mts", ".cts", ".js", ".mjs", ".cjs")

PI_HOME = "/opt/rsi/node"  # read-only mount with pi and its node in every task container
PI_CLI = f"{PI_HOME}/lib/node_modules/{PI_PKG}/dist/bundle/cli.js"
# fill the PI_HOME volume from the pi image, and touch .ready last so a half-filled volume gets filled again
PI_FILL = (
    "set -e\n"
    "rm -rf /mnt/bin /mnt/lib /mnt/.ready\n"
    "mkdir -p /mnt/bin /mnt/lib\n"
    "cp -a /usr/local/bin/node /mnt/bin/node\n"
    "cp -a /usr/local/lib/node_modules /mnt/lib/\n"
    f"ln -s ../lib/node_modules/{PI_PKG}/dist/bundle/cli.js /mnt/bin/pi\n"
    "touch /mnt/.ready\n"
)
INTERNET = "rsi-internet"  # network a task container joins while it's allowed internet access
OFFLINE = "rsi-offline"  # an internal network with nothing else on it, for checking a grader without a model
SETUP_TIMEOUT_S = 900  # Harbor's limit for a task's setup
GIT_TIMEOUT_S = 300  # time limit for patch_start() and capture_patch()
SOLUTION_TIMEOUT_S = 3600  # time limit for the reference solution when checking a grader
SNAPSHOT = "/var/tmp/.workdir-start.git"  # the workdir as the agent got it, when it isn't the top of a git repo
_ready_volumes: set[str] = set()
_volume_lock = threading.Lock()


def build(dockerfile: str, tag: str) -> None:
    with tempfile.TemporaryDirectory(prefix="rsi-dockerfile-") as tmp:
        (Path(tmp) / "Dockerfile").write_text(dockerfile)
        # the task images are amd64, so ours are too
        subprocess.run(["docker", "build", "--platform", "linux/amd64", "-t", tag, tmp], check=True)


def pi_volume(client: docker.DockerClient) -> str:
    """Name of the volume with pi and its node that every task container mounts at PI_HOME, filled on first use"""
    _ensure_image(client=client, tag=PI_BASE_IMAGE, dockerfile=PI_BASE_DOCKERFILE)
    name = f"rsi-pi-{PI_VERSION}-{_digest(client=client, image=PI_BASE_IMAGE).removeprefix('sha256:')[:12]}"
    with _volume_lock:
        if name not in _ready_volumes:
            client.volumes.create(name=name)
            mount = {name: {"bind": "/mnt", "mode": "rw"}}
            try:
                client.containers.run(PI_BASE_IMAGE, ["test", "-f", "/mnt/.ready"], volumes=mount, remove=True)
            except docker.errors.ContainerError:
                log.info(f"filling {name} with pi {PI_VERSION} and its node")
                client.containers.run(PI_BASE_IMAGE, ["sh", "-c", PI_FILL], volumes=mount, remove=True)
            _ready_volumes.add(name)
    return name


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


_PROXY_LABEL = "rsi.forwarder"
_sandbox_lock = threading.Lock()


def model_address(endpoint: core.ModelEndpoint) -> tuple[str, int]:
    url = urllib.parse.urlsplit(endpoint.base_url)
    return url.hostname or "", url.port or (443 if url.scheme == "https" else 80)


def request_log(port: int) -> Path:
    """The forwarder's request and usage log for this model port"""
    return paths.models_dir() / "logs" / f"requests-{port}.jsonl"


def sandbox_network(client: docker.DockerClient, endpoint: core.ModelEndpoint) -> str:
    """Get or create the sandbox network for this model port and make sure its forwarder is running"""
    host, port = model_address(endpoint=endpoint)
    name = f"rsi-sandbox-{port}"
    proxy_name = f"rsi-model-proxy-{port}"
    logs = request_log(port=port)
    command = ["python3", "/opt/rsi/forwarder.py", host, str(port), f"/logs/{logs.name}"]
    # hash the command, log folder and forwarder code into a label, so a forwarder started by other code gets replaced
    code = FORWARDER_SCRIPT.read_text()
    label = hashlib.sha256("\0".join([*command, str(logs.parent), code]).encode()).hexdigest()[:12]
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
            log.info(f"replacing {proxy_name}, whose forwarder is from other code")
            proxy.remove(force=True)
            proxy = None
        if proxy is None:
            logs.parent.mkdir(parents=True, exist_ok=True)
            proxy = client.containers.run(
                PI_BASE_IMAGE,
                command=command,
                name=proxy_name,
                labels={_PROXY_LABEL: label},
                volumes={
                    str(logs.parent): {"bind": "/logs", "mode": "rw"},
                    str(FORWARDER_SCRIPT): {"bind": "/opt/rsi/forwarder.py", "mode": "ro"},
                },
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
    """The container's IP on the sandbox network, which the forwarder log uses to tell containers apart"""
    container.reload()
    return container.attrs["NetworkSettings"]["Networks"][network]["IPAddress"]


def _network(client: docker.DockerClient, name: str, internal: bool) -> str:
    with _sandbox_lock:
        try:
            client.networks.get(name)
        except docker.errors.NotFound:
            client.networks.create(name, driver="bridge", internal=internal)
    return name


def _set_internet(client: docker.DockerClient, container: docker.models.containers.Container, on: bool) -> None:
    """Join or leave the internet network, so each part of a task has the network it asks for"""
    container.reload()
    joined = INTERNET in container.attrs["NetworkSettings"]["Networks"]
    if on and not joined:
        client.networks.get(_network(client=client, name=INTERNET, internal=False)).connect(container)
    elif joined and not on:
        client.networks.get(INTERNET).disconnect(container)


def _save_requests(log_file: Path, offset: int, ip: str, dest: Path) -> None:
    """Copy the forwarder log lines for ip, starting at offset, to dest"""
    if not ip:
        log.warning("the container's address is unknown, so its model requests weren't audited")
        return
    if not log_file.exists():
        log.warning(f"no request log at {log_file}, so the run's model requests weren't audited")
        return
    kept: list[bytes] = []
    with log_file.open("rb") as fh:
        fh.seek(offset)
        for line in fh:
            try:
                if json.loads(line).get("ip") == ip:
                    kept.append(line)
            except ValueError:
                continue  # skip a line the forwarder is still writing
    dest.write_bytes(b"".join(kept))


def pi_models_json(model: core.ModelSpec, endpoint: core.ModelEndpoint) -> dict:
    entry: dict = {"id": model.model_id, "contextWindow": model.ctx, "maxTokens": model.output_cap}
    if model.sampling:
        entry["samplingParams"] = dict(model.sampling)
    if model.thinking:
        entry["reasoning"] = True
    if model.chat_template_kwargs:
        entry["compat"] = {"thinkingFormat": "chat-template", "chatTemplateKwargs": dict(model.chat_template_kwargs)}
    return {
        "providers": {
            "local": {"baseUrl": endpoint.base_url, "api": "openai-completions", "apiKey": "local", "models": [entry]}
        }
    }


def pi_model_ref(model: core.ModelSpec) -> str:
    """The model ref at pi's default thinking level (like a plain `pi --model`), or :off for a non-thinking model"""
    return f"local/{model.model_id}" if model.thinking else f"local/{model.model_id}:off"


def _write_models(dest: Path, model: core.ModelSpec, endpoint: core.ModelEndpoint) -> str:
    """Write the models.json and model ref that try-variant reads, and return the models.json text"""
    dest.mkdir(parents=True, exist_ok=True)
    models = json.dumps(pi_models_json(model=model, endpoint=endpoint), indent=2)
    (dest / "models.json").write_text(models)
    (dest / "ref.txt").write_text(pi_model_ref(model=model) + "\n")
    return models


def _install_extensions(source_dir: Path, agent_dir: Path) -> None:
    # put our extensions in pi's agent dir, so they load whatever the variant package holds
    dst = agent_dir / "extensions"
    dst.mkdir(parents=True, exist_ok=True)
    for ext in sorted(source_dir.glob("*.ts")):
        (dst / ext.name).write_bytes(ext.read_bytes())


# what the run's diff starts from, written before the agent starts
PATCH_START = "/out/patch-start"
STAGING_FAILED = "staging-failed.txt"  # git's error if staging failed, which means the patch is partial
# `g` is git with hooks and fsmonitor off, and the diffs skip external diff and textconv, since the agent could set
# any of them to run code. safe.directory lets git work in a repo of any owner
GIT = 'g() { git -c core.fsmonitor=false -c core.hooksPath=/dev/null -c safe.directory="*" "$@"; }\n'


def patch_start() -> str:
    """Record the workdir as the agent gets it, so the run's diff is taken against that"""
    return (
        f"{GIT}"
        f"command -v git > /dev/null || {{ echo none > {PATCH_START}; exit 0; }}\n"
        'if [ "$(g rev-parse --show-toplevel 2> /dev/null)" = "$PWD" ]\n'
        "then\n"
        # stage the workdir into a copy of the index and write a tree, so changes from before the agent starts
        # aren't in the diff
        '  index="$(g rev-parse --git-path index)"\n'
        '  start="$(export GIT_INDEX_FILE=/tmp/.start-index; cp "$index" "$GIT_INDEX_FILE" 2> /dev/null\n'
        '    g add -A 2> /dev/null; g write-tree || g rev-parse HEAD; rm -f "$GIT_INDEX_FILE")"\n'
        f'  echo "repo $start" > {PATCH_START}\n'
        "else\n"
        f'  export GIT_DIR={SNAPSHOT} GIT_WORK_TREE="$PWD"\n'
        "  g init -q && g add -A && g -c user.name=start -c user.email=start@localhost commit -q --allow-empty "
        f'--no-verify -m start && echo "snapshot HEAD" > {PATCH_START}\n'
        "fi\n"
    )


def capture_patch() -> str:
    """Write every change since the start, new and binary files included, without touching a repo's own index"""
    # if git can't stage, save its error and diff only the tracked files
    return (
        f"{GIT}"
        f"read -r mode start < {PATCH_START} || exit 0\n"
        'case "$mode" in\n'
        '  repo) index="$(g rev-parse --git-path index)"; export GIT_INDEX_FILE=/tmp/.start-index\n'
        '    cp "$index" "$GIT_INDEX_FILE" ;;\n'
        f'  snapshot) export GIT_DIR={SNAPSHOT} GIT_WORK_TREE="$PWD" ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
        'if staged="$(g add -A 2>&1)"; then g diff --cached --binary --no-ext-diff --no-textconv "$start"\n'
        f'else echo "$staged" > /out/{STAGING_FAILED}; g diff --binary --no-ext-diff --no-textconv "$start"\n'
        "fi > /out/patch.diff\n"
        "g status --porcelain > /out/status.txt || true\n"
        "rm -f /tmp/.start-index\n"
    )


def _only_files(folder: Path) -> None:
    """Delete symlinks and special files the container left, so the host never follows them"""
    for root, dirs, files in os.walk(folder):
        for name in dirs + files:
            path = Path(root) / name
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                path.unlink()


def _as_root(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid, info.gid, info.uname, info.gname = 0, 0, "root", "root"
    return info


def _put(container: docker.models.containers.Container, source: Path, dest: str) -> None:
    """Copy the contents of a host folder into dest in the container, owned by root"""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w", dereference=True) as tar:
        tar.add(str(source), arcname=".", filter=_as_root)
    container.exec_run(["mkdir", "-p", dest], user="root")
    if not container.put_archive(dest, data.getvalue()):
        raise RuntimeError(f"couldn't copy {source} to {dest} in {container.name}")


def _fetch(container: docker.models.containers.Container, source: str, dest: Path) -> None:
    """Copy a folder out of the container to dest"""
    stream, _ = container.get_archive(source)
    with tempfile.TemporaryDirectory(prefix="rsi-fetch-") as tmp:
        with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as tar:
            tar.extractall(tmp, filter="data")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(Path(tmp) / PurePosixPath(source).name), str(dest))


def _exec(
    container: docker.models.containers.Container,
    script: str,
    workdir: str,
    timeout_s: float,
    environment: dict[str, str] | None = None,
    login: bool = False,
) -> int | None:
    """Run a bash script in the container and return its exit code, or None if it timed out and got killed"""
    api = container.client.api
    shell = ["bash", "-lc" if login else "-c", script]
    exec_id = api.exec_create(container.id, shell, environment=environment or {}, workdir=workdir)["Id"]
    api.exec_start(exec_id, detach=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = api.exec_inspect(exec_id)
        if not state["Running"]:
            return state["ExitCode"]
        time.sleep(1.0)
    # kill everything but pid 1, which keeps the container up for grading, so nothing keeps calling the model
    container.exec_run(["bash", "-c", "kill -9 -1"], user="root")
    return None


def _start(
    client: docker.DockerClient, instance: core.Instance, network: str, volumes: dict, name: str
) -> docker.models.containers.Container:
    """Start the task's container with an idle main process, since each part of the task runs as its own exec"""
    _remove_stale_container(client=client, name=name)
    limits: dict[str, object] = {}
    if instance.cpus:
        limits["nano_cpus"] = int(instance.cpus * 1e9)
    if instance.memory_mb:
        limits["mem_limit"] = f"{instance.memory_mb}m"
    container = client.containers.create(
        instance.image,
        # pid 1 survives `kill -9 -1` (used to stop a timed-out agent), so loop the sleep to keep the container up
        command=["sh", "-c", "while :; do sleep 3600 & wait $!; done"],
        environment=dict(instance.env),
        volumes=volumes,
        network=network,
        working_dir=instance.workdir or None,
        name=name,
        **limits,
    )
    _live.append(container)
    container.start()
    container.reload()
    return container


def _verify(
    client: docker.DockerClient,
    container: docker.models.containers.Container,
    verifier: core.Verifier,
    workdir: str,
    out_dir: Path,
) -> str | None:
    """Grade the task in its container Harbor-style, and return None or why grading failed"""
    _set_internet(client=client, container=container, on=verifier.internet)
    logs = verifier.logs
    container.exec_run(["bash", "-c", f"rm -rf {logs} /tests && mkdir -p {logs} && chmod 777 {logs}"], user="root")
    _put(container=container, source=Path(verifier.tests), dest="/tests")
    start = time.monotonic()
    code = _exec(
        container=container,
        script=f"({verifier.command}) > {logs}/test-stdout.txt 2>&1",
        workdir=workdir,
        timeout_s=verifier.timeout_s,
        environment=dict(verifier.env),
    )
    # kill everything in the container first, so nothing can change /out while the host writes to it
    container.exec_run(["bash", "-c", "kill -9 -1"], user="root")
    _only_files(folder=out_dir)
    folder = out_dir / harness.VERIFIER_DIR
    _fetch(container=container, source=logs, dest=folder)
    graded = {"exit_code": code, "timed_out": code is None, "seconds": round(time.monotonic() - start, 1)}
    (out_dir / harness.GRADING_FILE).write_text(json.dumps(graded) + "\n")
    if code is None or any((folder / name).is_file() for name in verifier.results):
        return None
    # don't say what the grader checks, since the teacher sees this error
    return f"grading exited {code} without a result"


class Act(Protocol):
    """What run_task runs in the task's container (e.g. the agent), returning the run's status and error"""

    def __call__(self, *, container: docker.models.containers.Container, workdir: str) -> tuple[str, str | None]: ...


def run_task(
    client: docker.DockerClient,
    instance: core.Instance,
    out_dir: Path,
    network: str,
    volumes: dict,
    act: Act,
    name: str,
) -> tuple[str, str | None, float, str]:
    """Run setup and act in the task's container, take the diff and grade it, and return status, error, seconds, IP"""
    container = None
    status, error, seconds, ip = "error", None, 0.0, ""
    try:
        container = _start(client=client, instance=instance, network=network, volumes=volumes, name=name)
        ip = _sandbox_ip(container=container, network=network)
        workdir = instance.workdir or container.attrs["Config"]["WorkingDir"] or "/"
        _set_internet(client=client, container=container, on=instance.internet)
        if instance.setup:
            setup = f"({instance.setup}) > /out/setup.log 2>&1"
            code = _exec(container=container, script=setup, workdir=workdir, timeout_s=SETUP_TIMEOUT_S)
            if code != 0:
                log_file = out_dir / "setup.log"
                printed = log_file.read_text(errors="replace").strip()[-500:] if log_file.exists() else ""
                raise RuntimeError(f"the task's setup {'timed out' if code is None else f'exited {code}'}: {printed}")
        _exec(container=container, script=patch_start(), workdir=workdir, timeout_s=GIT_TIMEOUT_S)
        start = time.monotonic()
        status, error = act(container=container, workdir=workdir)
        seconds = round(time.monotonic() - start, 1)
        _exec(container=container, script=capture_patch(), workdir=workdir, timeout_s=GIT_TIMEOUT_S)
        if instance.verifier is not None:
            failed = _verify(
                client=client, container=container, verifier=instance.verifier, workdir=workdir, out_dir=out_dir
            )
            if failed and status != "error":
                status, error = "error", failed
    except Exception as exc:  # noqa: BLE001
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        _finish_container(container=container)
        _only_files(folder=out_dir)
    return status, error, seconds, ip


def _remove_stale_container(client: docker.DockerClient, name: str) -> None:
    """Remove a container with this name left by a dead run, since it would block creating the new one"""
    try:
        client.containers.get(name).remove(force=True)
    except docker.errors.NotFound:
        pass


def _finish_container(container: docker.models.containers.Container | None, log_path: Path | None = None) -> None:
    """Save the container's log to log_path if given, then remove the container"""
    if container is None:
        return
    if container in _live:
        _live.remove(container)
    if log_path is not None:
        try:
            log_path.write_bytes(container.logs(tail=2000))
        except Exception as exc:  # noqa: BLE001
            log.warning(f"container log capture failed: {exc}")
    try:
        container.remove(force=True)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"container remove failed: {exc}")


def _wait_exited(container: docker.models.containers.Container, timeout_s: float = 120.0) -> None:
    """Wait for a killed container to exit, since it can keep calling the model until then"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        container.reload()
        if container.status == "exited":
            return
        time.sleep(2.0)
    log.warning(f"container {container.short_id} not exited within {timeout_s}s after kill")


def _timed_out(exc: Exception) -> bool:
    """Check if a docker wait timed out (a ReadTimeout, or a ConnectionError over a unix socket)"""
    return isinstance(exc, requests.exceptions.ReadTimeout) or "Read timed out" in str(exc)


def _run_check(
    what: str,
    name: str,
    command: list[str],
    volumes: dict[str, dict[str, str]],
    network: str,
    timeout_s: int,
    log_path: Path,
) -> tuple[int, str]:
    """Run a check in a fresh pi container and return its exit code and output (no network if network is "")"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    client = docker.from_env()
    _ensure_image(client=client, tag=PI_BASE_IMAGE, dockerfile=PI_BASE_DOCKERFILE)
    # add the pid, so two rsi processes (e.g. rounds for two models) don't remove each other's containers
    name = f"{name}-{os.getpid()}"
    container = None
    output = ""
    exit_code = -1
    try:
        _remove_stale_container(client=client, name=name)
        container = client.containers.run(
            PI_BASE_IMAGE,
            command=command,
            environment={"HOME": "/root"},
            volumes=volumes,
            name=name,
            **({"network": network} if network else {"network_mode": "none"}),
            detach=True,
        )
        _live.append(container)
        exit_code = int(container.wait(timeout=timeout_s).get("StatusCode", -1))
        output = container.logs().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        if _timed_out(exc=exc) and container is not None:
            container.kill()
            _wait_exited(container=container)
            output = f"problem: {what} did not finish within {timeout_s} s"
        else:
            output = f"unavailable: docker: {type(exc).__name__}: {exc}"
    finally:
        # on ctrl-c too, since a check that's still going can keep calling the model
        _finish_container(container=container, log_path=log_path)
    unavailable = _printed(output=output, label="unavailable")
    if unavailable:
        raise harness.CheckUnavailable(f"{what} couldn't run: {unavailable[0]}")
    return exit_code, output


def _printed(output: str, label: str) -> list[str]:
    """The lines a check printed as "<label>: ..." ("problem" or "unavailable"), without the label"""
    return [line.removeprefix(f"{label}: ") for line in output.splitlines() if line.startswith(f"{label}: ")]


def _check_problems(what: str, exit_code: int, output: str) -> list[str]:
    problems = _printed(output=output, label="problem")
    if exit_code != 0 and not problems:
        problems.append(f"{what} exited {exit_code}: {output.strip()[-300:]}")
    return problems


def _terms_file(terms: tuple[str, ...]) -> str:
    # hashed, since the file is in the teacher's container and the terms name the benchmark
    return "".join(f"{check.hashed(term=term)}\n" for term in terms)


def _agent_script(ctx: harness.RunContext) -> str:
    """Script that runs pi on the task message with its own node, so the task's environment stays as it is"""
    # no --tools, since variants change tools through settings or extensions
    pi_args = [
        f"{PI_HOME}/bin/node",
        PI_CLI,
        "--print",
        "-e",
        "/opt/harness",
        "--model",
        pi_model_ref(model=ctx.model),
        "--session-dir",
        "/out/session",
        # same for every task, so the shell env and session file name don't give away the task
        "--session-id",
        "task",
        "--mode",
        "json",
    ]
    # last on PATH, so the task's own tools come first
    return (
        f'export PATH="$PATH:{PI_HOME}/bin"\n'
        f"{' '.join(pi_args)} < /out/{harness.TASK_FILE} > /out/events.jsonl 2> /out/pi.err\n"
    )


def _ended(out_dir: Path) -> tuple[str, str | None]:
    """For a run that ended on its own, return an error if pi got no model output or its last request failed"""
    events_file = out_dir / "events.jsonl"
    if not events.has_model_output(events=events_file):
        # return an error, so finishing the run with `rsi run` runs the task again
        why = "no model output in events (pi died before the first assistant message)"
        err = out_dir / "pi.err"
        died = err.read_text(errors="replace").strip()[-1500:] if err.exists() else ""
        return "error", f"{why}: {died}" if died else why
    failed = events.model_error(events=events_file)
    return ("error", f"model request failed: {failed}") if failed else ("ok", None)


@dataclasses.dataclass
class PiHarness:
    name: str = "pi"
    version: str = PI_VERSION
    package_dir: Path = PI_DIR / "package"
    doc: Path = PI_DIR / "HARNESS.md"  # the teacher reads it as harness.md
    # our extensions, i.e. metrics.ts (only reads requests) and budget.ts (stops a run at its call budget)
    eval_dir: Path = PI_DIR / "eval"
    # runner paths that a variant may not mention in files the agent reads
    protocol_terms: tuple[str, ...] = ("/opt/harness", "/opt/rsi", str(PurePosixPath(harness.REPLY_FILE).parent))

    def prepare(self, instances: list[core.Instance]) -> None:
        client = docker.from_env()
        pi_volume(client=client)
        missing = sorted({i.image for i in instances if not _image_exists(client=client, image=i.image)})
        if missing:
            raise RuntimeError(
                f"{len(missing)} task images aren't present, like {missing[0]}; run `rsi images build` first"
            )

    def check_model(self, model: core.ModelSpec) -> str | None:
        # pi only sends chat_template_kwargs for thinking models
        if model.chat_template_kwargs and not model.thinking:
            return f"{model.name}: pi can't send chat_template_kwargs for a model with thinking off"
        return None

    def run(self, ctx: harness.RunContext) -> core.TaskRun:
        problem = self.check_model(model=ctx.model)
        if problem:
            return core.TaskRun(instance_id=ctx.instance.instance_id, status="error", error=problem)
        package = ctx.variant if ctx.variant is not None else self.package_dir
        if not (package / "package.json").exists():
            return core.TaskRun(
                instance_id=ctx.instance.instance_id, status="error", error=f"variant {package} has no package.json"
            )
        runtime_pkg = self._stage(ctx=ctx, package=package)
        client = docker.from_env()
        volumes = {
            pi_volume(client=client): {"bind": PI_HOME, "mode": "ro"},
            # only the runtime files, so the agent never sees the card
            str(runtime_pkg): {"bind": "/opt/harness", "mode": "ro"},
            str(ctx.out_dir): {"bind": "/out", "mode": "rw"},
            # read-only so the agent can't change them, while pi still writes its own files next to them
            str(ctx.out_dir / "piagent" / "extensions"): {"bind": "/out/piagent/extensions", "mode": "ro"},
            str(ctx.out_dir / "piagent" / "models.json"): {"bind": "/out/piagent/models.json", "mode": "ro"},
        }
        environment = {
            "PI_CODING_AGENT_DIR": "/out/piagent",
            "RSI_METRICS_PATH": "/out/metrics.jsonl",
            "RSI_MODEL_ROOT": ctx.endpoint.root_url,
            "RSI_MAX_CALLS": str(ctx.budget.max_calls),
            "RSI_CALLS_PATH": "/out/calls",
        }

        def agent(*, container: docker.models.containers.Container, workdir: str) -> tuple[str, str | None]:
            timeout_s = ctx.budget.timeout_s
            code = _exec(
                container=container,
                script=_agent_script(ctx=ctx),
                workdir=workdir,
                timeout_s=timeout_s,
                environment=environment,
                login=ctx.instance.login,
            )
            # save the agent's final answer at REPLY_FILE, for graders that check an answer
            reply = PurePosixPath(harness.REPLY_FILE)
            with tempfile.TemporaryDirectory(prefix="rsi-reply-") as tmp:
                (Path(tmp) / reply.name).write_text(events.final_reply(events=ctx.out_dir / "events.jsonl"))
                _put(container=container, source=Path(tmp), dest=str(reply.parent))
            return ("timeout", f"timed out after {timeout_s}s") if code is None else ("ok", None)

        network = sandbox_network(client=client, endpoint=ctx.endpoint)
        requests_log = request_log(port=model_address(endpoint=ctx.endpoint)[1])
        # note the log size before the container starts, so none of its requests are missed
        offset = requests_log.stat().st_size if requests_log.exists() else 0
        status, error, seconds, ip = run_task(
            client=client,
            instance=ctx.instance,
            out_dir=ctx.out_dir,
            network=network,
            volumes=volumes,
            act=agent,
            name=f"rsi-{ctx.instance.instance_id}",
        )
        events.drop_updates(events=ctx.out_dir / "events.jsonl")
        if status == "ok":
            # check how the run ended only after the container is gone, so it can't change the files we read
            status, error = _ended(out_dir=ctx.out_dir)
        _save_requests(log_file=requests_log, offset=offset, ip=ip, dest=ctx.out_dir / "requests.jsonl")
        patch = (ctx.out_dir / "patch.diff").read_text() if (ctx.out_dir / "patch.diff").exists() else ""
        if (ctx.out_dir / STAGING_FAILED).exists():
            why = (ctx.out_dir / STAGING_FAILED).read_text().strip()[-300:]
            log.warning(
                f"{ctx.instance.instance_id}: git couldn't stage the changes, so the patch only has the tracked "
                f"files' diff: {why}"
            )
        return core.TaskRun(
            instance_id=ctx.instance.instance_id,
            status=status,
            patch=patch,
            seconds=seconds,
            metrics=events.task_metrics(
                instance_dir=ctx.out_dir,
                model=ctx.model,
                seconds=seconds,
                patch=patch,
                max_calls=ctx.budget.max_calls,
            ),
            images={
                ctx.instance.image: _digest(client=client, image=ctx.instance.image),
                PI_BASE_IMAGE: _digest(client=client, image=PI_BASE_IMAGE),
            },
            error=error,
        )

    def check_grader(self, *, instance: core.Instance, out_dir: Path, solve: bool) -> None:
        if instance.verifier is None:
            raise ValueError(f"{instance.instance_id} is graded by its benchmark from the run's diff, not in place")
        if solve and not instance.solution:
            raise ValueError(f"{instance.instance_id} has no reference solution")
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        client = docker.from_env()

        def solution(*, container: docker.models.containers.Container, workdir: str) -> tuple[str, str | None]:
            if not solve:
                return "ok", None
            _put(container=container, source=Path(instance.solution), dest="/solution")
            script = "chmod +x /solution/solve.sh && /solution/solve.sh > /out/solution.txt 2>&1"
            code = _exec(container=container, script=script, workdir=workdir, timeout_s=SOLUTION_TIMEOUT_S)
            return ("ok", None) if code == 0 else ("error", f"the reference solution exited {code}")

        status, error, seconds, _ = run_task(
            client=client,
            instance=instance,
            out_dir=out_dir,
            network=_network(client=client, name=OFFLINE, internal=True),
            volumes={str(out_dir): {"bind": "/out", "mode": "rw"}},
            act=solution,
            name=f"rsi-check-{instance.instance_id}",
        )
        record = {"solve": solve, "status": status, "error": error, "seconds": seconds}
        (out_dir / "check.json").write_text(json.dumps(record, indent=2) + "\n")

    def _stage(self, ctx: harness.RunContext, package: Path) -> Path:
        """Set up the run's folder with pi's agent dir, our extensions, the task message and the staged package"""
        # start clean, or pi resumes the old session with old files
        shutil.rmtree(ctx.out_dir, ignore_errors=True)
        agent = ctx.out_dir / "piagent"
        agent.mkdir(parents=True, exist_ok=True)
        models = pi_models_json(model=ctx.model, endpoint=ctx.endpoint)
        (agent / "models.json").write_text(json.dumps(models, indent=2))
        check.copy_agent_resources(package=package, agent_dir=agent)
        _install_extensions(source_dir=self.eval_dir, agent_dir=agent)
        (ctx.out_dir / harness.TASK_FILE).write_text(ctx.instance.task)
        return check.stage_runtime_package(package=package, dest=ctx.out_dir / "harness-pkg")

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
        # files mounted read-only next to our scripts, deleted after the container is removed
        with tempfile.TemporaryDirectory(prefix=".teacher-", dir=out_dir.parent) as tmp:
            tools = Path(tmp)
            models = _write_models(dest=tools / "models", model=model, endpoint=endpoint)
            (out_dir / "piagent" / "models.json").write_text(models)
            (tools / "terms.txt").write_text(_terms_file(terms=terms))
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
                    str(tools / "models"): {"bind": "/opt/rsi/models", "mode": "ro"},
                    str(tools / "terms.txt"): {"bind": "/opt/rsi/terms.txt", "mode": "ro"},
                    str(PRACTICE_QUESTION): {"bind": "/opt/rsi/question", "mode": "ro"},
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
        (out_dir / "terms.txt").write_text(_terms_file(terms=terms))
        command = ["python3", "/opt/rsi/check_variant.py", "--terms", "/opt/rsi/terms.txt"]
        exit_code, output = _run_check(
            what="check-variant",
            name="rsi-validate",
            command=[*command, "--manifest", "/opt/rsi/package.json", "/variant"],
            volumes={
                str(variant.resolve()): {"bind": "/variant", "mode": "ro"},
                str(CHECK_SCRIPT): {"bind": "/opt/rsi/check_variant.py", "mode": "ro"},
                str(out_dir / "terms.txt"): {"bind": "/opt/rsi/terms.txt", "mode": "ro"},
                str(self.package_dir / "package.json"): {"bind": "/opt/rsi/package.json", "mode": "ro"},
            },
            network="",
            timeout_s=240,
            log_path=out_dir / "check.log",
        )
        return _check_problems(what="check-variant", exit_code=exit_code, output=output)

    def question_problems(
        self, *, variant: Path, out_dir: Path, model: core.ModelSpec, endpoint: core.ModelEndpoint
    ) -> list[str]:
        """Ask every question with try-variant, each in a fresh container"""
        problem = self.check_model(model=model)
        if problem:
            raise harness.CheckUnavailable(problem)
        _write_models(dest=out_dir / "models", model=model, endpoint=endpoint)
        network = sandbox_network(client=docker.from_env(), endpoint=endpoint)
        for question in sorted(d for d in QUESTIONS_DIR.iterdir() if d.is_dir()):
            what = f"the {question.name} question"
            exit_code, output = _run_check(
                what=what,
                name="rsi-question",
                command=[
                    "python3",
                    "/opt/rsi/try_variant.py",
                    "--question",
                    f"--tries={QUESTION_TRIES}",
                    f"--timeout={QUESTION_TIMEOUT_S}",
                    "/variant",
                ],
                volumes={
                    str(variant.resolve()): {"bind": "/variant", "mode": "ro"},
                    str(TRY_SCRIPT): {"bind": "/opt/rsi/try_variant.py", "mode": "ro"},
                    str(CHECK_SCRIPT): {"bind": "/opt/rsi/check_variant.py", "mode": "ro"},
                    str(out_dir / "models"): {"bind": "/opt/rsi/models", "mode": "ro"},
                    str(question): {"bind": "/opt/rsi/question", "mode": "ro"},
                    str(out_dir / question.name): {"bind": "/out", "mode": "rw"},
                },
                network=network,
                # each try also needs time to stage and stop pi, and up to 60 s for a plain model request
                timeout_s=QUESTION_TRIES * (QUESTION_TIMEOUT_S + 120) + 120,
                log_path=out_dir / f"{question.name}.log",
            )
            found = _check_problems(what=what, exit_code=exit_code, output=output)
            if found:
                return [f"{question.name} question: {p}" for p in found]
        return []

    def cleanup(self) -> None:
        for container in list(_live):
            try:
                container.kill()
                _wait_exited(container=container, timeout_s=60.0)
                container.remove(force=True)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"cleanup failed for {container.short_id}: {exc}")
            if container in _live:
                _live.remove(container)

    def contract(self) -> str:
        """Hash of pi's version, our extensions in eval/ and the baseline package"""
        files: dict[str, Path] = {}
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

    def records_activations(self, *, package: Path) -> bool:
        return any(
            path.suffix in CODE_SUFFIXES and ACTED_CALL.search(path.read_text(errors="replace"))
            for path in variants.package_files(root=package).values()
        )

    def run_files(self, *, instance_dir: Path) -> dict[str, Path]:
        # pi's session files (from --session-dir) and the metrics.ts log of each request
        sessions = sorted((instance_dir / "session").glob("*.jsonl"))
        files = {f"session-{k}.jsonl" if k > 1 else "session.jsonl": path for k, path in enumerate(sessions, start=1)}
        if (instance_dir / "metrics.jsonl").is_file():
            files["metrics.jsonl"] = instance_dir / "metrics.jsonl"
        return files

    def system_context(self, *, instance_dir: Path) -> str:
        return events.system_context(metrics=instance_dir / "metrics.jsonl")

    def trajectory(self, *, instance_dir: Path) -> views.Trajectory:
        return events.trajectory(events=instance_dir / "events.jsonl", metrics=instance_dir / "metrics.jsonl")


HARNESS = PiHarness()
