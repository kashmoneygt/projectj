import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import docker
import requests

from rsi import cards, core, folders, harness, paths, variants, views

log = logging.getLogger(__name__)

PI_DIR = paths.harnesses_dir() / "pi"
# questions about small repos that need no change, which every variant is asked when it's handed in. The teacher
# practices on the JavaScript one, and the others stay out of its container
QUESTIONS_DIR = PI_DIR / "questions"
PRACTICE_QUESTION = QUESTIONS_DIR / "javascript"
QUESTION_TRIES = 2  # a problem has to show up on both
QUESTION_TIMEOUT_S = 540  # each try's limit
# the same checks the teacher's container runs
check = folders.module(root=paths.harnesses_dir(), name="pi", file="check_variant.py", kind="pi variant check")
# reads what a run left behind: pi's events, the metrics extension's lines and the forwarder's log
events = folders.module(root=paths.harnesses_dir(), name="pi", file="events.py", kind="pi run events")
CHECK_SCRIPT = PI_DIR / "check_variant.py"
TRY_SCRIPT = PI_DIR / "try_variant.py"
FORWARDER_SCRIPT = PI_DIR / "forwarder.py"  # how containers reach the model, logging every request
CARDS_SCRIPT = Path(cards.__file__).resolve()
# runs the card and package checks, reporting both
CHECK_COMMAND = (
    "#!/bin/sh\n"
    'python3 /opt/rsi/cards.py "$@"; card=$?\n'
    'python3 /opt/rsi/check_variant.py --terms /opt/rsi/terms.txt --manifest /opt/harness/package.json "$@" || exit 1\n'
    'exit "$card"\n'
)
TRY_COMMAND = '#!/bin/sh\nexec python3 /opt/rsi/try_variant.py "$@"\n'

PI_VERSION = "1.4.0"
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
# HARNESS.md asks a variant to call pi.appendEntry("acted", ...) when it kicks in
ACTED_CALL = re.compile(r"""appendEntry\(\s*["']acted["']""")
CODE_SUFFIXES = (".ts", ".mts", ".cts", ".js", ".mjs", ".cjs")


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
    command = ["python3", "/opt/rsi/forwarder.py", host, str(port), f"/logs/{logs.name}"]
    # so a forwarder started by other code gets replaced
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
    """The container's IP on the sandbox network (how the forwarder log tells containers apart)"""
    container.reload()
    return container.attrs["NetworkSettings"]["Networks"][network]["IPAddress"]


def _save_requests(log_file: Path, offset: int, ip: str, dest: Path) -> None:
    """Copy this container's lines from the forwarder log"""
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
                continue  # a line the forwarder is still writing
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
    """The model at pi's default thinking level, like a plain `pi --model`"""
    return f"local/{model.model_id}" if model.thinking else f"local/{model.model_id}:off"


def _write_models(dest: Path, model: core.ModelSpec, endpoint: core.ModelEndpoint) -> str:
    """Write the models.json and model ref that try-variant reads, and return the models.json text"""
    dest.mkdir(parents=True, exist_ok=True)
    models = json.dumps(pi_models_json(model=model, endpoint=endpoint), indent=2)
    (dest / "models.json").write_text(models)
    (dest / "ref.txt").write_text(pi_model_ref(model=model) + "\n")
    return models


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
def capture_patch(workdir: str) -> str:
    return (
        f'cd {workdir} && start="$(cat {ORIG_HEAD_FILE})"\n'
        'g() { git -c core.fsmonitor=false -c core.hooksPath=/dev/null "$@"; }\n'
        'if staged="$(g add -A 2>&1)"; then g diff --cached --binary --no-ext-diff --no-textconv "$start"\n'
        f'else echo "$staged" > /out/{STAGING_FAILED}; g diff --binary --no-ext-diff --no-textconv "$start"\n'
        "fi > /out/patch.diff\n"
        "g status --porcelain > /out/status.txt || true\n"
    )


def _flush_partial_patch(container: docker.models.containers.Container, workdir: str) -> None:
    # exec_run takes no timeout, but git diff is quick
    try:
        container.exec_run(["bash", "-lc", capture_patch(workdir=workdir)])
    except Exception as exc:  # noqa: BLE001
        log.warning(f"partial patch capture failed: {exc}")


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
        log.warning(f"container log capture failed: {exc}")
    try:
        container.remove(force=True)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"container remove failed: {exc}")


def _wait_exited(container: docker.models.containers.Container, timeout_s: float = 120.0) -> None:
    """Wait for a killed container to exit, since it can keep calling the model"""
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
    """Run a check in a fresh pi container: its exit code and what it printed (no network if network is "")"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    client = docker.from_env()
    _ensure_image(client=client, tag=PI_BASE_IMAGE, dockerfile=PI_BASE_DOCKERFILE)
    # one per process, so rounds for two models can't remove each other's
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
    """The lines a check printed under a label ("problem" or "unavailable")"""
    return [line.removeprefix(f"{label}: ") for line in output.splitlines() if line.startswith(f"{label}: ")]


def _check_problems(what: str, exit_code: int, output: str) -> list[str]:
    problems = _printed(output=output, label="problem")
    if exit_code != 0 and not problems:
        problems.append(f"{what} exited {exit_code}: {output.strip()[-300:]}")
    return problems


def _task_script(ctx: harness.RunContext) -> str:
    """What the task's container runs: pi on the task message, then the patch capture"""
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
        # same for every task, so the shell env and session file name don't give away the task
        "--session-id",
        "task",
        "--mode",
        "json",
    ]
    # diff against the starting HEAD, not the dataset's base commit (the image re-records every file mode)
    return (
        "set -u\n"
        f"cd {ctx.instance.workdir}\n"
        f"git rev-parse HEAD > {ORIG_HEAD_FILE}\n"
        f"cat /out/{harness.TASK_FILE} | {' '.join(pi_args)} > /out/events.jsonl 2> /out/pi.err\n"
        f"{capture_patch(workdir=ctx.instance.workdir)}"
        "exit 0\n"
    )


def _run_container(
    ctx: harness.RunContext, client: docker.DockerClient, tag: str, runtime_pkg: Path
) -> tuple[str, str | None]:
    """Run the task's container to the end or the time limit: its status and error"""
    status, error = "error", None
    container = None
    name = f"rsi-{ctx.instance.instance_id}"
    requests_log = request_log(port=model_address(endpoint=ctx.endpoint)[1])
    offset = 0
    ip = ""
    try:
        _remove_stale_container(client=client, name=name)
        network = sandbox_network(client=client, endpoint=ctx.endpoint)
        # before the container starts, so none of its requests are missed
        offset = requests_log.stat().st_size if requests_log.exists() else 0
        container = client.containers.run(
            tag,
            command=["bash", "-lc", _task_script(ctx=ctx)],
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
            working_dir=ctx.instance.workdir,
            name=name,
            detach=True,
        )
        _live.append(container)
        ip = _sandbox_ip(container=container, network=network)
        try:
            container.wait(timeout=ctx.budget.timeout_s)
            status, error = _ended(out_dir=ctx.out_dir)
        except Exception as exc:  # noqa: BLE001
            if _timed_out(exc=exc):
                # out of time: score the diff so far
                _flush_partial_patch(container=container, workdir=ctx.instance.workdir)
                status, error = "timeout", f"timed out after {ctx.budget.timeout_s}s"
            else:
                error = f"wait failed: {type(exc).__name__}: {exc}"
            container.kill()
            _wait_exited(container=container)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    _finish_container(container=container, log_path=ctx.out_dir / "run.log")
    if container is not None:
        _save_requests(log_file=requests_log, offset=offset, ip=ip, dest=ctx.out_dir / "requests.jsonl")
    return status, error


def _ended(out_dir: Path) -> tuple[str, str | None]:
    """How a run that ended on its own went: ok, or an error if pi never got an answer or its last request failed"""
    events_file = out_dir / "events.jsonl"
    if not events.has_model_output(events=events_file):
        # an error, so finishing the run tries the task again
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
    # our metrics extension, which only reads requests
    eval_dir: Path = PI_DIR / "eval"
    # the runner's paths, which a variant can't put in front of the agent
    protocol_terms: tuple[str, ...] = ("/opt/harness", "/opt/rsi")

    def prepare(self, instances: list[core.Instance]) -> None:
        client = docker.from_env()
        _ensure_image(client=client, tag=PI_BASE_IMAGE, dockerfile=PI_BASE_DOCKERFILE)
        missing: list[tuple[str, str]] = []
        for instance in instances:
            tag = overlay_image(instance_id=instance.instance_id)
            if _image_exists(client=client, image=tag):
                continue
            if not _image_exists(client=client, image=instance.image):
                raise RuntimeError(f"task image {instance.image} not present; run `rsi images build` first")
            missing.append((instance.image, tag))
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
        failed = first if first is not None else "incomplete"
        log.warning(f"overlay builds failed ({failed}), retrying serially: {[tag for _, tag in still_missing]}")
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
        package = ctx.variant if ctx.variant is not None else self.package_dir
        if not (package / "package.json").exists():
            return core.TaskRun(
                instance_id=ctx.instance.instance_id, status="error", error=f"variant {package} has no package.json"
            )
        runtime_pkg = self._stage(ctx=ctx, package=package)
        client = docker.from_env()
        tag = overlay_image(instance_id=ctx.instance.instance_id)
        start = time.time()
        status, error = _run_container(ctx=ctx, client=client, tag=tag, runtime_pkg=runtime_pkg)
        seconds = round(time.time() - start, 1)
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
            metrics=events.task_metrics(instance_dir=ctx.out_dir, model=ctx.model, seconds=seconds, patch=patch),
            images={
                tag: _digest(client=client, image=tag),
                ctx.instance.image: _digest(client=client, image=ctx.instance.image),
            },
            error=error,
        )

    def _stage(self, ctx: harness.RunContext, package: Path) -> Path:
        """Set up the run's folder: pi's agent dir, our extensions, the task message and the staged package"""
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
        # read-only mounts besides our scripts, removed with the container
        with tempfile.TemporaryDirectory(prefix=".teacher-", dir=out_dir.parent) as tmp:
            tools = Path(tmp)
            models = _write_models(dest=tools / "models", model=model, endpoint=endpoint)
            (out_dir / "piagent" / "models.json").write_text(models)
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
        (out_dir / "terms.txt").write_text("".join(f"{term}\n" for term in terms))
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
                # each try can also spend a minute asking the model a plain request, besides staging and stopping pi
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
        """Hash of pi's version, the metrics extension and the baseline package"""
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

    def transcript(self, *, instance_dir: Path, full: bool) -> str:
        return events.transcript(events=instance_dir / "events.jsonl", full=full)

    def system_context(self, *, instance_dir: Path) -> str:
        return events.system_context(metrics=instance_dir / "metrics.jsonl")

    def trajectory(self, *, instance_dir: Path) -> views.Trajectory:
        return events.trajectory(events=instance_dir / "events.jsonl", metrics=instance_dir / "metrics.jsonl")


HARNESS = PiHarness()
