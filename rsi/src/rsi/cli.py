import concurrent.futures
import contextlib
import fcntl
import logging
import os
import platform
import re
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from rsi import benchmarks, cards, core, generation, harness, models, paths, runs, stats, tree, variants, viewer

app = typer.Typer(help="Improve a local model's coding-agent harness on a benchmark", no_args_is_help=True)
models_app = typer.Typer(help="Start, stop and list the model servers in models.toml", no_args_is_help=True)
images_app = typer.Typer(help="Build or remove the Docker images tasks run in", no_args_is_help=True)
variants_app = typer.Typer(help="Generate, compare and check variants", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(images_app, name="images")
app.add_typer(variants_app, name="variants")

log = logging.getLogger(__name__)


class _Elapsed(logging.Filter):
    start = time.time()

    def filter(self, record: logging.LogRecord) -> bool:
        record.elapsed = time.time() - self.start
        return True


_FMT = logging.Formatter(
    "[%(asctime)s +%(elapsed)7.1f s] %(name)s:%(funcName)s:%(lineno)d %(message)s", datefmt="%H:%M:%S"
)
_node_log: logging.FileHandler | None = None


def _file_handler(path: Path) -> logging.FileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a")
    handler.addFilter(_Elapsed())
    handler.setFormatter(_FMT)
    return handler


@app.callback()
def _setup_logging() -> None:
    """Log each command to the console and to its own file under logs/"""
    stream = logging.StreamHandler()
    stream.addFilter(_Elapsed())
    stream.setFormatter(_FMT)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stream)
    root.setLevel(logging.INFO)
    cmd = re.sub(r"[^A-Za-z0-9_-]", "-", sys.argv[1]) if len(sys.argv) > 1 else "rsi"
    path = paths.logs_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{cmd}.log"
    root.addHandler(_file_handler(path=path))
    log.info(f"=== rsi {' '.join(sys.argv[1:])} ===")
    log.info(f"log file: {path}")


def _retarget_logs(node_name: str) -> None:
    """Also log to runs/<node>/rsi.log"""
    global _node_log
    root = logging.getLogger()
    if _node_log is not None:
        root.removeHandler(_node_log)
        _node_log.close()
    _node_log = _file_handler(path=paths.node_dir(name=node_name) / "rsi.log")
    root.addHandler(_node_log)
    log.info(f"=== rsi {' '.join(sys.argv[1:])} ===")


def _stop(reason: str) -> NoReturn:
    """Log why we're stopping and exit"""
    log.error(reason, stacklevel=2)
    raise typer.Exit(code=1)


def _require_node(name: str) -> core.Node:
    try:
        return runs.read_node(name=name)
    except FileNotFoundError:
        runs_dir = paths.runs_dir()
        known = sorted(p.name for p in runs_dir.iterdir() if (p / "node.json").exists()) if runs_dir.is_dir() else []
        _stop(reason=f"node {name!r} not found; known nodes: {', '.join(known) or 'none'}")


def _variant_parent_node(variant: str) -> str:
    """The node a variant was made from, or "" for the baseline or a path"""
    if not variants.is_name(reference=variant):
        return ""
    name = variants.variant_name(reference=variant)
    if name == "baseline":
        return ""
    entries = variants.tree_load()
    entry = entries.get(name)
    if entry is None:
        known = ", ".join(sorted(entries)) or "none"
        _stop(reason=f"variant {name!r} is not in the tree (variants/tree.json); known: {known}")
    return entry.parent_node


def _variant_package(variant: str) -> Path | None:
    try:
        return variants.resolve_variant(reference=variant) if variant else None
    except (FileNotFoundError, ValueError) as exc:
        _stop(reason=str(exc))


def _variant_digest(package: Path | None) -> str:
    return variants.package_digest(package=package) if package else ""


def require_backend(model: core.ModelSpec) -> core.ModelBackend:
    backend = models.make_backend(spec=model)
    if not backend.is_running():
        _stop(reason=f"model backend {model.name} is not running; start it with: rsi models up {model.name}")
    return backend


@contextlib.contextmanager
def _model_lock(model: core.ModelSpec) -> Iterator[None]:
    """Hold the model server's lock, so only one process uses it at a time"""
    path = paths.models_dir() / "locks" / f"{model.port}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            holder = handle.read().strip() or "unknown"
            _stop(
                reason=f"{model.name} (port {model.port}) is in use by another rsi process, pid {holder} ({path}); "
                "wait for it to finish"
            )
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _settings_now(model: str, harness_name: str, benchmark: str, timeout: int) -> dict[str, object]:
    """The settings a run started now records"""
    h = harness.load(name=harness_name)
    return {
        "model": models.get(name=model),
        "harness": h.name,
        "harness_version": h.version,
        "dataset": benchmarks.get(name=benchmark).dataset,
        "budget": core.Budget(timeout_s=timeout),
    }


def _made_here(harness_name: str, backend: core.ModelBackend, package: Path | None) -> dict[str, object]:
    """What a run started now records about the model server, the harness's contract and the package"""
    return {
        "server": backend.identity(),
        "contract": harness.load(name=harness_name).contract(),
        "variant_digest": _variant_digest(package=package),
    }


def make_node(
    name: str,
    dataset: str,
    harness_name: str,
    model: str,
    tasks: int,
    seed: int,
    timeout: int,
    backend: core.ModelBackend,
    variant: str = "",
    parent: str = "",
) -> core.Node:
    if len(name) > 200:
        _stop(reason=f"node name is {len(name)} chars; keep it under 200 (filesystem filename limit is 255)")
    h = harness.load(name=harness_name)
    if paths.node_dir(name=name).exists():
        _stop(reason=f"node {name!r} already exists at {paths.node_dir(name=name)}; choose a different --node name")
    variant_pkg = _variant_package(variant=variant)
    card = cards.read(package=variant_pkg) if variant_pkg else None
    parent_node = _require_node(name=parent) if parent else None
    diff_summary: str | None = None
    if variant_pkg is not None:
        base = variants.node_package(node=parent_node, baseline=h.package_dir) if parent_node else h.package_dir
        diff_summary = variants.diff_summary(base=base, variant=variant_pkg)
    node = core.Node.model_validate(
        {
            "name": name,
            "variant": variants.reference(package=variant_pkg) if variant_pkg else "",
            "parent": parent or None,
            "title": card.title if card else None,
            "diff_summary": diff_summary,
            "tasks": tasks,
            "seed": seed,
            "machine": platform.platform(),
            **_settings_now(model=model, harness_name=harness_name, benchmark=dataset, timeout=timeout),
            **_made_here(harness_name=harness_name, backend=backend, package=variant_pkg),
        }
    )
    runs.write_node(node=node)
    log.info(f"node {name} — outputs in {paths.node_dir(name=name)}")
    log.info(f"  {dataset} x {model} x {h.name} {h.version}, {tasks or 'all'} tasks (seed {seed})")
    return node


def _read_instance_ids(text_file: str) -> list[str]:
    path = Path(text_file).expanduser()
    if not path.is_file():
        _stop(reason=f"instances file not found: {path}")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not ids:
        _stop(reason=f"instances file has no ids: {path}")
    if len(set(ids)) != len(ids):
        _stop(reason=f"instances file has duplicate ids: {path}")
    return ids


HEALTH_TIMEOUT_S = 60  # for the one-token check after a task goes wrong
BROKEN_START = 3  # stop if this many first tasks all fail to start on a healthy server


def run_tasks(
    node: core.Node,
    h: harness.Harness,
    backend: core.ModelBackend,
    n_concurrent: int,
    instance_ids: list[str] | None = None,
) -> None:
    _retarget_logs(node_name=node.name)
    with _model_lock(model=node.model):
        bm = benchmarks.get(name=node.dataset.name)
        if instance_ids is not None:
            instances = bm.by_ids(ids=instance_ids)
            log.info(f"using explicit set of {len(instances)} instances from {node.dataset.name}")
        else:
            instances = bm.select(tasks=node.tasks, seed=node.seed)
            log.info(f"selected {len(instances)} instances from {node.dataset.name} (seed {node.seed})")
        selected = [i.instance_id for i in instances]
        if not node.task_ids:
            node.task_ids = selected
            runs.write_node(node=node)
        elif sorted(node.task_ids) != sorted(selected):
            # a changed selection would mix two task sets in one node
            _stop(
                reason=f"task set for node {node.name!r} no longer matches the recorded set "
                f"(recorded {len(node.task_ids)} ids); dataset revision or selection changed, so it can't be finished"
            )
        finished = {iid for iid, run in node.runs.items() if run.status in ("ok", "timeout")}
        todo = [i for i in instances if i.instance_id not in finished]
        skipped = len(instances) - len(todo)
        if skipped:
            log.info(f"finishing {node.name}: skipping {skipped} finished tasks, running {len(todo)}")
        h.prepare(instances=instances)
        width = min(n_concurrent or node.model.parallel, node.model.parallel)
        if width < (n_concurrent or width):
            log.info(f"clamping concurrency {n_concurrent} -> {width} (model parallel={node.model.parallel})")
        log.info(f"running {width} tasks at a time")
        _run_pool(node=node, h=h, backend=backend, todo=todo, width=width)
    runs.write_preds(path=paths.node_dir(name=node.name) / "preds.json", runs=node.runs, model_name=node.model.name)


def _run_pool(
    node: core.Node, h: harness.Harness, backend: core.ModelBackend, todo: list[core.Instance], width: int
) -> None:
    """Run the tasks, stopping if the model server stops answering after a task goes wrong"""
    variant = variants.resolve_variant(reference=node.variant) if node.variant else None
    # one task that got going shows the package works
    started = any(run.metrics.requests > 0 for run in node.runs.values() if run.status in ("ok", "timeout"))
    broken: list[str] = []
    done = 0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=width)
    try:
        futures = {
            pool.submit(
                h.run,
                ctx=harness.RunContext(
                    node=node,
                    instance=instance,
                    out_dir=paths.instances_dir(node_name=node.name) / instance.instance_id,
                    model=node.model,
                    endpoint=backend.endpoint(),
                    budget=node.budget,
                    variant=variant,
                ),
            ): instance
            for instance in todo
        }
        for future in concurrent.futures.as_completed(futures):
            instance = futures[future]
            try:
                run = future.result()
            except Exception as exc:  # noqa: BLE001
                run = core.TaskRun(
                    instance_id=instance.instance_id,
                    status="error",
                    error=f"harness crashed: {type(exc).__name__}: {exc}",
                )
            ended = run.status
            broke = ended == "error" or run.metrics.requests == 0
            down = (broke or ended == "timeout") and not backend.healthy(timeout_s=HEALTH_TIMEOUT_S)
            if down:
                error = f"the model server stopped answering; the task ended {ended}: {run.error or '-'}"
                run = run.model_copy(update={"status": "error", "error": error})
            node.runs[run.instance_id] = run
            try:
                runs.record_task_run(node_name=node.name, run=run)
                runs.write_node(node=node)
            except OSError as exc:
                log.error(f"failed to persist task {run.instance_id} ({exc}); finishing the run runs it again")
            done += 1
            m = run.metrics
            log.info(
                f"[{done}/{len(todo)}] {instance.instance_id} {run.status} {run.seconds}s {m.requests} req "
                f"{m.output_tokens} tok out {m.tok_s} tok/s ctx {m.context_pct}% stop={m.stop_reason or '-'}"
            )
            if run.error:
                log.info(f"    error: {run.error[:300]}")
            if down:
                _abandon(pool=pool, h=h)
                _stop(
                    reason=f"the model server ({node.model.name}) didn't answer a one-token request within "
                    f"{HEALTH_TIMEOUT_S} s after {instance.instance_id} ended {ended}, so {node.name} stops here; "
                    "that task counts as an error, so finishing the run once the server answers runs it again"
                )
            if broke:
                broken.append(run.error or f"ended {ended} without a model request")
            started = started or not broke
            if not started and len(broken) == BROKEN_START:
                _abandon(pool=pool, h=h)
                _stop(
                    reason=f"the first {BROKEN_START} tasks of {node.name} all errored or never started while the "
                    f"model server answered, so the run stops (first error: {broken[0][:300]})"
                )
    except KeyboardInterrupt:
        log.warning("interrupted; stopping in-flight tasks")
        _abandon(pool=pool, h=h)
        raise SystemExit(130)
    pool.shutdown(wait=True)


def _abandon(pool: concurrent.futures.ThreadPoolExecutor, h: harness.Harness) -> None:
    # don't wait for running tasks, which could run until their timeout
    h.cleanup()
    pool.shutdown(wait=False, cancel_futures=True)


def score_node(name: str, workers: int, clone_tasks: bool) -> None:
    node = runs.read_node(name=name)
    _retarget_logs(node_name=node.name)
    preds = paths.node_dir(name=node.name) / "preds.json"
    if not preds.exists():
        _stop(reason=f"no preds.json for node {name!r}; run `rsi run` first")
    bm = benchmarks.get(name=node.dataset.name)
    bm.ensure_assets(clone=clone_tasks)
    # the scorer's work folder. It skips tasks it already graded here, so a second pass grades only the rest
    scratch = paths.node_dir(name=node.name) / "scoring"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir()
    for attempt in (1, 2):
        per_instance = bm.score(
            preds=preds,
            run_id=node.name,
            model_name=node.model.name,
            workers=workers,
            cwd=scratch,
            node_dir=paths.node_dir(name=node.name),
            instances_dir=paths.instances_dir(node_name=node.name),
        )
        ungraded = sorted(set(node.runs) - set(per_instance))
        if not ungraded or attempt == 2:
            break
        log.warning(f"grading {len(ungraded)} tasks of {name} again, whose evaluation failed: {', '.join(ungraded)}")
    if ungraded:
        log.error(
            f"{name}'s evaluation failed twice for {', '.join(ungraded)}, so they count as unresolved "
            f"(the scorer's files are in {scratch})"
        )
        per_instance |= dict.fromkeys(ungraded, False)
    else:
        shutil.rmtree(scratch, ignore_errors=True)
    runs.apply_score(node=node, per_instance=per_instance)
    log.info(f"score: {node.resolved}/{node.total} resolved ({node.score}%)")
    _log_server_stats(model=node.model)


def _log_server_stats(model: core.ModelSpec) -> None:
    server = models.make_backend(spec=model).log_stats()
    if server:
        log.info(
            f"server: {server.completion_tokens} tok gen, {server.tok_s:.1f} tok/s, "
            f"cache hit avg {server.avg_cache_hit or 0.0:.1f}% over {server.requests} requests"
        )


@models_app.command("list")
def models_list() -> None:
    """Print every model in models.toml with its settings"""
    for spec in models.registry().values():
        log.info(
            f"{spec.name}: {spec.source} backend={spec.backend} port={spec.port} ctx={spec.ctx} "
            f"parallel={spec.parallel} thinking={spec.thinking} output cap={spec.output_cap} seed={spec.seed}"
        )
        log.info(f"  sampling: {spec.sampling}")
        if spec.chat_template_kwargs:
            log.info(f"  chat_template_kwargs: {spec.chat_template_kwargs}")


@models_app.command("up")
def models_up(name: str, wait: int = typer.Option(300, help="Seconds to wait for it to be ready")) -> None:
    """Start a model's server, downloading the model the first time"""
    backend = models.make_backend(spec=models.get(name=name))
    if backend.is_running():
        log.info(f"{name} already running at {backend.host_url()}")
        return
    log.info(f"starting {name} (backend {backend.spec.backend})")
    backend.start()
    backend.wait_ready(timeout=float(wait))
    log.info(f"{name} ready at {backend.host_url()}")


@models_app.command("down")
def models_down(name: str) -> None:
    """Stop a model's server"""
    backend = models.make_backend(spec=models.get(name=name))
    if not backend.is_running():
        log.info(f"{name} is not running")
        return
    backend.stop()
    log.info(f"{name} stopped")


@models_app.command("status")
def models_status(name: str) -> None:
    """Show whether a model's server is up and how it's doing"""
    backend = models.make_backend(spec=models.get(name=name))
    state = "running" if backend.is_running() else "stopped"
    log.info(f"{name}: {state} at {backend.host_url()}")
    for line in backend.status_lines():
        log.info(line)


@images_app.command("build")
def images_build(
    benchmark: str = typer.Option("swebench-verified", "--benchmark", help="Benchmark name"),
    harness_name: str = typer.Option("pi", "--harness"),
    tasks: int = typer.Option(0, "--tasks", help="Number of tasks (0 for all)"),
    seed: int = typer.Option(42, "--seed"),
    instances_file: str = typer.Option("", "--instances", help="File of task ids, one per line (overrides --tasks)"),
    n_concurrent: int = typer.Option(
        1, "--n-concurrent", help="Images built at once (a task's conda build can take most of Docker's memory)"
    ),
    clone_tasks: bool = typer.Option(False, "--clone-tasks"),
) -> None:
    """Build the task images and harness overlays a run needs"""
    bm = benchmarks.get(name=benchmark)
    bm.ensure_assets(clone=clone_tasks)
    if instances_file:
        instances = bm.by_ids(ids=_read_instance_ids(text_file=instances_file))
        log.info(f"building {len(instances)} instance images from {instances_file}")
    else:
        instances = bm.select(tasks=tasks, seed=seed)
        log.info(f"building {len(instances)} instance images (seed {seed})")
    bm.build_images(instances=instances, workers=n_concurrent, cwd=paths.rsi_root())
    log.info("building harness overlays")
    harness.load(name=harness_name).prepare(instances=instances)


@images_app.command("clean")
def images_clean(benchmark: str = typer.Option("swebench-verified", "--benchmark")) -> None:
    """Remove the benchmark's eval images to free disk space"""
    benchmarks.get(name=benchmark).clean_images(cwd=paths.rsi_root())


@app.command()
def run(
    variant: str = typer.Option("", "--variant", help="Variant name or package path (default: the baseline)"),
    model: str = typer.Option("", "--model", help="Model from models.toml (default for a variant: its parent's)"),
    benchmark: str = typer.Option("", "--benchmark", help="Benchmark (default for a variant: its parent's)"),
    harness_name: str = typer.Option(
        "", "--harness", help="Harness, a folder under harnesses/ (default: the parent's, else pi)"
    ),
    tasks: int = typer.Option(10, "--tasks", help="Number of tasks, sampled by --seed (0 for all)"),
    instances_file: str = typer.Option("", "--instances", help="File of task ids, one per line (overrides --tasks)"),
    probe: bool = typer.Option(
        False,
        "--probe",
        help="Run the lineage's probe set (overrides --tasks and --instances)",
    ),
    seed: int = typer.Option(42, "--seed", help="Seed for the --tasks sample, also part of the run's name"),
    node: str = typer.Option(
        "",
        "--node",
        help="Node name (default: <model>--<harness>--<benchmark>--<variant>--<tasks>t--s<seed>)",
    ),
    n_concurrent: int = typer.Option(
        0, "--n-concurrent", help="Tasks at once, up to the model's parallel setting (default: that setting)"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Time limit per task in seconds, after which it's scored as is"
    ),
    score_after: bool = typer.Option(True, "--score/--no-score", help="Score the run once its tasks are done"),
    env: str = typer.Option("docker", "--env", help="Where tasks run (only docker)"),
) -> None:
    """Run a harness on a set of tasks and score it (run it again to finish a stopped run)"""
    if env != "docker":
        _stop(reason=f"env {env!r} not implemented; only docker is supported")
    if probe and instances_file:
        _stop(reason="--probe and --instances are mutually exclusive")
    parent = _variant_parent_node(variant=variant)
    if parent:
        made_from = _require_node(name=parent)
        model = model or made_from.model.name
        benchmark = benchmark or made_from.dataset.name
        harness_name = harness_name or made_from.harness
    harness_name = harness_name or "pi"
    if not model or not benchmark:
        _stop(reason="the baseline needs --model and --benchmark")
    instance_ids: list[str] | None = None
    if probe:
        root = _probe_root(parent=parent, model=model, harness_name=harness_name, benchmark=benchmark, timeout=timeout)
        parent = parent or root.name  # a baseline run on the probe set counts as made from the root
        instance_ids = _lineage_probe(root=root)
        log.info(f"probe set of {root.name}: {len(instance_ids)} tasks ({runs.probe_path(root=root)})")
    elif instances_file:
        instance_ids = _read_instance_ids(text_file=instances_file)
    if instance_ids is not None:
        tasks = len(instance_ids)
    name = node or _node_name(
        model=model, harness_name=harness_name, benchmark=benchmark, variant=variant, tasks=tasks, seed=seed
    )
    if paths.node_dir(name=name).exists():
        node_record = _require_node(name=name)
        if node_record.total > 0 and all(task.status != "error" for task in node_record.runs.values()):
            again = "" if node else f"; pass --seed {_free_seed(name_for=name, seed=seed)} for another run of it"
            _stop(reason=f"{name} already ran: {node_record.resolved}/{node_record.total} resolved{again}")
        h, backend = _resumable(
            record=node_record,
            model=model,
            benchmark=benchmark,
            tasks=tasks,
            seed=seed,
            variant=variant,
            timeout=timeout,
        )
    else:
        # the node records the server's identity, so it has to be up first
        backend = require_backend(model=models.get(name=model))
        if probe:
            _check_compares(
                parent=_require_node(name=parent),
                variant=variant,
                model=model,
                harness_name=harness_name,
                benchmark=benchmark,
                tasks=tuple(sorted(instance_ids or [])),
                backend=backend,
                timeout=timeout,
            )
        node_record = make_node(
            name=name,
            dataset=benchmark,
            harness_name=harness_name,
            model=model,
            tasks=tasks,
            seed=seed,
            timeout=timeout,
            backend=backend,
            variant=variant,
            parent=parent,
        )
        h = harness.load(name=harness_name)
    run_tasks(node=node_record, h=h, backend=backend, n_concurrent=n_concurrent, instance_ids=instance_ids)
    if score_after:
        score_node(name=name, workers=2, clone_tasks=True)


def _node_name(model: str, harness_name: str, benchmark: str, variant: str, tasks: int, seed: int) -> str:
    return tree.derive_node_name(
        model_name=model,
        harness_name=harness_name,
        benchmark_name=benchmark,
        variant_name=variants.variant_name(reference=variant),
        tasks=tasks,
        seed=seed,
    )


def _free_seed(name_for: str, seed: int) -> int:
    """The next seed without a run of this harness on these tasks"""
    stem = name_for.removesuffix(f"--s{seed}")
    taken = seed + 1
    while paths.node_dir(name=f"{stem}--s{taken}").exists():
        taken += 1
    return taken


def _resumable(
    record: core.Node, model: str, benchmark: str, tasks: int, seed: int, variant: str, timeout: int
) -> tuple[harness.Harness, core.ModelBackend]:
    """Check that a node resumes with the same arguments, settings, checkout and model server"""
    variant_pkg = _variant_package(variant=variant)
    asked = {
        "model": (record.model.name, model),
        "benchmark": (record.dataset.name, benchmark),
        "tasks": (record.tasks, tasks),
        "seed": (record.seed, seed),
        "variant": (
            record.variant or "baseline",
            variants.reference(package=variant_pkg) if variant_pkg else "baseline",
        ),
    }
    mismatches = [f"{what} {was} != {now}" for what, (was, now) in asked.items() if was != now]
    if not mismatches:
        backend = require_backend(model=record.model)
        now = record.model_copy(
            update={
                "name": "this run",
                **_settings_now(model=model, harness_name=record.harness, benchmark=benchmark, timeout=timeout),
                **_made_here(harness_name=record.harness, backend=backend, package=variant_pkg),
            }
        )
        mismatches = tree.made_differently(compared=[record, now], packages=[[record, now]])
        if not mismatches:
            log.info(f"finishing {record.name}")
            return harness.load(name=record.harness), backend
    _stop(
        reason=f"node {record.name!r} was run under different settings ({'; '.join(mismatches)}); finishing it needs "
        "the same arguments, settings, checkout and model server, or remove its folder to start over"
    )


@app.command("tree")
def tree_command(model: str = typer.Option("", "--model", help="Model name prefix to show")) -> None:
    """Print every run as a tree with its score (* marks the best on each task set)"""
    nodes = runs.list_nodes()
    if not nodes:
        _stop(reason="no nodes under runs/")
    print(tree.render_tree(nodes=nodes, model_filter=model))


@app.command()
def view(
    port: int = typer.Option(8765, "--port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Address to listen on (default: this machine only)"),
    open_in_browser: bool = typer.Option(False, "--open", help="Open it in the browser"),
    save: str = typer.Option("", "--save", help="Save the viewer to this HTML file to share, instead of serving it"),
    nodes: Annotated[
        list[str] | None,
        typer.Option("--node", help="With --save, include only this node's trajectories (repeatable)"),
    ] = None,
) -> None:
    """Show the tree, runs, task trajectories and teacher sessions in a web page"""
    if not save:
        viewer.serve(host=host, port=port, open_browser=open_in_browser)
        return
    path = Path(save).expanduser()
    try:
        size = viewer.save(path=path, node_names=nodes or [])
    except KeyError as exc:
        _stop(reason=str(exc))
    log.info(f"saved the viewer to {path} ({size / 1e6:.1f} MB)")


@app.command()
def score(
    node: str = typer.Option(..., "--node"),
    workers: int = typer.Option(2, help="Eval containers at once"),
    clone_tasks: bool = typer.Option(False, "--clone-tasks"),
) -> None:
    """Score a run with the benchmark's official grader"""
    score_node(name=node, workers=workers, clone_tasks=clone_tasks)


@app.command("probe")
def probe_command(
    root: str = typer.Option(..., "--root", help="The scored root node, e.g. the baseline run"),
    history: Annotated[
        list[str] | None,
        typer.Option(
            "--history",
            help="Folder of older runs that count toward never-solved tasks (repeatable)",
        ),
    ] = None,
    tasks_file: str = typer.Option("", "--tasks", help="File of the probe's task ids, to use instead of picking them"),
    hidden_file: str = typer.Option("", "--hidden", help="With --tasks, a file of the ones hidden from the teacher"),
) -> None:
    """Pick the probe set from the root's results (its failures plus 30 successes, half hidden), or take it as given"""
    record = _require_node(name=root)
    if record.total == 0:
        _stop(reason=f"node {root!r} has no scored tasks; score it first")
    nodes = runs.list_nodes()
    first = tree.lineage(name=record.name, nodes=nodes)[0]
    if first.name != record.name:
        _stop(reason=f"{root} isn't a lineage's root: its lineage starts at {first.name}")
    path = runs.probe_path(root=record)
    if path.exists():
        _stop(reason=f"{path} is written already, and a lineage's probe set never changes once a run has used it")
    if bool(tasks_file) != bool(hidden_file):
        _stop(reason="--tasks and --hidden go together")
    if tasks_file and history:
        _stop(reason="--history only counts toward a probe set picked from the root's results, not --tasks")
    if tasks_file:
        probe = _read_instance_ids(text_file=tasks_file)
        hidden_ids = _read_instance_ids(text_file=hidden_file)
        try:
            tasks = runs.carry_probe(root=record, probe=probe, hidden=hidden_ids)
        except ValueError as exc:
            _stop(reason=str(exc))
    else:
        for folder in history or []:
            found, unread = runs.history_in(folder=Path(folder).expanduser())
            if not found:
                _stop(reason=f"no node records (*/node.json) this code can read under {folder}")
            log.info(f"counting {len(found)} runs from {folder}")
            if unread:
                log.warning(f"left out {len(unread)} runs whose records are in an older format: {', '.join(unread)}")
            nodes += found
        # don't count the same run twice
        unique = list({(n.name, n.created): n for n in nodes}.values())
        instances = benchmarks.get(name=record.dataset.name).by_ids(ids=sorted(record.runs))
        repos = {instance.instance_id: instance.repo for instance in instances}
        tasks = runs.write_probe(root=record, nodes=unique, repos=repos)
    log.info(f"probe set of {record.name}: {len(tasks)} tasks, written to {path}")
    hidden = runs.lineage_hidden(root=record)
    log.info(f"the teacher never sees {len(hidden)} of them, listed in {runs.hidden_path(root=record)}")
    for line in path.read_text().splitlines():
        if line.startswith("# left out"):
            log.info(line.removeprefix("# "))


@app.command()
def compare(
    a: Annotated[list[str], typer.Option("--a", help="Node to compare against, e.g. a baseline run (repeatable)")],
    b: Annotated[list[str], typer.Option("--b", help="Node to compare, e.g. an evolved harness's run (repeatable)")],
) -> None:
    """Compare two sets of runs on the tasks they share"""
    side_a, side_b = [_require_node(name=name) for name in a], [_require_node(name=name) for name in b]
    both = side_a + side_b
    unscored = [n.name for n in both if n.total == 0]
    if unscored:
        _stop(reason=f"not scored yet: {', '.join(unscored)}")
    shared = sorted(set.intersection(*[set(n.runs) for n in both]))
    if not shared:
        _stop(reason="these runs have no task in common")
    for problem in tree.made_differently(compared=both, packages=[side_a, side_b]):
        log.warning(f"the comparison mixes runs made differently, {problem}")
    width = max(len(n.name) for n in both) + 2
    print(f"solves on the {len(shared)} tasks every run has")
    for label, side in (("a", side_a), ("b", side_b)):
        solves = [sum(n.runs[task].resolved for task in shared) for n in side]
        for n, solved in zip(side, solves, strict=True):
            print(f"  {label}  {n.name:<{width}}{solved}")
        print(f"  {label}  {'mean':<{width}}{sum(solves) / len(solves):.1f}")
    print(f"b against a: {stats.evidence(variant=side_b, reference=side_a, tasks=shared).line(reference='a')}")


@app.command()
def gold(
    benchmark: str = typer.Option("swebench-verified", "--benchmark"),
    instance: str = typer.Option("sympy__sympy-20590", "--instance"),
    run_id: str = typer.Option("gold-check", "--run-id"),
    clone_tasks: bool = typer.Option(False, "--clone-tasks"),
) -> None:
    """Check the grader by scoring a task's reference patch, which should pass"""
    out_dir = paths.runs_dir() / "gold" / f"{instance}-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bm = benchmarks.get(name=benchmark)
    bm.ensure_assets(clone=clone_tasks)
    bm.gold(instance_id=instance, run_id=run_id, cwd=out_dir)
    log.info(f"gold artifacts: {out_dir}")


@app.command()
def report(node: str = typer.Option(..., "--node")) -> None:
    """Print a run's tasks, where its time went and the model server's stats"""
    record = runs.read_node(name=node)
    _retarget_logs(node_name=record.name)
    model = record.model
    log.info(f"node {record.name}: {record.resolved}/{record.total} resolved ({record.score}%)")
    log.info(
        f"model {model.name} ctx {model.ctx} thinking {model.thinking} output cap {model.output_cap} "
        f"sampling {model.sampling} chat_template_kwargs {model.chat_template_kwargs}"
    )
    log.info(f"{'instance':<44} {'status':<8} {'secs':>7} {'req':>4} {'out_tok':>8} {'tok/s':>7} {'ctx%':>6}  resolved")
    for instance_id, task in sorted(record.runs.items()):
        m = task.metrics
        log.info(
            f"{instance_id:<44} {task.status:<8} {task.seconds:>7.0f} {m.requests:>4d} {m.output_tokens:>8d} "
            f"{m.tok_s:>7.1f} {m.context_pct:>6.1f}  {str(task.resolved).lower()}"
        )
    metrics = [task.metrics for task in record.runs.values()]
    wall_ms = sum(int(task.seconds * 1000) for task in record.runs.values())
    model_ms, tool_ms = sum(m.model_ms for m in metrics), sum(m.tool_ms for m in metrics)
    think_s, other_s = sum(m.think_ms for m in metrics) / 1000, (wall_ms - model_ms - tool_ms) / 1000
    log.info(
        f"timings: model {model_ms / 1000:.1f}s (thinking {think_s:.1f}s), tools {tool_ms / 1000:.1f}s, "
        f"harness/IO {other_s:.1f}s, patch {sum(m.patch_lines for m in metrics)} lines"
    )
    ttfts = [m.ttft_avg_ms for m in metrics if m.ttft_avg_ms]
    if ttfts:
        worst = max(m.ttft_max_ms for m in metrics)
        log.info(f"ttft: avg per task {sum(ttfts) / len(ttfts):.0f} ms, worst single request {worst:.0f} ms")
    _log_server_stats(model=model)


@variants_app.command("generate")
def variants_generate(
    parent: str = typer.Option(..., "--parent", help="Scored node to generate variants from"),
    teacher: str = typer.Option("opus", "--teacher", help="A teacher named in models.toml"),
    areas: Annotated[
        list[str] | None,
        typer.Option(
            "--area",
            help="Session area from prompts/areas.md, repeatable (default: open, least-covered areas, compose)",
        ),
    ] = None,
    timeout: int = typer.Option(3600, "--timeout", help="Time limit per teacher session in seconds"),
) -> None:
    """Generate variants from a scored node, one teacher session per area"""
    record = _require_node(name=parent)
    if record.total == 0:
        _stop(reason=f"node {record.name!r} has no scored tasks; the teacher learns from scored runs")
    known = generation.area_texts()
    unknown = [area for area in areas or [] if area not in known]
    if unknown:
        _stop(reason=f"unknown areas {', '.join(unknown)}; prompts/areas.md has {', '.join(known)}")
    _retarget_logs(node_name=record.name)
    h = harness.load(name=record.harness)
    # the teacher's container tries variants on the agent's model
    backend = require_backend(model=record.model)
    with _model_lock(model=record.model):
        areas = areas or generation.default_areas(record=record, nodes=runs.list_nodes())
        try:
            found = generation.generate(
                record=record,
                h=h,
                teacher_spec=models.teacher(name=teacher),
                endpoint=backend.endpoint(),
                areas=areas,
                timeout_s=timeout,
            )
        except (generation.TeacherUnavailable, harness.CheckUnavailable) as exc:
            _stop(reason=str(exc))
    for area, reason in found.declined.items():
        log.info(f"nothing worth running in {area}: {reason}")
    if found.registered:
        names = ", ".join(entry.name for entry in found.registered)
        log.info(f"registered {names}; run each with `rsi run --variant <name> --probe`")
    elif set(found.declined) != set(areas):
        _stop(reason=f"no variant was registered; the round's sessions and their problems are in {found.out}")


@variants_app.command("list")
def variants_list(
    show: str = typer.Option("", "--show", help="Show one variant in full: card, results and diff"),
) -> None:
    """Show how each variant did against its parent's harness"""
    entries = variants.tree_load()
    nodes = runs.list_nodes()
    if show:
        entry = entries.get(show)
        if entry is None:
            _stop(reason=f"variant {show!r} is not in the tree")
        reuses = f"; reuses {', '.join(entry.reuses)}" if entry.reuses else ""
        print(f"{entry.name}: {entry.title}")
        print(f"  parent:     {entry.parent} (node {entry.parent_node}){reuses}")
        print(f"  created:    {entry.created} by {entry.created_by}")
        print(f"  what:       {entry.what}")
        if entry.why:
            print(f"  why:        {entry.why}")
        if entry.expect:
            print(f"  expect:     {entry.expect}")
        for node in variants.variant_nodes(entry=entry, nodes=nodes):
            print(f"  run:        {variants.result_line(node=node, nodes=nodes)} [{node.name}]")
        for line in variants.scoreboard(entry=entry, nodes=nodes):
            print(f"  result:     {line}")
        parent = next((n for n in nodes if n.name == entry.parent_node), None)
        if parent is not None:
            base = variants.node_package(node=parent, baseline=harness.load(name=parent.harness).package_dir)
            print("\n" + variants.unified_diff(base=base, variant=variants.resolve_variant(reference=entry.name)))
        return
    if not entries:
        log.info("the variant tree is empty")
        return
    for name, entry in sorted(entries.items(), key=lambda kv: kv[1].created):
        print(f"{name}: {entry.title}")
        for line in variants.scoreboard(entry=entry, nodes=nodes):
            print(f"    {line}")


@variants_app.command("validate")
def variants_validate(
    name: str,
    harness_name: str = typer.Option(
        "pi", "--harness", help="Harness to check against, if the variant isn't in the tree"
    ),
    benchmark: str = typer.Option("swebench-verified", "--benchmark", help="Benchmark whose terms to check for"),
) -> None:
    """Run a variant's package checks again (a generation round also asks it the question, against the model)"""
    entry = variants.tree_load().get(name)
    if entry is not None:
        parent = runs.read_node(name=entry.parent_node)
        harness_name, benchmark = parent.harness, parent.dataset.name
    h = harness.load(name=harness_name)
    package = variants.resolve_variant(reference=name)
    try:
        problems = generation.package_problems(
            package=package,
            h=h,
            terms=benchmarks.banned_terms(benchmark=benchmarks.get(name=benchmark)) + h.protocol_terms,
            log_dir=paths.logs_dir() / "checks" / package.name,
        )
    except harness.CheckUnavailable as exc:
        _stop(reason=str(exc))
    if problems:
        _stop(reason="problems:\n" + "\n".join(f"  - {p}" for p in problems))
    log.info(f"{name} passes all checks")


def _probe_root(parent: str, model: str, harness_name: str, benchmark: str, timeout: int) -> core.Node:
    """The root whose probe set a run uses (for the baseline, the one root with a probe set made like a run now)"""
    nodes = runs.list_nodes()
    if parent:
        root = tree.lineage(name=parent, nodes=nodes)[0]
    else:
        settings = _settings_now(model=model, harness_name=harness_name, benchmark=benchmark, timeout=timeout)
        found = [
            n
            for n in nodes
            if not n.parent
            and not n.variant
            and (n.model.name, n.harness, n.dataset.name) == (model, harness_name, benchmark)
            and runs.probe_path(root=n).is_file()
            and tree.comparable(a=n, b=n.model_copy(update=settings))
        ]
        if len(found) != 1:
            which = f"more than one ({', '.join(n.name for n in found)})" if found else "none"
            _stop(
                reason=f"the baseline's probe set comes from its root run, and {which} of {model} on {benchmark} "
                "has one; write it with `rsi probe --root <node>`"
            )
        root = found[0]
    if root.total == 0:
        _stop(reason=f"node {root.name!r} has no scored tasks; score it first")
    return root


def _check_compares(
    parent: core.Node,
    variant: str,
    model: str,
    harness_name: str,
    benchmark: str,
    tasks: tuple[str, ...],
    backend: core.ModelBackend,
    timeout: int,
) -> None:
    """Refuse a probe run that wouldn't compare with its parent and the runs it's judged with"""
    nodes = runs.list_nodes()
    package = _variant_package(variant=variant)
    now = parent.model_copy(
        update={
            "name": "this run",
            **_settings_now(model=model, harness_name=harness_name, benchmark=benchmark, timeout=timeout),
            **_made_here(harness_name=harness_name, backend=backend, package=package),
        }
    )
    own = tree.runs_of(
        variant=variants.reference(package=package) if package else "", tasks=tasks, like=parent, nodes=nodes
    )
    reference = tree.reference_runs(parent=parent, tasks=tasks, nodes=nodes)
    problems = tree.made_differently(compared=[parent, *reference, *own, now], packages=[[*own, now]])
    if problems:
        why = "; ".join(problems)
        _stop(reason=f"this run wouldn't compare with the runs it's judged with, so it isn't started: {why}")


def _lineage_probe(root: core.Node) -> list[str]:
    try:
        return runs.lineage_probe(root=root)
    except FileNotFoundError as exc:
        _stop(reason=str(exc))
