import fcntl
import logging
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import typer
from click.testing import Result
from typer.testing import CliRunner

from rsi import benchmarks, cards, cli, core, generation, harness, models, paths, runs, variants

SERVER = {"engine": "fake 1.0", "weights": "sha256:abc"}
CONTRACT = "contract-1"


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    # node commands add log handlers, so restore the suite's afterwards
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_ids = {id(h) for h in saved}
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        if id(handler) not in saved_ids:
            handler.close()
    for handler in saved:
        root.addHandler(handler)


def _parent_node() -> core.Node:
    node = core.Node(
        name="t-node",
        dataset=core.DatasetSpec(name="swebench-verified", dataset_id="princeton-nlp/SWE-bench_Verified"),
        harness="pi",
        harness_version="1.3.0",
        model=core.ModelSpec(name="gemma-4-12b", source="path:/tmp/x.gguf", model_id="m"),
        budget=core.Budget(),
        tasks=1,
        seed=1,
    )
    node.runs["a__b-1"] = core.TaskRun(instance_id="a__b-1", status="ok", seconds=5.0)
    return node


def _copy_models_toml(root: Path) -> None:
    shutil.copy(paths.project_dir() / "models.toml", root / "models.toml")


def _card(title: str) -> cards.VariantCard:
    return cards.VariantCard(title=title, what=f"{title}: the change")


class _FakeBackend:
    def __init__(self, answers: bool = True, server: dict[str, str] | None = None) -> None:
        self.answers = answers
        self.server = server or SERVER
        self.checks = 0

    def is_running(self) -> bool:
        return True

    def endpoint(self) -> core.ModelEndpoint:
        return core.ModelEndpoint(model_id="m", base_url="http://fake")

    def identity(self) -> dict[str, str]:
        return dict(self.server)

    def healthy(self, timeout_s: float) -> bool:
        self.checks += 1
        return self.answers


class _FakeHarness:
    name = "pi"
    version = "1.3.0"
    package_dir = paths.harnesses_dir() / "pi" / "package"

    def __init__(self, outcomes: dict[str, core.TaskRun] | None = None, contract: str = CONTRACT) -> None:
        self.outcomes = outcomes or {}
        self.digest = contract
        self.cleaned = False
        self.ran: list[str] = []

    def contract(self) -> str:
        return self.digest

    def prepare(self, instances: list[core.Instance]) -> None:
        return None

    def run(self, ctx: harness.RunContext) -> core.TaskRun:
        iid = ctx.instance.instance_id
        self.ran.append(iid)
        ok = core.TaskRun(instance_id=iid, status="ok", patch="+x", metrics=core.TaskMetrics(requests=3))
        return self.outcomes.get(iid, ok)

    def cleanup(self) -> None:
        self.cleaned = True


def _fake_world(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend | None = None, h: _FakeHarness | None = None
) -> _FakeBackend:
    backend = backend or _FakeBackend()
    fake = h or _FakeHarness()
    monkeypatch.setattr(cli, "require_backend", lambda model: backend)
    monkeypatch.setattr(cli.harness, "load", lambda name: fake)
    return backend


def test_images_build_takes_a_task_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    built: list[list[str]] = []

    class Benchmark:
        def ensure_assets(self, clone: bool) -> None:
            pass

        def by_ids(self, ids: list[str]) -> list[core.Instance]:
            return [core.Instance(instance_id=i, repo="r", base_commit="c", task="t") for i in ids]

        def build_images(self, instances: list[core.Instance], workers: int, cwd: Path) -> None:
            built.append([i.instance_id for i in instances])

    class Harness:
        def prepare(self, instances: list[core.Instance]) -> None:
            built.append([i.instance_id for i in instances])

    monkeypatch.setattr(cli.benchmarks, "get", lambda name: Benchmark())
    monkeypatch.setattr(cli.harness, "load", lambda name: Harness())
    pool = tmp_path / "pool.txt"
    pool.write_text("# a pool picked by hand\nb__b-2\na__a-1\n")
    assert _invoke(args=["images", "build", "--instances", str(pool)]).exit_code == 0
    assert built == [["b__b-2", "a__a-1"], ["b__b-2", "a__a-1"]]


def test_a_stop_reason_reaches_the_log_files(caplog: pytest.LogCaptureFixture) -> None:
    def refuse() -> None:
        cli._stop(reason="nothing to do")

    with pytest.raises(typer.Exit) as stopped:
        refuse()
    assert stopped.value.exit_code == 1
    record = caplog.records[-1]
    assert (record.levelno, record.getMessage(), record.funcName) == (logging.ERROR, "nothing to do", "refuse")


def test_variants_generate_needs_a_scored_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    runs.write_node(node=_parent_node())
    with pytest.raises(typer.Exit):
        cli.variants_generate(parent="t-node", teacher="opus", areas=None, timeout=1)
    assert "no scored tasks" in caplog.text


def _ready_to_generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, found: generation.Round | Exception
) -> list[list[str]]:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    parent = _parent_node()
    parent.total = 1
    runs.write_node(node=parent)
    _fake_world(monkeypatch=monkeypatch)
    monkeypatch.setattr(
        cli.models, "teacher", lambda name: core.TeacherSpec(name=name, agent="copilot", model="m", effort="max")
    )
    asked: list[list[str]] = []

    def generate(
        record: core.Node,
        h: harness.Harness,
        teacher_spec: core.TeacherSpec,
        endpoint: core.ModelEndpoint,
        areas: list[str],
        timeout_s: int,
    ) -> generation.Round:
        asked.append(areas)
        if isinstance(found, Exception):
            raise found
        return found

    monkeypatch.setattr(cli.generation, "generate", generate)
    return asked


def test_variants_generate_runs_the_areas_asked_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    made = variants.tree_register(name="v", parent="baseline", parent_node="t-node", card=_card(title="V"), diff="+")
    asked = _ready_to_generate(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        found=generation.Round(out=tmp_path / "round", registered=[made], declined={}),
    )
    command = ["variants", "generate", "--parent", "t-node"]
    assert _invoke(args=command).exit_code == 0
    assert _invoke(args=[*command, "--area", "context", "--area", "tools"]).exit_code == 0
    assert asked == [["open", "tools", "middleware", "context"], ["context", "tools"]]
    unknown = _invoke(args=[*command, "--area", "tools", "--area", "everything"])
    assert unknown.exit_code == 1 and "unknown areas everything" in unknown.output
    assert len(asked) == 2


@pytest.mark.parametrize("declined", [{}, {"tools": "Every tool idea here is tried."}])
def test_variants_generate_fails_when_nothing_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, declined: dict[str, str]
) -> None:
    nothing = generation.Round(out=tmp_path / "round", registered=[], declined=declined)
    _ready_to_generate(tmp_path=tmp_path, monkeypatch=monkeypatch, found=nothing)
    with pytest.raises(typer.Exit) as stopped:
        cli.variants_generate(parent="t-node", teacher="opus", areas=["tools", "open"], timeout=1)
    assert stopped.value.exit_code == 1 and "no variant was registered" in caplog.text


def test_a_round_where_every_session_found_nothing_worth_running_ends_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    reasons = {"tools": "Every tool idea here is tried.", "open": "Nothing else either."}
    declined = generation.Round(out=tmp_path / "round", registered=[], declined=reasons)
    _ready_to_generate(tmp_path=tmp_path, monkeypatch=monkeypatch, found=declined)
    with caplog.at_level(logging.INFO):
        cli.variants_generate(parent="t-node", teacher="opus", areas=["tools", "open"], timeout=1)
    assert "nothing worth running in tools: Every tool idea here is tried." in caplog.text
    assert "nothing worth running in open: Nothing else either." in caplog.text
    assert "no variant was registered" not in caplog.text


@pytest.mark.parametrize(
    "down",
    [
        generation.TeacherUnavailable(
            "session 1 failed before doing any work (429 rate limited), so the round stops; the areas left are "
            "tools, open"
        ),
        # the model stopped answering, so the variant's question check says nothing about it
        harness.CheckUnavailable("the javascript question couldn't run: pi got no response from the model"),
    ],
)
def test_a_teacher_or_model_that_is_down_stops_the_round_with_its_reason_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, down: Exception
) -> None:
    _ready_to_generate(tmp_path=tmp_path, monkeypatch=monkeypatch, found=down)
    with pytest.raises(typer.Exit) as stopped:
        cli.variants_generate(parent="t-node", teacher="opus", areas=["tools", "open"], timeout=1)
    assert stopped.value.exit_code == 1 and str(down) in caplog.text


def test_make_node_records_what_made_its_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    parent = _parent_node()
    parent.total = 1
    runs.write_node(node=parent)
    vdir = paths.variants_dir() / "tight-loop"
    vdir.mkdir(parents=True)
    (vdir / "VARIANT.md").write_text("End every turn with progress\nWhat: w\n")
    (vdir / "APPEND_SYSTEM.md").write_text("Work in a tight loop.\n")
    monkeypatch.setattr(
        cli.models, "get", lambda name: core.ModelSpec(name="gemma-4-12b", source="path:/tmp/x.gguf", model_id="m")
    )
    backend = _fake_world(monkeypatch=monkeypatch)
    cli.variants.tree_register(
        name="tight-loop", parent="baseline", parent_node="t-node", card=_card(title="Tight loop"), diff="+ a.md"
    )
    node = cli.make_node(
        name="gemma-4-12b--pi--swebench-verified--tight-loop--1t--s1",
        dataset="swebench-verified",
        harness_name="pi",
        model="gemma-4-12b",
        tasks=1,
        seed=1,
        timeout=3600,
        backend=backend,
        variant="tight-loop",
        parent="t-node",
    )
    assert node.variant == "tight-loop"
    assert node.parent == "t-node"
    assert node.title == "End every turn with progress"
    assert "+ APPEND_SYSTEM.md" in node.diff_summary
    assert (node.server, node.contract) == (SERVER, CONTRACT)
    assert node.variant_digest == variants.package_digest(package=vdir)
    stored = runs.read_node(name=node.name)
    assert stored.variant == "tight-loop" and stored.parent == "t-node"
    assert stored.server == SERVER and stored.variant_digest == node.variant_digest
    baseline = cli.make_node(
        name="gemma-4-12b--pi--swebench-verified--baseline--1t--s1",
        dataset="swebench-verified",
        harness_name="pi",
        model="gemma-4-12b",
        tasks=1,
        seed=1,
        timeout=3600,
        backend=backend,
    )
    assert baseline.variant_digest == "" and baseline.contract == CONTRACT


def test_variants_list_shows_how_each_variant_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    parent = _parent_node()
    parent.total = 1
    runs.write_node(node=parent)
    vdir = paths.variants_dir() / "tight-loop"
    vdir.mkdir(parents=True)
    (vdir / "package.json").write_text("{}\n")
    (vdir / "APPEND_SYSTEM.md").write_text("Work in a tight loop.\n")
    cli.variants.tree_register(
        name="tight-loop", parent="baseline", parent_node="t-node", card=_card(title="Tight loop"), diff="+ a.md"
    )
    cli.variants_list(show="")
    assert capsys.readouterr().out == "tight-loop: Tight loop\n    from t-node\n    not run\n"
    child = parent.model_copy(update={"name": "child", "parent": "t-node", "variant": str(vdir), "resolved": 1})
    child.runs = {"a__b-1": core.TaskRun(instance_id="a__b-1", status="ok", resolved=True)}
    runs.write_node(node=child)
    cli.variants_list(show="")
    assert capsys.readouterr().out.splitlines()[:4] == [
        "tight-loop: Tight loop",
        "    from t-node",
        "    1 run, 1.0/1 solved per run",
        "    solved +1.0 ± 1.0 tasks per run, within noise (1 run vs 1 of the parent, 1 tasks)",
    ]
    cli.variants_list(show="tight-loop")
    out = capsys.readouterr().out
    assert "what:       Tight loop: the change" in out
    assert "\n  run:        1/1 vs parent 0/1 on the same tasks (+1 fixed, -0 broken; " in out
    assert "\n  result:     1 run, 1.0/1 solved per run\n  result:     solved +1.0 ± 1.0 tasks per run" in out
    assert "+Work in a tight loop." in out


def _successor(tmp_path: Path, name: str, outcomes: list[bool], variant: str | None = None) -> core.Node:
    node = core.Node(
        name=f"gemma-4-12b--pi--swebench-verified--{name}--4t--s1",
        parent="t-node",
        variant=str(tmp_path / "variants" / name) if variant is None else variant,
        dataset=core.DatasetSpec(name="swebench-verified", dataset_id="princeton-nlp/SWE-bench_Verified"),
        harness="pi",
        harness_version="1.3.0",
        model=core.ModelSpec(name="gemma-4-12b", source="path:/tmp/x.gguf", model_id="m"),
        budget=core.Budget(),
        tasks=4,
        seed=1,
        server=SERVER,
        contract=CONTRACT,
    )
    node.runs = {f"t__t-{i}": core.TaskRun(instance_id=f"t__t-{i}", resolved=ok) for i, ok in enumerate(outcomes)}
    node.task_ids = sorted(node.runs)
    node.total, node.resolved, node.score = 4, sum(outcomes), 25.0 * sum(outcomes)
    return node


def _one_repo(node: core.Node) -> dict[str, str]:
    """Every task of node in the same repo"""
    return dict.fromkeys(node.runs, "r")


def _made_up_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the benchmark take a test's made-up task ids, all in one repo"""

    def by_ids(ids: list[str]) -> list[core.Instance]:
        return [core.Instance(instance_id=i, repo="r", base_commit="c", task="p") for i in ids]

    monkeypatch.setattr(benchmarks.get(name="swebench-verified"), "by_ids", by_ids)


class _FakeBench:
    dataset = core.DatasetSpec(name="swebench-verified", dataset_id="princeton-nlp/SWE-bench_Verified")

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def select(self, tasks: int, seed: int) -> list[core.Instance]:
        return [core.Instance(instance_id=i, repo="r", base_commit="c", task="p") for i in self._ids]

    def by_ids(self, ids: list[str]) -> list[core.Instance]:
        return [core.Instance(instance_id=i, repo="r", base_commit="c", task="p") for i in ids]


def _empty_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ids: list[str]) -> core.Node:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    node = _parent_node()
    node.runs = {}
    runs.write_node(node=node)
    monkeypatch.setattr(cli.benchmarks, "get", lambda name: _FakeBench(ids=ids))
    return node


def test_run_tasks_records_task_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    node = _empty_node(tmp_path=tmp_path, monkeypatch=monkeypatch, ids=["a__b-1"])
    cli.run_tasks(node=node, h=_FakeHarness(), backend=_FakeBackend(), n_concurrent=1)
    stored = runs.read_node(name=node.name)
    assert stored.task_ids == ["a__b-1"]
    assert stored.runs["a__b-1"].status == "ok"


def test_run_tasks_runs_as_many_at_once_as_the_model_is_served_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _empty_node(tmp_path=tmp_path, monkeypatch=monkeypatch, ids=["a__b-1"])
    node.model = node.model.model_copy(update={"parallel": 2})
    widths: list[int] = []
    monkeypatch.setattr(cli, "_run_pool", lambda node, h, backend, todo, width: widths.append(width))
    for asked in (0, 1, 5):
        cli.run_tasks(node=node, h=_FakeHarness(), backend=_FakeBackend(), n_concurrent=asked)
    assert widths == [2, 1, 2]


def test_run_tasks_uses_explicit_instance_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    node = _empty_node(tmp_path=tmp_path, monkeypatch=monkeypatch, ids=["a__b-1", "c__d-2"])
    cli.run_tasks(
        node=node, h=_FakeHarness(), backend=_FakeBackend(), n_concurrent=1, instance_ids=["c__d-2", "a__b-1"]
    )
    stored = runs.read_node(name=node.name)
    assert stored.task_ids == ["c__d-2", "a__b-1"]
    assert set(stored.runs) == {"c__d-2", "a__b-1"}


def test_a_timeout_while_the_server_answers_stays_a_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    node = _empty_node(tmp_path=tmp_path, monkeypatch=monkeypatch, ids=["a__b-1", "c__d-2"])
    slow = core.TaskRun(instance_id="a__b-1", status="timeout", patch="+x", metrics=core.TaskMetrics(requests=40))
    backend = _FakeBackend()
    cli.run_tasks(node=node, h=_FakeHarness(outcomes={"a__b-1": slow}), backend=backend, n_concurrent=1)
    stored = runs.read_node(name=node.name)
    assert stored.runs["a__b-1"].status == "timeout" and stored.runs["c__d-2"].status == "ok"
    assert backend.checks == 1
    assert (paths.node_dir(name=node.name) / "preds.json").is_file()


def test_a_task_that_went_wrong_with_the_server_down_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    node = _empty_node(tmp_path=tmp_path, monkeypatch=monkeypatch, ids=["a__b-1", "c__d-2", "e__f-3"])
    slow = core.TaskRun(
        instance_id="a__b-1",
        status="timeout",
        patch="+x",
        error="timed out after 3600s",
        metrics=core.TaskMetrics(requests=40),
    )
    h = _FakeHarness(outcomes={"a__b-1": slow})
    with pytest.raises(typer.Exit):
        cli.run_tasks(node=node, h=h, backend=_FakeBackend(answers=False), n_concurrent=1)
    stored = runs.read_node(name=node.name)
    assert stored.runs["a__b-1"].status == "error" and h.cleaned
    assert stored.runs["a__b-1"].error.startswith("the model server stopped answering; the task ended timeout")
    assert not (paths.node_dir(name=node.name) / "preds.json").exists()
    assert "didn't answer a one-token request within 60 s after a__b-1 ended timeout" in caplog.text


def test_a_run_whose_first_tasks_all_break_stops_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ids = [f"t__t-{i}" for i in range(5)]
    node = _empty_node(tmp_path=tmp_path, monkeypatch=monkeypatch, ids=ids)
    broken = {iid: core.TaskRun(instance_id=iid, status="error", error="pi died at startup") for iid in ids}
    broken["t__t-1"] = core.TaskRun(instance_id="t__t-1", status="timeout")
    with pytest.raises(typer.Exit) as stopped:
        cli.run_tasks(node=node, h=_FakeHarness(outcomes=broken), backend=_FakeBackend(), n_concurrent=1)
    assert stopped.value.exit_code == 1
    assert len(runs.read_node(name=node.name).runs) == 3
    assert "the first 3 tasks of t-node all errored or never started" in caplog.text


def test_errors_after_the_run_got_going_dont_stop_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = [f"t__t-{i}" for i in range(5)]
    node = _empty_node(tmp_path=tmp_path, monkeypatch=monkeypatch, ids=ids)
    node.runs = {"t__t-0": core.TaskRun(instance_id="t__t-0", status="ok", metrics=core.TaskMetrics(requests=9))}
    runs.write_node(node=node)
    broken = {iid: core.TaskRun(instance_id=iid, status="error", error="model request failed") for iid in ids[1:]}
    cli.run_tasks(node=node, h=_FakeHarness(outcomes=broken), backend=_FakeBackend(), n_concurrent=1)
    stored = runs.read_node(name=node.name)
    assert [stored.runs[iid].status for iid in ids] == ["ok", "error", "error", "error", "error"]


def test_read_instance_ids(tmp_path: Path) -> None:
    good = tmp_path / "ids.txt"
    good.write_text("# a comment\na__b-1\n\nc__d-2\n")
    assert cli._read_instance_ids(text_file=str(good)) == ["a__b-1", "c__d-2"]
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    with pytest.raises(typer.Exit):
        cli._read_instance_ids(text_file=str(empty))
    dup = tmp_path / "dup.txt"
    dup.write_text("a__b-1\na__b-1\n")
    with pytest.raises(typer.Exit):
        cli._read_instance_ids(text_file=str(dup))


def test_run_tasks_rejects_drifted_task_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    node = _empty_node(tmp_path=tmp_path, monkeypatch=monkeypatch, ids=["a__b-1"])
    node.task_ids = ["z__z-9"]
    runs.write_node(node=node)
    with pytest.raises(typer.Exit):
        cli.run_tasks(node=node, h=_FakeHarness(), backend=_FakeBackend(), n_concurrent=1)


class _FlakyGrader:
    def __init__(self, fails: int) -> None:
        self.fails = fails
        self.places: list[Path] = []

    def ensure_assets(self, clone: bool) -> None:
        return None

    def score(
        self,
        preds: Path,
        run_id: str,
        model_name: str,
        workers: int,
        cwd: Path,
        node_dir: Path,
        instances_dir: Path,
    ) -> dict[str, bool]:
        self.places.append(cwd)
        graded = {"a__b-1": True, "c__d-2": False}
        return graded if len(self.places) <= self.fails else graded | {"e__f-3": True}


@pytest.mark.parametrize("fails", [1, 2])
def test_scoring_grades_a_failed_evaluation_once_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fails: int
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    node = _parent_node()
    node.runs = {iid: core.TaskRun(instance_id=iid, status="ok", patch="+x") for iid in ("a__b-1", "c__d-2", "e__f-3")}
    runs.write_node(node=node)
    runs.write_preds(path=paths.node_dir(name="t-node") / "preds.json", runs=node.runs, model_name="gemma-4-12b")
    grader = _FlakyGrader(fails=fails)
    monkeypatch.setattr(cli.benchmarks, "get", lambda name: grader)
    monkeypatch.setattr(cli, "_log_server_stats", lambda model: None)
    cli.score_node(name="t-node", workers=2, clone_tasks=False)
    scored = runs.read_node(name="t-node")
    scratch = paths.node_dir(name="t-node") / "scoring"
    assert grader.places == [scratch, scratch]
    if fails == 1:
        assert (scored.resolved, scored.total) == (2, 3) and not scratch.exists()
    else:
        assert (scored.resolved, scored.total) == (1, 3) and scored.runs["e__f-3"].resolved is False
        assert "t-node's evaluation failed twice for e__f-3, so they count as unresolved" in caplog.text
        assert scratch.is_dir()


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(cli.app, args)


def test_finishing_a_run_rejects_mismatched_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _copy_models_toml(root=tmp_path)
    runs.write_node(node=_parent_node())
    args = ["run", "--benchmark", "swebench-verified", "--model", "gemma-4-12b", "--node", "t-node"]
    result = _invoke(args=[*args, "--tasks", "5", "--seed", "1"])
    assert result.exit_code == 1
    assert "node 't-node' was run under different settings (" in result.output and "tasks 1 != 5" in result.output


def test_finishing_a_run_rejects_mismatched_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _copy_models_toml(root=tmp_path)
    runs.write_node(node=_parent_node().model_copy(update={"contract": harness.load(name="pi").contract()}))
    args = ["run", "--benchmark", "other-benchmark", "--model", "gemma-4-12b", "--node", "t-node"]
    result = _invoke(args=[*args, "--tasks", "1", "--seed", "1"])
    assert result.exit_code == 1
    assert "was run under different settings (benchmark swebench-verified != other-benchmark)" in result.output


def test_finishing_a_run_rejects_changed_model_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _copy_models_toml(root=tmp_path)
    _fake_world(monkeypatch=monkeypatch)
    gemma = models.get(name="gemma-4-12b")
    made = {
        "model": gemma.model_copy(update={"ctx": 4096, "port": 1}),
        "dataset": benchmarks.get(name="swebench-verified").dataset,
        "server": SERVER,
        "contract": CONTRACT,
    }
    runs.write_node(node=_parent_node().model_copy(update=made))
    args = ["run", "--benchmark", "swebench-verified", "--model", "gemma-4-12b", "--node", "t-node"]
    result = _invoke(args=[*args, "--tasks", "1", "--seed", "1"])
    assert result.exit_code == 1
    assert "was run under different settings (not the same model settings: " in result.output
    # the port doesn't count
    assert '"ctx": 4096' in result.output and '"port"' not in result.output


@pytest.mark.parametrize("changed", ["server", "contract", "files"])
def test_finishing_a_run_rejects_a_node_made_differently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _copy_models_toml(root=tmp_path)
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "SYSTEM.md").write_text("Be brief.\n")
    made = {
        "model": models.get(name="gemma-4-12b"),
        "dataset": benchmarks.get(name="swebench-verified").dataset,
        "variant": str(package),
        "server": SERVER,
        "contract": CONTRACT,
        "variant_digest": variants.package_digest(package=package),
    }
    runs.write_node(node=_parent_node().model_copy(update=made))
    backend, h = _FakeBackend(), _FakeHarness()
    if changed == "server":
        backend = _FakeBackend(server={"engine": "fake 1.1"})
    elif changed == "contract":
        h = _FakeHarness(contract="contract-2")
    else:
        (package / "SYSTEM.md").write_text("Be thorough.\n")
    _fake_world(monkeypatch=monkeypatch, backend=backend, h=h)
    args = ["run", "--benchmark", "swebench-verified", "--model", "gemma-4-12b", "--node", "t-node"]
    result = _invoke(args=[*args, "--variant", str(package), "--tasks", "1", "--seed", "1"])
    assert result.exit_code == 1
    field = {"server": "model server", "contract": "harness contract", "files": "package files"}[changed]
    assert f"was run under different settings (not the same {field}: " in result.output


@pytest.mark.parametrize("scored", [False, True])
def test_running_a_run_again_finishes_it_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scored: bool
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _copy_models_toml(root=tmp_path)
    h = _FakeHarness()
    _fake_world(monkeypatch=monkeypatch, h=h)
    scores: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(
        cli, "score_node", lambda name, workers, clone_tasks: scores.append((name, workers, clone_tasks))
    )
    ids = ["a__b-1", "c__d-2", "e__f-3"]
    monkeypatch.setattr(cli.benchmarks, "get", lambda name: _FakeBench(ids=ids))
    name = "gemma-4-12b--pi--swebench-verified--baseline--3t--s1"
    made = {
        "name": name,
        "model": models.get(name="gemma-4-12b"),
        "dataset": _FakeBench.dataset,
        "tasks": 3,
        "task_ids": ids,
        "server": SERVER,
        "contract": CONTRACT,
    }
    node = _parent_node().model_copy(update=made)
    node.runs = {"a__b-1": core.TaskRun(instance_id="a__b-1", status="ok", resolved=True)}
    if scored:
        node.runs["c__d-2"] = core.TaskRun(instance_id="c__d-2", status="error", error="docker went away")
        node.runs["e__f-3"] = core.TaskRun(instance_id="e__f-3", status="timeout")
        node.total, node.resolved = 3, 1
    runs.write_node(node=node)
    args = ["run", "--model", "gemma-4-12b", "--benchmark", "swebench-verified", "--tasks", "3", "--seed", "1"]
    result = _invoke(args=args)
    assert result.exit_code == 0
    left = ["c__d-2"] if scored else ["c__d-2", "e__f-3"]
    assert f"finishing {name}: skipping {3 - len(left)} finished tasks, running {len(left)}" in result.output
    assert sorted(h.ran) == left and scores == [(name, 2, True)]
    finished = runs.read_node(name=name)
    assert {iid: run.status for iid, run in finished.runs.items()} == {
        "a__b-1": "ok",
        "c__d-2": "ok",
        "e__f-3": "timeout" if scored else "ok",
    }


def test_a_scored_run_runs_again_only_under_another_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    for seed in (1, 2):
        name = f"gemma-4-12b--pi--swebench-verified--baseline--1t--s{seed}"
        runs.write_node(node=_parent_node().model_copy(update={"name": name, "seed": seed, "total": 1}))
    name = "gemma-4-12b--pi--swebench-verified--baseline--1t--s1"
    args = ["run", "--model", "gemma-4-12b", "--benchmark", "swebench-verified", "--tasks", "1", "--seed", "1"]
    result = _invoke(args=args)
    assert result.exit_code == 1
    assert f"{name} already ran: 0/1 resolved; pass --seed 3 for another run of it" in result.output
    named = _invoke(args=[*args, "--node", name])
    assert named.exit_code == 1 and f"{name} already ran: 0/1 resolved\n" in named.output


def test_the_baseline_needs_a_model_and_a_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    for given in (["--model", "gemma-4-12b"], ["--benchmark", "swebench-verified"]):
        result = _invoke(args=["run", *given, "--tasks", "1", "--seed", "1"])
        assert result.exit_code == 1 and "the baseline needs --model and --benchmark" in result.output


def test_run_probe_requires_scored_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    runs.write_node(node=_parent_node())
    (tmp_path / "variants" / "probe-v").mkdir(parents=True)
    cli.variants.tree_register(
        name="probe-v", parent="baseline", parent_node="t-node", card=_card(title="Probe"), diff="+ a.md"
    )
    args = [
        "run",
        "--benchmark",
        "swebench-verified",
        "--model",
        "gemma-4-12b",
        "--variant",
        "probe-v",
        "--probe",
        "--seed",
        "1",
        "--node",
        "probe-v-node",
        "--tasks",
        "1",
    ]
    result = _invoke(args=args)
    assert result.exit_code == 1
    assert "no scored tasks" in result.output


def _scored_root() -> core.Node:
    root = _parent_node()
    root.runs = {
        iid: core.TaskRun(instance_id=iid, status="ok", resolved=ok)
        for iid, ok in [("a__b-1", False), ("c__d-2", True), ("e__f-3", True)]
    }
    root.total, root.resolved = 3, 2
    runs.write_node(node=root)
    return root


def test_probe_set_is_fixed_by_the_lineage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _made_up_ids(monkeypatch=monkeypatch)
    _scored_root()
    child = _successor(tmp_path=tmp_path, name="v1", outcomes=[True])
    (tmp_path / "variants" / "v1").mkdir(parents=True)
    child.dataset = benchmarks.get(name="swebench-verified").dataset
    child.runs = {"a__b-1": core.TaskRun(instance_id="a__b-1", resolved=True)}
    runs.write_node(node=child)
    (tmp_path / "variants" / "v2").mkdir(parents=True)
    cli.variants.tree_register(name="v2", parent="v1", parent_node=child.name, card=_card(title="V2"), diff="+ a.md")
    monkeypatch.setattr(
        cli.models, "get", lambda name: core.ModelSpec(name="gemma-4-12b", source="path:/tmp/x.gguf", model_id="m")
    )
    _fake_world(monkeypatch=monkeypatch)

    def stop(
        node: core.Node,
        h: harness.Harness,
        backend: core.ModelBackend,
        n_concurrent: int,
        instance_ids: list[str] | None,
    ) -> None:
        cli._stop(reason=f"stop before running {len(instance_ids or [])} tasks")

    monkeypatch.setattr(cli, "run_tasks", stop)
    args = ["run", "--benchmark", "swebench-verified", "--model", "gemma-4-12b", "--variant", "v2", "--probe"]
    result = _invoke(args=args + ["--seed", "1"])
    assert result.exit_code == 1 and "`rsi probe --root t-node" in result.output
    written = _invoke(args=["probe", "--root", "t-node"])
    assert written.exit_code == 0 and "the teacher never sees 1 of them" in written.output
    result = _invoke(args=args + ["--seed", "1"])
    assert "stop before running 3 tasks" in result.output
    probe = (paths.node_dir(name="t-node") / "probe.txt").read_text().splitlines()
    assert probe[0].startswith("# probe set of the lineage rooted at t-node: 3 tasks")
    assert sorted(probe[1:]) == ["a__b-1", "c__d-2", "e__f-3"]
    created = runs.read_node(name="gemma-4-12b--pi--swebench-verified--v2--3t--s1")
    assert created.parent == child.name and created.tasks == 3 and created.server == SERVER


def test_the_probe_set_counts_older_runs_and_is_written_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _made_up_ids(monkeypatch=monkeypatch)
    root = _parent_node()
    root.runs = {
        iid: core.TaskRun(instance_id=iid, status="ok", resolved=ok)
        for iid, ok in [("hard-1", False), ("flip-2", False), ("easy-3", True)]
    }
    root.total, root.resolved = 3, 1
    runs.write_node(node=root)
    archive = tmp_path / "pilot" / "runs"
    for i in range(8):
        older = root.model_copy(update={"name": f"older-{i}", "created": float(i)})
        older.runs = {
            "hard-1": core.TaskRun(instance_id="hard-1", status="ok", resolved=False),
            "flip-2": core.TaskRun(instance_id="flip-2", status="ok", resolved=i == 0),
        }
        (archive / older.name).mkdir(parents=True)
        (archive / older.name / "node.json").write_text(older.model_dump_json())
    child = root.model_copy(update={"name": "child", "parent": "t-node"})
    runs.write_node(node=child)
    assert "isn't a lineage's root" in _invoke(args=["probe", "--root", "child"]).output
    assert "no node records" in _invoke(args=["probe", "--root", "t-node", "--history", str(tmp_path)]).output
    result = _invoke(args=["probe", "--root", "t-node", "--history", str(archive), "--history", f"{archive}/"])
    assert result.exit_code == 0
    assert "left out, failed in 8+ runs and never solved: hard-1" in result.output
    assert runs.lineage_probe(root=root) == ["flip-2", "easy-3"]
    result = _invoke(args=["probe", "--root", "t-node"])
    assert result.exit_code == 1 and "written already" in result.output


def test_the_probe_set_can_be_carried_over_as_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    root = _parent_node()
    root.runs = {iid: core.TaskRun(instance_id=iid, status="ok", resolved=True) for iid in ("a-1", "b-2", "c-3")}
    root.total, root.resolved = 3, 3
    runs.write_node(node=root)
    (tmp_path / "probe.txt").write_text("# an older lineage's probe set\na-1\nb-2\n")
    (tmp_path / "hidden.txt").write_text("b-2\n")
    given = ["--tasks", str(tmp_path / "probe.txt"), "--hidden", str(tmp_path / "hidden.txt")]
    assert "go together" in _invoke(args=["probe", "--root", "t-node", given[0], given[1]]).output
    assert "--history only counts" in _invoke(args=["probe", "--root", "t-node", *given, "--history", "x"]).output
    result = _invoke(args=["probe", "--root", "t-node", *given])
    assert result.exit_code == 0 and "the teacher never sees 1 of them" in result.output
    assert runs.lineage_probe(root=root) == ["a-1", "b-2"] and runs.lineage_hidden(root=root) == {"b-2"}


def test_run_variant_not_in_tree_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    (tmp_path / "variants" / "stray").mkdir(parents=True)
    args = [
        "run",
        "--benchmark",
        "swebench-verified",
        "--model",
        "gemma-4-12b",
        "--variant",
        "stray",
        "--seed",
        "1",
        "--node",
        "stray-node",
        "--tasks",
        "1",
    ]
    result = _invoke(args=args)
    assert result.exit_code == 1
    assert "not in the tree" in result.output


def test_make_node_rejects_overlong_name(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(typer.Exit):
        cli.make_node(
            name="a" * 201,
            dataset="swebench-verified",
            harness_name="pi",
            model="gemma-4-12b",
            tasks=10,
            seed=42,
            timeout=3600,
            backend=_FakeBackend(),
        )
    assert "under 200" in caplog.text


def test_cli_logs_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    root = logging.getLogger()
    before = list(root.handlers)
    cli._setup_logging()
    try:
        files = [Path(h.baseFilename) for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert [f.parent for f in files] == [tmp_path / "logs"] and files[0].exists()
    finally:
        root.handlers[:] = before


def test_node_commands_log_into_node_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    cli._retarget_logs(node_name="t-node")
    expected = tmp_path / "runs" / "t-node" / "rsi.log"
    handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
    assert any(Path(h.baseFilename) == expected for h in handlers)
    assert expected.exists()


def test_one_process_at_a_time_uses_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    model = core.ModelSpec(name="gemma-4-12b", source="path:/tmp/x.gguf", model_id="m", port=11999)
    lock = tmp_path / "models" / "locks" / "11999.lock"
    with cli._model_lock(model=model):
        assert lock.read_text() == f"{os.getpid()}\n"
    with lock.open("a+") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write_text("4242\n")
        with pytest.raises(typer.Exit), cli._model_lock(model=model):
            pass
    assert "gemma-4-12b (port 11999) is in use by another rsi process, pid 4242" in caplog.text
    with cli._model_lock(model=model):
        pass


def test_compare_pools_two_sets_of_runs_on_the_tasks_they_share(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    tasks = [f"t__t-{i}" for i in range(4)]

    def scored(name: str, solved: set[str], server: dict[str, str], extra: bool) -> None:
        node = _parent_node().model_copy(update={"name": name, "server": server, "contract": CONTRACT})
        node.runs = {t: core.TaskRun(instance_id=t, status="ok", resolved=t in solved) for t in tasks}
        if extra:
            node.runs["extra"] = core.TaskRun(instance_id="extra", status="ok", resolved=True)
        node.total, node.resolved = len(node.runs), sum(run.resolved for run in node.runs.values())
        runs.write_node(node=node)

    scored(name="base-1", solved={"t__t-0"}, server=SERVER, extra=True)
    scored(name="base-2", solved={"t__t-0"}, server=SERVER, extra=True)
    scored(name="evolved-1", solved={"t__t-0", "t__t-1", "t__t-2"}, server={"engine": "fake 1.1"}, extra=False)
    cli.compare(a=["base-1", "base-2"], b=["evolved-1"])
    assert [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()] == [
        "solves on the 4 tasks every run has",
        "a base-1 1",
        "a base-2 1",
        "a mean 1.0",
        "b evolved-1 3",
        "b mean 3.0",
        "b against a: +2.0 ± 1.0 tasks per run, better, beyond noise (1 run vs 2 of a, 4 tasks)",
    ]
    assert "not the same model server" in caplog.text


def _lineage(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend | None = None, h: _FakeHarness | None = None
) -> tuple[core.Node, list[str]]:
    monkeypatch.setattr(runs, "PROBE_SUCCESSES", 1)
    _copy_models_toml(root=paths.rsi_root())
    _fake_world(monkeypatch=monkeypatch, backend=backend, h=h)
    now = {"model": models.get(name="gemma-4-12b"), "dataset": benchmarks.get(name="swebench-verified").dataset}
    root = _parent_node().model_copy(update={**now, "server": SERVER, "contract": CONTRACT})
    tasks = [f"t__t-{i}" for i in range(6)]
    root.runs = {iid: core.TaskRun(instance_id=iid, status="ok", resolved=i >= 4) for i, iid in enumerate(tasks)}
    root.task_ids, root.tasks, root.total, root.resolved = tasks, 6, 6, 2
    runs.write_node(node=root)
    return root, runs.write_probe(root=root, nodes=[root], repos=_one_repo(node=root))


def _probe_node(root: core.Node, probe: list[str], node: str, variant: str, seed: int) -> None:
    run = root.model_copy(
        update={
            "name": node,
            "parent": root.name,
            "variant": variant,
            "tasks": len(probe),
            "seed": seed,
            "server": SERVER,
            "contract": CONTRACT,
            "variant_digest": cli._variant_digest(
                package=variants.resolve_variant(reference=variant) if variant else None
            ),
        }
    )
    run.runs = {iid: core.TaskRun(instance_id=iid, status="ok") for iid in probe}
    run.task_ids, run.total = probe, len(probe)
    runs.write_node(node=run)


def _register(names: tuple[str, ...], parent: str) -> None:
    for name in names:
        (paths.variants_dir() / name).mkdir(parents=True)
        cli.variants.tree_register(name=name, parent="baseline", parent_node=parent, card=_card(title=name), diff="+")


class _Started:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ran: list[tuple[str, list[str] | None]] = []
        self.scored: list[tuple[str, int, bool]] = []
        monkeypatch.setattr(cli, "run_tasks", self.run_tasks)
        monkeypatch.setattr(cli, "score_node", self.score_node)

    def run_tasks(
        self,
        node: core.Node,
        h: harness.Harness,
        backend: core.ModelBackend,
        n_concurrent: int,
        instance_ids: list[str] | None,
    ) -> None:
        self.ran.append((node.name, instance_ids))

    def score_node(self, name: str, workers: int, clone_tasks: bool) -> None:
        self.scored.append((name, workers, clone_tasks))


def test_a_variant_runs_with_its_parents_model_benchmark_and_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _copy_models_toml(root=tmp_path)
    _fake_world(monkeypatch=monkeypatch)
    started = _Started(monkeypatch=monkeypatch)
    parent = {"harness": "pi-next", "model": models.get(name="qwen-3.8-27b"), "total": 1}
    runs.write_node(node=_parent_node().model_copy(update=parent))
    _register(names=("tight",), parent="t-node")
    result = _invoke(args=["run", "--variant", "tight", "--tasks", "2", "--seed", "1"])
    assert result.exit_code == 0
    made = runs.read_node(name="qwen-3.8-27b--pi-next--swebench-verified--tight--2t--s1")
    assert (made.model.name, made.dataset.name, made.parent) == ("qwen-3.8-27b", "swebench-verified", "t-node")
    assert made.variant == "tight"
    assert started.ran == [(made.name, None)] and started.scored == [(made.name, 2, True)]
    _register(names=("orphan",), parent="gone")
    result = _invoke(args=["run", "--variant", "orphan", "--tasks", "2", "--seed", "1"])
    assert result.exit_code == 1 and "node 'gone' not found; known nodes: " in result.output


def test_a_baseline_probe_run_is_made_from_the_one_root_with_a_probe_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    root, probe = _lineage(monkeypatch=monkeypatch)
    started = _Started(monkeypatch=monkeypatch)
    unprobed = root.model_copy(update={"name": "unprobed", "seed": 2})
    runs.write_node(node=unprobed)
    other = root.model_copy(update={"name": "other-model", "model": root.model.model_copy(update={"name": "qwen"})})
    runs.write_node(node=other)
    runs.write_probe(root=other, nodes=[other], repos=_one_repo(node=other))
    args = ["run", "--model", "gemma-4-12b", "--benchmark", "swebench-verified", "--probe", "--no-score"]
    result = _invoke(args=[*args, "--seed", "43"])
    assert result.exit_code == 0
    made = runs.read_node(name=f"gemma-4-12b--pi--swebench-verified--baseline--{len(probe)}t--s43")
    assert (made.parent, made.variant, made.tasks) == ("t-node", "", len(probe))
    assert started.ran == [(made.name, probe)] and started.scored == []
    runs.write_probe(root=unprobed, nodes=[unprobed], repos=_one_repo(node=unprobed))
    result = _invoke(args=[*args, "--seed", "44"])
    assert result.exit_code == 1
    assert "and more than one (t-node, unprobed) of gemma-4-12b on swebench-verified has one" in result.output
    for node in (root, unprobed):
        runs.probe_path(root=node).unlink()
    result = _invoke(args=[*args, "--seed", "44"])
    assert result.exit_code == 1 and "and none of gemma-4-12b on swebench-verified has one" in result.output
    assert len(started.ran) == 1


def test_a_baseline_probe_run_leaves_out_roots_made_with_another_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    root, _ = _lineage(monkeypatch=monkeypatch)
    started = _Started(monkeypatch=monkeypatch)
    asked = root.dataset.model_copy(update={"request": "{task}"})
    older = root.model_copy(update={"name": "older", "seed": 2, "dataset": asked})
    runs.write_node(node=older)
    runs.write_probe(root=older, nodes=[older], repos=_one_repo(node=older))
    args = ["run", "--model", "gemma-4-12b", "--benchmark", "swebench-verified", "--probe", "--no-score"]
    assert _invoke(args=[*args, "--seed", "43"]).exit_code == 0
    assert runs.read_node(name=started.ran[0][0]).parent == root.name


@pytest.mark.parametrize("changed", ["nothing", "server", "contract", "files", "request", "timeout", "model"])
def test_a_run_is_not_spent_when_it_would_not_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    backend = _FakeBackend(server={"engine": "fake 1.1"}) if changed == "server" else None
    h = _FakeHarness(contract="contract-2") if changed == "contract" else None
    root, probe = _lineage(monkeypatch=monkeypatch, backend=backend, h=h)
    if changed == "request":
        root = root.model_copy(update={"dataset": root.dataset.model_copy(update={"request": "{task}"})})
        runs.write_node(node=root)
    started = _Started(monkeypatch=monkeypatch)
    for seed in (2, 3):
        _probe_node(root=root, probe=probe, node=f"repeat-s{seed}", variant="", seed=seed)
    _register(names=("next",), parent="t-node")
    package = paths.variants_dir() / "next"
    _probe_node(root=root, probe=probe, node="next-s1", variant="next", seed=1)
    if changed == "files":
        (package / "SYSTEM.md").write_text("Edited after its first run.\n")
    timeout = ["--timeout", "1800"] if changed == "timeout" else []
    # the model the run asks for, not its parent's
    model = ["--model", "qwen-3.8-27b"] if changed == "model" else []
    result = _invoke(args=["run", "--variant", "next", "--probe", "--seed", "2", *timeout, *model])
    used = "qwen-3.8-27b" if changed == "model" else "gemma-4-12b"
    name = f"{used}--pi--swebench-verified--next--{len(probe)}t--s2"
    if changed == "nothing":
        assert result.exit_code == 0 and started.ran == [(name, probe)]
        return
    assert result.exit_code == 1 and started.ran == [] and not paths.node_dir(name=name).exists()
    field = {
        "server": "model server",
        "contract": "harness contract",
        "files": "package files",
        "request": "dataset",
        "timeout": "time budget",
        "model": "model settings",
    }[changed]
    reason = f"this run wouldn't compare with the runs it's judged with, so it isn't started: not the same {field}: "
    assert reason in result.output


def test_a_first_probe_run_is_checked_against_its_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _lineage(monkeypatch=monkeypatch, backend=_FakeBackend(server={"engine": "fake 1.1"}))
    started = _Started(monkeypatch=monkeypatch)
    args = ["run", "--model", "gemma-4-12b", "--benchmark", "swebench-verified", "--probe", "--seed", "2"]
    result = _invoke(args=args)
    assert result.exit_code == 1 and started.ran == []
    assert "not the same model server" in result.output
