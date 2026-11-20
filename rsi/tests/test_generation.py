import contextlib
import json
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from rsi import benchmarks, cards, core, folders, generation, harness, paths, runs, teacher, variants


def _parent_node() -> core.Node:
    node = core.Node(
        name="t-node",
        dataset=core.DatasetSpec(name="swebench-verified", dataset_id="princeton-nlp/SWE-bench_Verified"),
        harness="pi",
        harness_version="1.3.0",
        model=core.ModelSpec(name="qwen-3.8-27b", source="path:/tmp/x.gguf", model_id="m"),
        budget=core.Budget(),
        tasks=2,
        seed=1,
    )
    node.runs["a__b-1"] = core.TaskRun(instance_id="a__b-1", status="ok", seconds=60.0, resolved=True, patch="+fix\n")
    node.runs["c__d-2"] = core.TaskRun(instance_id="c__d-2", status="timeout", seconds=3600.0)
    node.total, node.resolved = 2, 1
    return node


def _card(title: str, what: str = "w", reuses: tuple[str, ...] = ()) -> str:
    return json.dumps({"title": title, "what": what, "why": "y", "expect": "e", "reuses": list(reuses)})


def _hide(record: core.Node, task: str = "z__z-9") -> None:
    """Write the probe set as the root's tasks plus `task`, and hide `task` from the teacher"""
    runs.probe_path(root=record).write_text("\n".join(sorted({*record.runs, task})) + "\n")
    runs.hidden_path(root=record).write_text(f"# hidden from the teacher\n{task}\n")


def _alias(task: str) -> str:
    return generation.task_alias(task=task, kind="")


def test_stage_holds_the_raw_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    record.kinds = {"a__b-1": "bug fixes", "c__d-2": "data analysis"}
    record.runs["a__b-1"].metrics = core.TaskMetrics(requests=2, output_tokens=4000, stop_reason="stop")
    record.runs["c__d-2"].error = "container rsi-c__d-2 went away"
    runs.write_node(node=record)
    _hide(record=record)
    for iid in record.runs:
        folder = paths.instances_dir(node_name="t-node") / iid
        (folder / "session").mkdir(parents=True)
        (folder / "session" / f"2026-11-06T21-19-02Z_{iid}.jsonl").write_text('{"type": "session"}\n')
        (folder / "metrics.jsonl").write_text('{"type": "context"}\n')
        (folder / "events.jsonl").write_text("{}\n")
        (folder / "result.json").write_text('{"resolved": true}\n')
        # the agent could leave symlinks to the score (result.json) or to a missing file
        (folder / "session" / "x.jsonl").symlink_to(folder / "result.json")
        (folder / "session" / "y.jsonl").symlink_to(folder / "missing")
    again = record.model_copy(update={"name": "again", "parent": "t-node", "created": record.created + 1})
    again.runs = {"a__b-1": core.TaskRun(instance_id="a__b-1", status="ok", seconds=120.0, resolved=False)}
    pi = harness.load(name="pi")
    out = generation.stage(record=record, h=pi, dest=tmp_path / "stage", nodes=[record, again])
    assert (out / "harness.md").read_text() == pi.doc.read_text()
    assert (out / "parent" / "package.json").is_file() and not (out / "parent" / cards.FILE).exists()
    first = json.loads((out / "runs" / "parent" / "1" / "run.json").read_text())
    assert (first["model"]["name"], first["model"]["ctx"]) == ("qwen-3.8-27b", record.model.ctx)
    # task ids name the benchmark, so the teacher sees each task as its kind of work and a hash
    fix, data = (generation.task_alias(task=iid, kind=record.kinds[iid]) for iid in ("a__b-1", "c__d-2"))
    assert re.fullmatch(r"bug-fixes-[0-9a-f]{8}", fix) and data.startswith("data-analysis-")
    assert list(first["tasks"]) == [fix, data]
    assert first["tasks"][fix] == {
        "status": "ok",
        "seconds": 60.0,
        "patch": "+fix\n",
        "error": None,
        "metrics": record.runs["a__b-1"].metrics.model_dump(),
    }
    assert first["tasks"][data]["error"] == f"container rsi-{data} went away"
    # stage copies pi's session log and the metrics, but nothing that says whether the run passed
    task = out / "runs" / "parent" / "1" / fix
    assert sorted(p.name for p in task.iterdir()) == ["metrics.jsonl", "session.jsonl"]
    assert list(json.loads((out / "runs" / "parent" / "2" / "run.json").read_text())["tasks"]) == [fix]
    assert not [p for p in out.rglob("*") if "a__b-1" in p.name or "c__d-2" in p.name]
    with pytest.raises(FileExistsError):
        generation.stage(record=record, h=pi, dest=out, nodes=[record])


def test_the_parents_runs_are_every_run_of_its_harness_in_the_lineage_oldest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    record.created = 3.0
    runs.write_node(node=record)
    _hide(record=record)

    def repeat(name: str, created: float, statuses: dict[str, str], variant: str = "") -> core.Node:
        update = {"name": name, "parent": "t-node", "created": created, "variant": variant, "seed": int(created) + 1}
        node = record.model_copy(update=update)
        node.runs = {
            iid: core.TaskRun(instance_id=iid, status=status, seconds=created * 60) for iid, status in statuses.items()
        }
        return node

    older = repeat(name="older", created=1.0, statuses={"a__b-1": "error", "c__d-2": "ok"})
    # x__y-7 isn't one of the root's tasks (e.g. a held-out task), so it stays out
    newer = repeat(name="newer", created=2.0, statuses={"a__b-1": "ok", "x__y-7": "ok"})
    # leave newer unscored, since the teacher doesn't need scores
    newer.total, newer.resolved = 0, 0
    other = repeat(name="other", created=4.0, statuses={"a__b-1": "ok", "c__d-2": "ok"}, variant=str(tmp_path / "v"))
    # a run with no parent starts its own lineage, so it isn't in this one
    apart = repeat(name="apart", created=5.0, statuses={"a__b-1": "ok"}).model_copy(update={"parent": None})
    # a run on none of the root's tasks has nothing to show, so it gets no folder
    elsewhere = repeat(name="elsewhere", created=6.0, statuses={"x__y-8": "ok"})
    out = generation.stage(
        record=record,
        h=_FakeHarness(),
        dest=tmp_path / "stage",
        nodes=[record, older, newer, other, apart, elsewhere],
    )
    staged = [json.loads((out / "runs" / "parent" / str(k) / "run.json").read_text()) for k in (1, 2, 3)]
    ab, cd = _alias(task="a__b-1"), _alias(task="c__d-2")
    assert [run["tasks"][ab]["status"] for run in staged] == ["error", "ok", "ok"]
    assert [sorted(run["tasks"]) for run in staged] == [sorted([ab, cd]), [ab], sorted([ab, cd])]
    # other ran a different variant, so it isn't one of the parent's runs
    assert not (out / "runs" / "parent" / "4").exists()


def test_the_teacher_sees_only_the_probe_tasks_the_lineage_doesnt_hide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    record.runs["e__f-3"] = core.TaskRun(instance_id="e__f-3", status="ok", resolved=True)
    # g__h-4 isn't in the probe set. The probe picks as many solved tasks as unsolved ones, so showing the tasks it
    # skipped would hint at how they went
    record.runs["g__h-4"] = core.TaskRun(instance_id="g__h-4", status="ok")
    record.total, record.resolved = 4, 2
    runs.write_node(node=record)
    _hide(record=record, task="a__b-1")
    runs.probe_path(root=record).write_text("a__b-1\nc__d-2\ne__f-3\n")
    again = record.model_copy(update={"name": "again", "parent": "t-node", "created": 2.0})
    again.runs = {
        iid: core.TaskRun(instance_id=iid, status="ok", resolved=not run.resolved) for iid, run in record.runs.items()
    }
    for node in (record, again):
        for iid in node.runs:
            (paths.instances_dir(node_name=node.name) / iid).mkdir(parents=True)
            (paths.instances_dir(node_name=node.name) / iid / "session.jsonl").write_text(f'{{"task": "{iid}"}}\n')
    out = generation.stage(record=record, h=_FakeHarness(), dest=tmp_path / "stage", nodes=[record, again])
    cd, ef = _alias(task="c__d-2"), _alias(task="e__f-3")
    assert sorted(p.relative_to(out / "runs").as_posix() for p in (out / "runs").rglob("*.jsonl")) == sorted(
        f"parent/{k}/{alias}/session.jsonl" for k in (1, 2) for alias in (cd, ef)
    )
    staged = [p.read_text(errors="replace") for p in out.rglob("*") if p.is_file()]
    hidden = (_alias(task="a__b-1"), _alias(task="g__h-4"), "a__b-1", "g__h-4")
    assert not [text for text in staged if any(name in text for name in hidden)]


def test_stage_refuses_a_lineage_that_hides_nothing_yet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    runs.write_node(node=record)
    # no probe set yet, so stage refuses, since a task hidden later would already have been shown
    with pytest.raises(ValueError, match="no tasks hidden from the teacher yet"):
        generation.stage(record=record, h=_FakeHarness(), dest=tmp_path / "stage", nodes=[record])
    _hide(record=record)
    # the hidden list is kept with the root, so stage refuses when the root is missing
    record.parent = "archived-root"
    with pytest.raises(ValueError, match="no tasks hidden from the teacher yet"):
        generation.stage(record=record, h=_FakeHarness(), dest=tmp_path / "stage", nodes=[record])
    assert not (tmp_path / "stage").exists()


def test_stage_shows_how_the_parents_children_ran_on_the_tasks_it_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    runs.write_node(node=record)
    _hide(record=record, task="a__b-1")
    card = cards.VariantCard(title="check first", what="w")
    variants.tree_register(name="tight", parent="baseline", parent_node="t-node", card=card, diff="+ a.md")
    variants.tree_register(name="idle", parent="baseline", parent_node="t-node", card=card, diff="+ b.md")
    child = _scored_run(
        name="child",
        variant=str(paths.variants_dir() / "tight"),
        outcomes={"a__b-1": True, "c__d-2": False},
        created=2.0,
    )
    # a variant made from another run of the parent's variant (`again`) counts as a child too
    again = _scored_run(name="again", variant="", outcomes={"a__b-1": True, "c__d-2": True}, created=3.0)
    variants.tree_register(name="later", parent="baseline", parent_node="again", card=card, diff="+ c.md")
    later = _scored_run(
        name="later-run", variant=str(paths.variants_dir() / "later"), outcomes={"c__d-2": True}, created=4.0
    ).model_copy(update={"parent": "again"})
    nodes = [record, child, again, later]
    out = generation.stage(record=record, h=_FakeHarness(), dest=tmp_path / "stage", nodes=nodes)
    # idle hasn't run yet, so it gets no folder
    assert sorted(p.name for p in (out / "runs").iterdir()) == ["later", "parent", "tight"]
    assert list(json.loads((out / "runs" / "tight" / "1" / "run.json").read_text())["tasks"]) == [_alias(task="c__d-2")]
    assert sorted(p.name for p in (out / "runs" / "parent").iterdir()) == ["1", "2"]


def test_the_teacher_never_sees_which_runs_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    runs.write_node(node=record)
    _hide(record=record)
    flipped = record.model_copy(update={"name": "flipped", "parent": "t-node", "created": record.created + 1})
    flipped.runs = {iid: run.model_copy(update={"resolved": not run.resolved}) for iid, run in record.runs.items()}
    out = generation.stage(record=record, h=_FakeHarness(), dest=tmp_path / "stage", nodes=[record, flipped])
    staged = [p.read_text().lower() for p in (out / "runs").rglob("run.json")]
    # nor node names or the dataset name, since in real runs both give away the benchmark
    words = ("resolved", "score", "swebench", "t-node", "flipped")
    assert len(staged) == 2 and [word for text in staged for word in words if word in text] == []


def test_every_file_a_prompt_names_is_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    runs.write_node(node=record)
    _hide(record=record)
    out = generation.stage(record=record, h=_FakeHarness(), dest=tmp_path / "stage", nodes=[record])
    generation.refresh_tree(dest=out, entries={}, nodes=[record])
    prompt = generation.prompt_text(harness_name="pi", model_name="m", parent="baseline", work="w", timeout_s=60)
    named = [line.split("`")[1].split("<")[0] for line in prompt.splitlines() if line.startswith("- `")]
    assert named and [name for name in named if not (out / name).exists()] == []


def test_prompt_text_fills_every_placeholder() -> None:
    work = benchmarks.get(name="swebench-verified").work
    text = generation.prompt_text(
        harness_name="pi", model_name="qwen-3.8-27b", parent="tight-loop", work=work, timeout_s=3600
    )
    # the teacher looks up the parent and its variants in tree.json by this name
    assert "(`tight-loop`)" in text
    assert re.findall(r"\{[a-z_]+\}", text) == [] and "qwen-3.8-27b" in text and "You have 60 minutes" in text
    assert "runs on issues from public Python projects." in text
    # the benchmark, model and harness are swappable, so the prompt file must not name any of them
    prompt = (paths.project_dir() / "prompts" / "variant-generator.md").read_text().lower()
    assert [word for word in ("python", "qwen", "swe-bench", "swebench", " pi ") if word in prompt] == []


class _FakeHarness:
    name = "pi"
    version = "1.3.0"
    package_dir = paths.harnesses_dir() / "pi" / "package"
    doc = paths.harnesses_dir() / "pi" / "HARNESS.md"
    protocol_terms = ("exactly done",)

    def __init__(
        self,
        plan: tuple[str, ...] = (),
        idle: tuple[str, int] = ("error", 0),
    ) -> None:
        self.plan = plan
        self.idle = idle
        self.prompts: list[str] = []
        self.trees: list[str] = []
        self.asked: list[tuple[str, str]] = []

    def run_files(self, *, instance_dir: Path) -> dict[str, Path]:
        session = instance_dir / "session.jsonl"
        return {"session.jsonl": session} if session.is_file() else {}

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
        assert "exactly done" in terms and "swebench" in terms
        self.workspace = workspace
        self.teacher_out = out_dir
        yield f"rsi-teacher-{name}"

    def validate_variant(self, *, variant: Path, out_dir: Path, terms: tuple[str, ...]) -> list[str]:
        check = folders.module(root=paths.harnesses_dir(), name="pi", file="check_variant.py")
        return check.static_problems(package=variant, terms=tuple(check.hashed(term=term) for term in terms))

    def question_problems(
        self, *, variant: Path, out_dir: Path, model: core.ModelSpec, endpoint: core.ModelEndpoint
    ) -> list[str]:
        self.asked.append((variant.name, out_dir.relative_to(out_dir.parents[2]).as_posix()))
        appended = variant / "APPEND_SYSTEM.md"
        pushy = appended.is_file() and "Never stop" in appended.read_text()
        return ["the agent answered, and the package made it go on"] if pushy else []


class _FakeTeacher:
    name = "fake"

    def __init__(self, h: _FakeHarness) -> None:
        self.h = h

    def describe(self, *, model: str, effort: str) -> str:
        return f"fake teacher, {model} at {effort}"

    def transcript(self, *, session_dir: Path) -> str:
        return "## the teacher's session\n"

    def run(
        self, *, prompt: str, model: str, effort: str, container: str, out_dir: Path, timeout_s: int
    ) -> teacher.TeacherRun:
        h, workspace, produced = self.h, self.h.workspace, self.h.teacher_out / "variants"
        assert container.startswith("rsi-teacher-t-node-") and (out_dir / "prompt.md").read_text() == prompt
        assert h.teacher_out == out_dir / generation.TEACHER_OUT
        h.prompts.append(prompt)
        h.trees.append(json.loads((workspace / "tree.json").read_text()))
        assert (workspace / "runs" / "parent" / "1" / "run.json").is_file()
        if h.plan:
            action, number = h.plan[len(h.prompts) - 1], len(h.prompts)
            if action == "down":
                status, tools = h.idle
                error = "429 rate limited" if status == "error" else None
                return teacher.TeacherRun(status=status, seconds=1.0, error=error, tool_calls=tools)
            if action != "empty":
                package = produced / f"{action}-{number}"
                shutil.copytree(workspace / "parent", package)
                if action != "bad":
                    (package / "APPEND_SYSTEM.md").write_text(f"Change number {number}.\n")
                (package / cards.FILE).write_text(_card(title=f"change {number}"))
            status = "error" if action == "late" else "ok"
            return teacher.TeacherRun(status=status, seconds=1.0, model_calls=3, tool_calls=12)
        if len(h.prompts) == 1:
            good = produced / "tight-loop"
            shutil.copytree(workspace / "parent", good)
            (good / "APPEND_SYSTEM.md").write_text("Work in a tight loop.\n")
            (good / cards.FILE).write_text(_card(title="End every turn with progress", reuses=("ghost",)))
            same = produced / "no-op"
            shutil.copytree(workspace / "parent", same)
            (same / cards.FILE).write_text(_card(title="Change nothing at all"))
            leaky = produced / "leaky"
            shutil.copytree(workspace / "parent", leaky)
            (leaky / "APPEND_SYSTEM.md").write_text("When done, reply with exactly DONE.\n")
            (leaky / cards.FILE).write_text(_card(title="Name the protocol"))
            pushy = produced / "pushy"
            shutil.copytree(workspace / "parent", pushy)
            (pushy / "APPEND_SYSTEM.md").write_text("Never stop at an answer.\n")
            (pushy / cards.FILE).write_text(_card(title="Keep going"))
            linked = produced / "linked"
            shutil.copytree(workspace / "parent", linked)
            (linked / "SYSTEM.md").symlink_to(paths.node_dir(name="t-node") / "node.json")
            # card-link's card is a symlink to a secret on the host, which the next session must not see
            secret = paths.node_dir(name="t-node") / "host-secret.txt"
            secret.write_text("TOKEN=abc123\n")
            card_link = produced / "card-link"
            shutil.copytree(workspace / "parent", card_link)
            (card_link / "APPEND_SYSTEM.md").write_text("Read the card.\n")
            (card_link / cards.FILE).symlink_to(secret)
            (produced / "outside").symlink_to(workspace / "parent", target_is_directory=True)
        if len(h.prompts) == 2:
            essay = produced / "Keep A Copy"
            shutil.copytree(workspace / "parent", essay)
            (essay / "SYSTEM.md").write_text("Keep a pristine copy before editing.\n")
            (essay / cards.FILE).write_text(_card(title="keep a pristine copy", what="Keep a copy."))
        return teacher.TeacherRun(status="ok", seconds=1.0, model_calls=3, tool_calls=12, cost="1 AI credit")


def test_regular_file_takes_no_file_a_symlink_leads_to(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "session").mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("a file on the host machine\n")
    (run / "session" / "a.jsonl").write_text("{}\n")
    assert generation.regular_file(path=run / "session" / "a.jsonl", folder=run)
    (run / "session" / "link.jsonl").symlink_to(secret)
    (run / "elsewhere").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(run / "pipe")
    for path in (run / "session" / "link.jsonl", run / "elsewhere" / "secret.txt", run / "pipe", run / "missing"):
        assert not generation.regular_file(path=path, folder=run)
    assert not generation.regular_file(path=secret, folder=run)


def _collect(produced: Path, checks: Path) -> list[variants.VariantEntry]:
    record = _parent_node()
    runs.write_node(node=record)
    return generation.collect(
        produced=produced,
        record=record,
        parent_package=_FakeHarness.package_dir,
        created_by="teacher:opus",
        round="r1",
        session=1,
        h=_FakeHarness(),
        terms=("swebench",),
        checks=checks,
        endpoint=core.ModelEndpoint(model_id="m", base_url="http://fake"),
    )


def _candidate(folder: Path, rule: str) -> None:
    shutil.copytree(_FakeHarness.package_dir, folder)
    if rule:
        (folder / "APPEND_SYSTEM.md").write_text(f"{rule}\n")
    (folder / cards.FILE).write_text(_card(title=folder.name))


def test_collect_takes_no_candidate_from_a_folder_a_symlink_leads_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    elsewhere = tmp_path / "elsewhere"
    _candidate(folder=elsewhere / "kept", rule="Work in a tight loop.")
    produced = tmp_path / "out" / "variants"
    produced.parent.mkdir()
    produced.symlink_to(elsewhere, target_is_directory=True)
    assert _collect(produced=produced, checks=tmp_path / "checks") == []
    assert (elsewhere / "kept" / cards.FILE).is_file()


def test_collect_keeps_none_when_a_session_hands_in_more_than_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    produced = tmp_path / "out" / "variants"
    _candidate(folder=produced / "tight-loop", rule="Work in a tight loop.")
    _candidate(folder=produced / "read-first", rule="Read a file before you edit it.")
    # a draft that fails the checks doesn't count
    _candidate(folder=produced / "draft", rule="")
    assert _collect(produced=produced, checks=tmp_path / "checks") == []
    for name in ("read-first", "tight-loop"):
        assert json.loads((tmp_path / "checks" / name / generation.PROBLEMS).read_text()) == [
            "the session handed in 2 variants that pass the checks (read-first, tight-loop), not one"
        ]
    assert "identical to the parent" in (tmp_path / "checks" / "draft" / generation.PROBLEMS).read_text()
    assert variants.tree_load() == {}
    _candidate(folder=tmp_path / "again" / "tight-loop", rule="Work in a tight loop.")
    (registered,) = _collect(produced=tmp_path / "again", checks=tmp_path / "checks-again")
    assert registered.name == "tight-loop"


def _generate(fake: _FakeHarness, sessions: int, monkeypatch: pytest.MonkeyPatch) -> generation.Round:
    record = _parent_node()
    runs.write_node(node=record)
    _hide(record=record)
    monkeypatch.setattr(generation.teacher, "load", lambda name: _FakeTeacher(h=fake))
    return generation.generate(
        record=record,
        h=fake,
        teacher_spec=core.TeacherSpec(name="opus", agent="fake", model="m", effort="max"),
        endpoint=core.ModelEndpoint(model_id="m", base_url="http://fake"),
        sessions=sessions,
        timeout_s=1,
        round_id="r1",
    )


def test_generate_registers_good_candidates_and_rejects_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    fake = _FakeHarness()
    found = _generate(fake=fake, sessions=2, monkeypatch=monkeypatch)
    out, registered = found.out, found.registered
    assert out.parent == paths.node_dir(name="t-node") / "generations"
    first = out / "session-1"
    assert (first / "transcript.md").read_text() == "## the teacher's session\n"
    assert [e.name for e in registered] == ["tight-loop", "keep-a-copy"]
    entry = variants.tree_load()["tight-loop"]
    assert (entry.title, entry.parent, entry.parent_node) == ("End every turn with progress", "baseline", "t-node")
    assert entry.created_by == "teacher:opus"
    assert (entry.round, entry.session) == (out.name, 1)
    assert entry.diff == "+ APPEND_SYSTEM.md"
    assert (paths.variants_dir() / "tight-loop" / cards.FILE).is_file()
    essay = variants.tree_load()["keep-a-copy"]
    assert (essay.title, essay.what) == ("keep a pristine copy", "Keep a copy.")
    assert (paths.variants_dir() / "keep-a-copy" / "SYSTEM.md").is_file()
    assert (first / generation.TEACHER_OUT / "variants" / "no-op").is_dir()
    checks = out / "checks" / "session-1"
    assert "identical to the parent's" in (checks / "no-op" / generation.PROBLEMS).read_text()
    assert "exactly done" in (checks / "leaky" / generation.PROBLEMS).read_text()
    pushy = json.loads((checks / "pushy" / generation.PROBLEMS).read_text())
    assert pushy == ["the agent answered, and the package made it go on"]
    # only candidates that pass the other checks get asked the question
    assert fake.asked == [
        ("pushy", "session-1/pushy/question"),
        ("tight-loop", "session-1/tight-loop/question"),
        ("Keep A Copy", "session-2/Keep A Copy/question"),
    ]
    linked = json.loads((checks / "linked" / generation.PROBLEMS).read_text())
    assert linked[0].startswith("SYSTEM.md is a symlink")
    assert "TOKEN" not in json.dumps(fake.trees[1])
    assert runs.read_node(name="t-node").name == "t-node"
    assert not (paths.variants_dir() / "outside").exists()
    assert entry.reuses == []
    # session 2 sees the variants session 1 made, as tree.json records them
    assert fake.trees[0] == {}
    seen = fake.trees[1]["tight-loop"]
    assert (seen["title"], seen["parent"], seen["outcome"], seen["kicked_in"]) == (
        "End every turn with progress",
        "baseline",
        "not run",
        "",
    )
    assert sorted(p.name for p in out.glob("session-*")) == ["session-1", "session-2"]
    assert json.loads((out / generation.ROUND).read_text())["sessions"] == 2


def test_generate_runs_its_sessions_in_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    fake = _FakeHarness(plan=("empty", "bad", "late", "good"))
    found = _generate(fake=fake, sessions=4, monkeypatch=monkeypatch)
    assert [(e.name, e.round, e.session) for e in found.registered] == [
        ("late-3", found.out.name, 3),
        ("good-4", found.out.name, 4),
    ]
    checks = found.out / "checks" / "session-2" / "bad-2" / generation.PROBLEMS
    assert "identical to the parent's" in checks.read_text()
    assert sorted(p.name for p in found.out.glob("session-*")) == [f"session-{n}" for n in range(1, 5)]
    # each session sees the variants that earlier sessions made
    assert "late-3" not in fake.trees[2] and "late-3" in fake.trees[3]


@pytest.mark.parametrize("idle", [("error", 0), ("ok", 0), ("error", 7)])
def test_a_session_that_fails_before_doing_any_work_stops_the_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, idle: tuple[str, int]
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    fake = _FakeHarness(plan=("good", "down", "good"), idle=idle)
    with pytest.raises(generation.TeacherUnavailable) as stopped:
        _generate(fake=fake, sessions=3, monkeypatch=monkeypatch)
    (out,) = (paths.node_dir(name="t-node") / "generations").iterdir()
    reason = "429 rate limited" if idle[0] == "error" else "no tool calls"
    assert str(stopped.value) == (
        f"session 2 failed before doing any work ({reason}), so the round stops with 2 of its 3 sessions left; "
        f"the round is in {out}"
    )
    assert len(fake.prompts) == 2 and list(variants.tree_load()) == ["good-1"]


def test_a_variant_is_in_variants_before_the_tree_names_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    there: list[bool] = []
    save = variants.tree_save

    def checked(entries: dict[str, variants.VariantEntry]) -> None:
        there.append(all((paths.variants_dir() / name / cards.FILE).is_file() for name in entries))
        save(entries=entries)

    monkeypatch.setattr(variants, "tree_save", checked)
    found = _generate(fake=_FakeHarness(), sessions=2, monkeypatch=monkeypatch)
    assert len(found.registered) == 2 and there == [True, True]


def _scored_run(name: str, variant: str, outcomes: dict[str, bool], created: float) -> core.Node:
    node = _parent_node()
    node.name, node.variant, node.parent, node.created = name, variant, "t-node", created
    node.runs = {
        iid: core.TaskRun(instance_id=iid, status="ok", resolved=ok, patch="+x\n" if ok else "")
        for iid, ok in outcomes.items()
    }
    node.task_ids = sorted(outcomes)
    node.total, node.resolved = len(outcomes), sum(outcomes.values())
    return node


def test_tree_shows_how_a_variant_did_but_not_which_runs_it_flipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    parent = _parent_node()
    runs.write_node(node=parent)
    (tmp_path / "variants" / "tight-loop").mkdir(parents=True)
    (tmp_path / "variants" / "tight-loop" / cards.FILE).write_text(_card(title="End every turn with progress"))
    card = cards.VariantCard(title="End every turn with progress", what="h", why="23 of 37 failed runs")
    variants.tree_register(name="tight-loop", parent="baseline", parent_node="t-node", card=card, diff="+ a.md")
    tasks = {"a__b-1": False, "c__d-2": True, "e__f-3": True}
    repeat = _scored_run(name="repeat", variant="", outcomes=tasks, created=2.0)
    child = _scored_run(
        name="child",
        variant=str(tmp_path / "variants" / "tight-loop"),
        outcomes={"a__b-1": True, "c__d-2": False, "e__f-3": True},
        created=3.0,
    )
    for node in (repeat, child):
        runs.write_node(node=node)
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    generation.refresh_tree(dest=stage_dir, entries=variants.tree_load(), nodes=runs.list_nodes())
    assert sorted(p.name for p in stage_dir.iterdir()) == ["tree", "tree.json"]
    assert sorted(p.name for p in (stage_dir / "tree" / "tight-loop").iterdir()) == []
    text = (stage_dir / "tree.json").read_text()
    shown = json.loads(text)["tight-loop"]
    assert (shown["what"], shown["why"], shown["outcome"]) == ("h", "23 of 37 failed runs", "no clear difference")
    # no node names, since they give away the benchmark
    assert "parent_node" not in shown and "t-node" not in text
    assert "a__b-1" not in text and "c__d-2" not in text
