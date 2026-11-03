import contextlib
import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from rsi import cards, core, folders, generation, harness, paths, runs, teacher, variants


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


def _message(output: int, reasoning: int, stop: str, thinking: str) -> str:
    message = {
        "role": "assistant",
        "stopReason": stop,
        "usage": {"input": 100, "output": output, "reasoning": reasoning},
        "content": [{"type": "thinking", "thinking": thinking}],
    }
    return json.dumps({"type": "message_end", "message": message}) + "\n"


def test_stage_holds_the_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    runs.write_node(node=record)
    for iid in record.runs:
        (paths.instances_dir(node_name="t-node") / iid).mkdir(parents=True)
    task_message = {"role": "user", "content": [{"type": "text", "text": "The parser drops trailing commas."}]}
    (paths.instances_dir(node_name="t-node") / "a__b-1" / "events.jsonl").write_text(
        json.dumps({"type": "message_end", "message": task_message})
        + "\n"
        + _message(output=1000, reasoning=600, stop="toolUse", thinking="Look at the parser. " * 100)
        + _message(output=3000, reasoning=2400, stop="stop", thinking="Done.")
    )
    (paths.instances_dir(node_name="t-node") / "c__d-2" / "events.jsonl").write_text(
        _message(output=16384, reasoning=16384, stop="length", thinking="Let me recall the fix.")
    )
    context = {"type": "context", "ts": 1, "system": "You are pi, a coding agent.", "tools": [{"name": "bash"}]}
    (paths.instances_dir(node_name="t-node") / "a__b-1" / "metrics.jsonl").write_text(json.dumps(context) + "\n")
    again = record.model_copy(update={"name": "again"})
    again.runs = {
        "a__b-1": core.TaskRun(instance_id="a__b-1", status="ok", resolved=False),
        "c__d-2": core.TaskRun(instance_id="c__d-2", status="ok", resolved=True),
    }
    out = generation.stage(record=record, h=harness.load(name="pi"), dest=tmp_path / "stage", nodes=[record, again])
    assert (out / "harness.md").read_text().startswith("# How a variant runs in pi")
    assert (out / "parent" / "package.json").is_file()
    assert not (out / "parent" / cards.FILE).exists()
    told = (out / "system-prompt.md").read_text()
    assert "parent's run on a__b-1" in told and "You are pi, a coding agent." in told and '"name": "bash"' in told
    overview = (out / "overview.md").read_text()
    assert "- result: 1 of 2 tasks passed the hidden tests" in overview
    assert "| share of output tokens spent thinking | 75% | 100% | 95% |" in overview
    assert "| largest single response, tokens (median) | 3000 | 16384 | 9692 |" in overview
    assert "| runs with a response cut at the output cap | 0 | 1 | 1 |" in overview
    assert "| runs that changed no file | 0 | 1 | 1 |" in overview
    assert "| timeout | 1 | 0 |" in overview
    assert "| a__b-1 | yes | 1 of 2 | stop | 1.0 | 2 | 4000 | 75% | 3000 | 0 |" in overview
    assert "| c__d-2 | no | 1 of 2 | timeout |" in overview
    task = (out / "tasks" / "a__b-1.md").read_text()
    assert task.startswith("# a__b-1: passed")
    assert "ended stop · 1.0 min · 2 requests · 4000 output tokens (75% thinking)" in task
    assert "```diff\n+fix\n```" in task
    assert "(none: the run changed nothing)" in (out / "tasks" / "c__d-2.md").read_text()
    long_thought = ("Look at the parser. " * 100).strip()
    assert long_thought in (out / "tasks" / "full" / "a__b-1.md").read_text()
    assert long_thought not in task
    assert "task: The parser drops trailing commas.\n" in task
    assert "## task message\nThe parser drops trailing commas.\n" in (out / "tasks" / "full" / "a__b-1.md").read_text()
    assert sorted(p.name for p in (out / "tasks" / "diverged").iterdir()) == ["a__b-1.md", "c__d-2.md"]
    assert "| a__b-1 | 1 of 2 |\n| c__d-2 | 1 of 2 |" in overview
    with pytest.raises(FileExistsError):
        generation.stage(record=record, h=harness.load(name="pi"), dest=out, nodes=[record])


def test_stage_puts_a_passing_and_a_failing_run_of_the_same_task_side_by_side(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    record.runs["e__f-3"] = core.TaskRun(instance_id="e__f-3", status="ok", resolved=True)
    record.runs["g__h-4"] = core.TaskRun(instance_id="g__h-4", status="ok", resolved=False)
    record.total, record.resolved = 4, 2
    runs.write_node(node=record)

    def repeat(name: str, created: float, outcomes: dict[str, tuple[str, bool]], variant: str = "") -> core.Node:
        node = record.model_copy(update={"name": name, "parent": "t-node", "created": created, "variant": variant})
        node.runs = {
            iid: core.TaskRun(instance_id=iid, status=status, resolved=ok) for iid, (status, ok) in outcomes.items()
        }
        return node

    older = repeat(
        name="older",
        created=1.0,
        outcomes={"a__b-1": ("ok", False), "c__d-2": ("ok", True), "e__f-3": ("error", False), "g__h-4": ("ok", False)},
    )
    newer = repeat(
        name="newer",
        created=2.0,
        outcomes={"a__b-1": ("ok", False), "c__d-2": ("ok", True), "e__f-3": ("ok", True), "g__h-4": ("ok", False)},
    )
    other = repeat(name="other", created=3.0, outcomes={"g__h-4": ("ok", True)}, variant=str(tmp_path / "v"))
    nodes = [record, older, newer, other]
    out = generation.stage(record=record, h=_FakeHarness(), dest=tmp_path / "stage", nodes=nodes)
    assert sorted(p.name for p in (out / "tasks" / "diverged").iterdir()) == ["a__b-1.md", "c__d-2.md"]
    first = (out / "tasks" / "diverged" / "a__b-1.md").read_text()
    assert first.startswith("# a__b-1: passed in 1 of 3 runs of the parent's harness\n\n## passed (run t-node)\n")
    assert "## failed (run newer)" in first and "ended " in first
    second = (out / "tasks" / "diverged" / "c__d-2.md").read_text()
    assert "## passed (run newer)" in second and "## failed (run t-node)" in second
    overview = (out / "overview.md").read_text()
    assert overview.index("## Where the parent's runs disagree") < overview.index("## How the agent works")
    assert "| a__b-1 | 1 of 3 |\n| c__d-2 | 2 of 3 |\n" in overview


def test_the_teacher_never_sees_a_task_the_lineage_hides(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    record = _parent_node()
    record.runs["e__f-3"] = core.TaskRun(instance_id="e__f-3", status="ok", resolved=True)
    record.total, record.resolved = 3, 2
    runs.write_node(node=record)
    runs.hidden_path(root=record).write_text("# hidden from the teacher\na__b-1\n")
    again = record.model_copy(update={"name": "again", "parent": "t-node", "created": 2.0})
    again.runs = {
        iid: core.TaskRun(instance_id=iid, status="ok", resolved=not run.resolved) for iid, run in record.runs.items()
    }
    nodes = [record, again]
    assert generation.pass_counts(record=record, nodes=nodes, hidden={"a__b-1"}) == {"c__d-2": (1, 2), "e__f-3": (1, 2)}
    out = generation.stage(record=record, h=_FakeHarness(), dest=tmp_path / "stage", nodes=nodes)
    assert sorted(p.relative_to(out / "tasks").as_posix() for p in (out / "tasks").rglob("*.md")) == [
        "c__d-2.md",
        "diverged/c__d-2.md",
        "diverged/e__f-3.md",
        "e__f-3.md",
        "full/c__d-2.md",
        "full/e__f-3.md",
    ]
    assert not [p for p in out.rglob("*") if p.is_file() and "a__b-1" in p.read_text(errors="replace")]
    assert "- result: 1 of 2 tasks passed the hidden tests" in (out / "overview.md").read_text()
    assert "parent's run on c__d-2" in (out / "system-prompt.md").read_text()


def test_overview_tells_a_run_that_never_started_from_one_that_failed() -> None:
    record = _parent_node()
    record.runs["e__f-3"] = core.TaskRun(instance_id="e__f-3", status="error", seconds=60.0)
    record.runs["g__h-4"] = core.TaskRun(instance_id="g__h-4", status="error", seconds=600.0)
    metrics = {
        "a__b-1": core.TaskMetrics(requests=2, stop_reason="stop"),
        "c__d-2": core.TaskMetrics(requests=1),
        "e__f-3": core.TaskMetrics(),
        "g__h-4": core.TaskMetrics(requests=5),
    }
    overview = generation.overview_markdown(record=record, metrics=metrics, passes={}, diverged=[])
    assert "| never started | 1 | 0 |" in overview and "| error | 1 | 0 |" in overview
    assert "| e__f-3 | no | 0 of 1 | never started |" in overview and "| g__h-4 | no | 0 of 1 | error |" in overview


def test_prompt_text_fills_every_placeholder() -> None:
    text = generation.prompt_text(harness_name="pi", model_name="qwen-3.8-27b", area="tools", timeout_s=3600, start=0.0)
    assert "{" not in text
    assert "qwen-3.8-27b" in text and not re.search(r"\bturns\b", text)
    assert "You have 60 minutes: the session stops without warning at 01:00 UTC" in text
    assert "This session's area is the model's tools: which ones it has, and what each one is called" in text
    assert "`tasks/diverged/<id>.md`" in text


def test_areas_come_from_areas_md() -> None:
    texts = generation.area_texts()
    assert list(texts) == ["open", "tools", "middleware", "context", "compose"]
    assert texts["open"] == "the whole harness: make the change you'd bet on most, wherever it is"
    assert texts["tools"].endswith("Keep your change inside it")
    assert generation.AREAS == ("tools", "middleware", "context")


@pytest.mark.parametrize(
    ("given", "candidates", "areas"),
    [
        ([], [], ["open", "tools", "middleware", "context"]),
        ([], ["a"], ["open", "tools", "middleware", "context"]),
        ([], ["a", "b"], ["open", "tools", "middleware", "compose"]),
        (["tools", "tools", "context", "open"], ["a", "b"], ["open", "middleware", "context", "compose"]),
    ],
)
def test_a_round_is_open_first_then_the_areas_given_fewest_sessions_then_compose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: list[str], candidates: list[str], areas: list[str]
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    monkeypatch.setattr(generation.variants, "compose_candidates", lambda record, entries, nodes: candidates)
    record = _parent_node()
    for n, area in enumerate(given, start=1):
        (paths.node_dir(name=record.name) / "generations" / f"2026102{n}-120000" / f"session-1-{area}").mkdir(
            parents=True
        )
    assert generation.default_areas(record=record, nodes=[record]) == areas


def _areas_of(prompts: list[str]) -> list[str]:
    texts = generation.area_texts()
    return [next(area for area, text in texts.items() if text in prompt) for prompt in prompts]


class _FakeHarness:
    name = "pi"
    version = "1.3.0"
    package_dir = paths.harnesses_dir() / "pi" / "package"
    doc = paths.harnesses_dir() / "pi" / "HARNESS.md"
    protocol_terms = ("exactly done",)

    def __init__(
        self,
        write: bool = True,
        decline: str | None = None,
        plan: tuple[str, ...] = (),
        idle: tuple[str, int] = ("error", 0),
    ) -> None:
        self.write = write
        self.decline = decline
        self.plan = plan
        self.idle = idle
        self.prompts: list[str] = []
        self.trees: list[str] = []

    def task_metrics(
        self, *, instance_dir: Path, model: core.ModelSpec, seconds: float, patch: str
    ) -> core.TaskMetrics:
        return core.TaskMetrics(requests=3)

    def transcript(self, *, instance_dir: Path, full: bool) -> str:
        return "## turn 1\n" if full else "[1] 10 tok out\n"

    def system_context(self, *, instance_dir: Path) -> str:
        return f"told before {instance_dir.name}\n"

    @contextlib.contextmanager
    def teacher_container(self, *, workspace, out_dir, model, endpoint, name, terms) -> Iterator[str]:
        assert "exactly done" in terms and "swebench" in terms
        self.workspace = workspace
        self.teacher_out = out_dir
        yield f"rsi-teacher-{name}"

    def validate_variant(self, *, variant: Path, out_dir: Path, terms: tuple[str, ...]) -> list[str]:
        check = folders.module(root=paths.harnesses_dir(), name="pi", file="check_variant.py")
        return check.static_problems(package=variant, terms=terms)


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
        h.trees.append((workspace / "tree.md").read_text())
        assert (workspace / "overview.md").is_file()
        if h.plan:
            action, number = h.plan[len(h.prompts) - 1], len(h.prompts)
            if action == "down":
                status, tools = h.idle
                error = "429 rate limited" if status == "error" else None
                return teacher.TeacherRun(status=status, seconds=1.0, error=error, tool_calls=tools)
            produced.mkdir(parents=True, exist_ok=True)
            if action == "none":
                (produced / "NONE.md").write_text("Nothing in this area is worth a run.\n")
            else:
                package = produced / f"{action}-{number}"
                shutil.copytree(workspace / "parent", package)
                if action != "bad":
                    (package / "APPEND_SYSTEM.md").write_text(f"Change number {number}.\n")
                (package / cards.FILE).write_text(f"# Change {number}\nhypothesis: h\n")
            status = "error" if action == "late" else "ok"
            return teacher.TeacherRun(status=status, seconds=1.0, model_calls=3, tool_calls=12)
        if h.decline is not None:
            produced.mkdir(parents=True, exist_ok=True)
            (produced / "NONE.md").write_text(h.decline)
        if h.write and len(h.prompts) == 1:
            good = produced / "tight-loop"
            shutil.copytree(workspace / "parent", good)
            (good / "APPEND_SYSTEM.md").write_text("Work in a tight loop.\n")
            (good / cards.FILE).write_text(
                "# End every turn with progress\nhypothesis: h1\nevidence: e1\npredicts: p1\nports: ghost\n"
            )
            same = produced / "no-op"
            shutil.copytree(workspace / "parent", same)
            (same / cards.FILE).write_text("# Change nothing at all\nhypothesis: h2\nevidence: e2\npredicts: p2\n")
            leaky = produced / "leaky"
            shutil.copytree(workspace / "parent", leaky)
            (leaky / "APPEND_SYSTEM.md").write_text("When done, reply with exactly DONE.\n")
            (leaky / cards.FILE).write_text("# Name the protocol\nhypothesis: h3\nevidence: e3\npredicts: p3\n")
            essay = produced / "Keep A Copy"
            shutil.copytree(workspace / "parent", essay)
            (essay / "SYSTEM.md").write_text("Keep a pristine copy before editing.\n")
            (essay / cards.FILE).write_text("# essay\n\n**Why:** runs repeat work.\n\n## Evidence\n\nMany runs.\n")
            linked = produced / "linked"
            shutil.copytree(workspace / "parent", linked)
            (linked / "PROBLEMS.md").symlink_to(paths.node_dir(name="t-node") / "node.json")
            (produced / "outside").symlink_to(workspace / "parent", target_is_directory=True)
        return teacher.TeacherRun(status="ok", seconds=1.0, model_calls=3, tool_calls=12, cost="1 AI credit")


def _generate(fake: _FakeHarness, areas: list[str], monkeypatch: pytest.MonkeyPatch) -> generation.Round:
    record = _parent_node()
    runs.write_node(node=record)
    monkeypatch.setattr(generation.teacher, "load", lambda name: _FakeTeacher(h=fake))
    return generation.generate(
        record=record,
        h=fake,
        teacher_spec=core.TeacherSpec(name="opus", agent="fake", model="m", effort="max"),
        endpoint=core.ModelEndpoint(model_id="m", base_url="http://fake"),
        areas=areas,
        timeout_s=1,
    )


def test_generate_registers_good_candidates_and_rejects_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    fake = _FakeHarness()
    found = _generate(fake=fake, areas=["tools", "middleware"], monkeypatch=monkeypatch)
    out, registered = found.out, found.registered
    assert out.parent == paths.node_dir(name="t-node") / "generations" and found.declined == {}
    first = out / "session-1-tools"
    assert (first / "transcript.md").read_text() == "## the teacher's session\n"
    assert [e.name for e in registered] == ["keep-a-copy", "tight-loop"]
    entry = variants.tree_load()["tight-loop"]
    assert (entry.title, entry.parent, entry.parent_node) == ("End every turn with progress", "baseline", "t-node")
    assert entry.created_by == "teacher:opus"
    assert (entry.area, entry.round) == ("tools", out.name)
    assert entry.diff == "+ APPEND_SYSTEM.md"
    assert (paths.variants_dir() / "tight-loop" / cards.FILE).is_file()
    essay = variants.tree_load()["keep-a-copy"]
    assert (essay.title, essay.hypothesis, essay.evidence) == ("essay", "**Why:** runs repeat work.", "Many runs.")
    assert (paths.variants_dir() / "keep-a-copy" / "SYSTEM.md").is_file()
    assert (first / generation.TEACHER_OUT / "variants" / "no-op").is_dir()
    problems = (out / "checks" / "session-1-tools" / "no-op" / "PROBLEMS.md").read_text()
    assert "identical to the parent's" in problems
    assert "exactly done" in (out / "checks" / "session-1-tools" / "leaky" / "PROBLEMS.md").read_text()
    linked = (out / "checks" / "session-1-tools" / "linked" / "PROBLEMS.md").read_text()
    assert linked.startswith("- PROBLEMS.md is a symlink")
    assert runs.read_node(name="t-node").name == "t-node"
    assert not (paths.variants_dir() / "outside").exists()
    assert entry.ports == []
    assert "None yet: this is the first round." in fake.trees[0]
    assert "Made from this parent so far: `keep-a-copy`, `tight-loop`." in fake.trees[1]
    assert "| tight-loop | baseline | tools | End every turn with progress | not run |" in fake.trees[1]
    assert _areas_of(prompts=fake.prompts) == ["tools", "middleware"]
    assert sorted(p.name for p in out.glob("session-*")) == ["session-1-tools", "session-2-middleware"]


def test_generate_without_candidates_registers_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    fake = _FakeHarness(write=False)
    found = _generate(fake=fake, areas=["tools"], monkeypatch=monkeypatch)
    assert found.registered == [] and found.declined == {} and variants.tree_load() == {}
    assert _areas_of(prompts=fake.prompts) == ["tools"]


@pytest.mark.parametrize(
    ("text", "declined"),
    [
        ("Every pattern I found\nis already tried.\n", "Every pattern I found is already tried."),
        ("", "it gave no reason"),
    ],
)
def test_a_session_that_finds_nothing_worth_running_declines_its_area(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, declined: str
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    fake = _FakeHarness(write=False, decline=text)
    areas = ["tools", "middleware", "context", "open"]
    found = _generate(fake=fake, areas=areas, monkeypatch=monkeypatch)
    assert found.registered == [] and found.declined == dict.fromkeys(areas, declined)
    assert _areas_of(prompts=fake.prompts) == areas


def test_generate_runs_one_session_per_area_in_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    waits: list[float] = []
    monkeypatch.setattr(generation.time, "sleep", waits.append)
    fake = _FakeHarness(plan=("none", "bad", "late", "good"))
    areas = ["tools", "middleware", "context", "compose"]
    found = _generate(fake=fake, areas=areas, monkeypatch=monkeypatch)
    assert _areas_of(prompts=fake.prompts) == areas
    assert [(e.name, e.area, e.round) for e in found.registered] == [
        ("late-3", "context", found.out.name),
        ("good-4", "compose", found.out.name),
    ]
    assert found.declined == {"tools": "Nothing in this area is worth a run."}
    checks = found.out / "checks" / "session-2-middleware" / "bad-2" / "PROBLEMS.md"
    assert "identical to the parent's" in checks.read_text()
    sessions = sorted(p.name for p in found.out.glob("session-*"))
    assert sessions == [f"session-{n}-{area}" for n, area in enumerate(areas, start=1)]
    assert waits == []


@pytest.mark.parametrize("idle", [("error", 0), ("ok", 0), ("error", 7)])
def test_a_session_that_fails_before_doing_any_work_stops_the_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, idle: tuple[str, int]
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    waits: list[float] = []
    monkeypatch.setattr(generation.time, "sleep", waits.append)
    fake = _FakeHarness(plan=("good", "down", "good"), idle=idle)
    with pytest.raises(generation.TeacherUnavailable) as stopped:
        _generate(fake=fake, areas=["tools", "middleware", "context"], monkeypatch=monkeypatch)
    (out,) = (paths.node_dir(name="t-node") / "generations").iterdir()
    reason = "429 rate limited" if idle[0] == "error" else "no tool calls"
    assert str(stopped.value) == (
        f"session 2 failed before doing any work ({reason}), so the round stops; the areas left are middleware, "
        f"context, and the round is in {out}"
    )
    assert len(fake.prompts) == 2 and list(variants.tree_load()) == ["good-1"] and waits == []


def test_a_variant_is_in_variants_before_the_tree_names_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    there: list[bool] = []
    register = variants.tree_register

    def checked(
        name: str,
        parent: str,
        parent_node: str,
        card: cards.VariantCard,
        diff: str,
        created_by: str,
        area: str,
        round: str,
    ) -> variants.VariantEntry:
        there.append((paths.variants_dir() / name / cards.FILE).is_file())
        return register(
            name=name,
            parent=parent,
            parent_node=parent_node,
            card=card,
            diff=diff,
            created_by=created_by,
            area=area,
            round=round,
        )

    monkeypatch.setattr(generation.variants, "tree_register", checked)
    found = _generate(fake=_FakeHarness(), areas=["tools"], monkeypatch=monkeypatch)
    assert len(found.registered) == 2 and there == [True, True]
    assert found.registered[1].diff == "+ APPEND_SYSTEM.md"


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


def test_tree_stages_the_runs_an_evaluated_variant_flipped(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    parent = _parent_node()
    runs.write_node(node=parent)
    (tmp_path / "variants" / "tight-loop").mkdir(parents=True)
    card = cards.VariantCard(title="End every turn with progress", hypothesis="h", evidence="23 of 37 failed runs")
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
    h = harness.load(name="pi")
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    generation.refresh_tree(dest=stage_dir, h=h, record=parent, entries=variants.tree_load(), nodes=runs.list_nodes())
    staged = sorted(p.name for p in (stage_dir / "results" / "tight-loop").iterdir())
    assert staged == ["broken-c__d-2.md", "fixed-a__b-1.md"]
    fixed = (stage_dir / "results" / "tight-loop" / "fixed-a__b-1.md").read_text()
    assert fixed.startswith("# a__b-1: fixed by tight-loop (run child, parent run repeat)")
    assert "## tight-loop" in fixed and "## parent" in fixed
    tree_md = (stage_dir / "tree.md").read_text()
    assert "- evidence: 23 of 37 failed runs" in tree_md
    assert "- result: 2.0/3 per run; +0.0 ± 1.4 tasks per run (z 0.00; 1 run vs 1 of the parent, 3 tasks)" in tree_md
    assert "| tight-loop | baseline |  | End every turn with progress | 2.0/3 per run; +0.0 ± 1.4" in tree_md
    assert "- runs: `results/tight-loop/` (1 fixed and 1 broken tasks, each run next to the parent's)" in tree_md
    runs.hidden_path(root=parent).write_text("c__d-2\n")
    generation.refresh_tree(dest=stage_dir, h=h, record=parent, entries=variants.tree_load(), nodes=runs.list_nodes())
    assert [p.name for p in (stage_dir / "results" / "tight-loop").iterdir()] == ["fixed-a__b-1.md"]
    tree_md = (stage_dir / "tree.md").read_text()
    assert "- runs: `results/tight-loop/` (1 fixed and 0 broken tasks" in tree_md
    assert "- result: 2.0/3 per run; +0.0 ± 1.4 tasks per run (z 0.00; 1 run vs 1 of the parent, 3 tasks)" in tree_md
