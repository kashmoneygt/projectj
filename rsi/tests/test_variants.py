from pathlib import Path

import pytest

from rsi import cards, core, runs, variants


def test_safe_variant_name() -> None:
    assert variants.safe_variant_name(name="test-before-edit") == "test-before-edit"
    assert variants.safe_variant_name(name="run_tests_hook") == "run_tests_hook"
    assert variants.safe_variant_name(name="fix2") == "fix2"
    for bad in ("", "Test-edit", "1fix", "-lead", "foo.bar", "has space", "a--b", "a" * 41):
        with pytest.raises(ValueError):
            variants.safe_variant_name(name=bad)


def test_free_name_makes_any_proposed_name_safe_and_unique() -> None:
    assert variants.free_name(proposed="test-before-edit", taken=()) == "test-before-edit"
    assert variants.free_name(proposed="Keep A Copy", taken=()) == "keep-a-copy"
    assert variants.free_name(proposed="run_tests--hook!", taken=()) == "run-tests-hook"
    assert variants.free_name(proposed="2-step plan", taken=()) == "v-2-step-plan"
    assert variants.free_name(proposed="...", taken=()) == "variant"
    assert variants.free_name(proposed="fix", taken={"fix", "fix-2"}) == "fix-3"
    assert variants.free_name(proposed="Baseline", taken=()) == "baseline-2"
    # the teacher's /work/runs/parent/ holds the parent's own runs
    assert variants.free_name(proposed="parent", taken=()) == "parent-2"
    long = variants.free_name(proposed="a" * 50, taken={"a" * 40})
    assert long == "a" * 38 + "-2"
    for name in ("keep-a-copy", "run-tests-hook", "v-2-step-plan", "fix-3", long):
        assert variants.safe_variant_name(name=name) == name


def test_variant_name_and_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    assert variants.variant_name(reference="") == "baseline"
    assert variants.variant_name(reference="test-before-edit") == "test-before-edit"
    registry = tmp_path / "variants"
    variant = registry / "test-before-edit"
    variant.mkdir(parents=True)
    assert variants.resolve_variant(reference="test-before-edit") == variant
    with pytest.raises(FileNotFoundError, match="unknown variant"):
        variants.resolve_variant(reference="run-tests-hook")
    direct = tmp_path / "elsewhere"
    direct.mkdir()
    assert variants.variant_name(reference=str(direct)) == "elsewhere"
    assert variants.resolve_variant(reference=str(direct)) == direct
    with pytest.raises(FileNotFoundError):
        variants.resolve_variant(reference=str(tmp_path / "missing"))


def _card(title: str = "Verify before finishing", reuses: tuple[str, ...] = ()) -> cards.VariantCard:
    return cards.VariantCard(
        title=title,
        what="Runs settle unverified; a check before settling catches it.",
        expect="fewer patched runs fail",
        reuses=reuses,
    )


def test_package_comparison_ignores_the_card(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "a.md").write_text("one\ntwo\n")
    (base / "keep.md").write_text("same\n")
    (base / "gone.md").write_text("bye\n")
    variant = tmp_path / "variant"
    variant.mkdir()
    (variant / "a.md").write_text("one\nTWO\nthree\n")
    (variant / "keep.md").write_text("same\n")
    (variant / "new.ts").write_text("x\n")
    (variant / cards.FILE).write_text("t\nWhat: w\n")
    s = variants.diff_summary(base=base, variant=variant)
    assert "+ new.ts" in s
    assert "~ a.md (+2 -1)" in s
    assert "- gone.md" in s
    assert "keep.md" not in s and cards.FILE not in s
    full = variants.unified_diff(base=base, variant=variant)
    assert "+TWO" in full and "b/new.ts" in full and "a/gone.md" in full
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in ("a.md", "keep.md", "gone.md"):
        (copy / name).write_text((base / name).read_text())
    (copy / cards.FILE).write_text("a card does not make a change\nWhat: w\n")
    assert variants.same_package(a=base, b=copy)
    assert variants.diff_summary(base=base, variant=copy) == "identical"
    assert not variants.same_package(a=base, b=variant)
    (copy / ".DS_Store").write_text("finder\n")
    assert variants.package_digest(package=copy) == variants.package_digest(package=base)
    assert variants.package_digest(package=variant) != variants.package_digest(package=base)
    (copy / "keep.md").write_text("same, edited\n")
    assert variants.package_digest(package=copy) != variants.package_digest(package=base)


def test_tree_register(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    assert variants.tree_load() == {}
    entry = variants.tree_register(name="verify", parent="baseline", parent_node="n1", card=_card(), diff="+ a.md")
    assert entry.title == "Verify before finishing" and entry.reuses == []
    loaded = variants.tree_load()["verify"]
    assert loaded == entry
    assert loaded.what.startswith("Runs settle unverified") and loaded.expect == "fewer patched runs fail"
    with pytest.raises(ValueError, match="already in the tree"):
        variants.tree_register(name="verify", parent="baseline", parent_node="n1", card=_card(), diff="d")
    with pytest.raises(ValueError, match="aren't in the tree"):
        variants.tree_register(
            name="reuser", parent="baseline", parent_node="n1", card=_card(reuses=("ghost",)), diff="d"
        )
    reuser = variants.tree_register(
        name="verify-again", parent="other", parent_node="n2", card=_card(reuses=("verify",)), diff="d"
    )
    assert reuser.reuses == ["verify"]


def _node(name: str, variant: str, parent: str | None, resolved: list[bool], created: float) -> core.Node:
    node = core.Node(
        name=name,
        parent=parent,
        dataset=core.DatasetSpec(name="d", dataset_id="i"),
        harness="pi",
        harness_version="1",
        variant=variant,
        model=core.ModelSpec(name="m", source="s", model_id="i"),
        budget=core.Budget(timeout_s=1),
        tasks=len(resolved),
        seed=1,
        created=created,
    )
    for i, ok in enumerate(resolved):
        node.runs[f"t__t-{i}"] = core.TaskRun(instance_id=f"t__t-{i}", status="ok", resolved=ok)
    node.task_ids = sorted(node.runs)
    node.total, node.resolved = len(resolved), sum(resolved)
    return node


def test_a_variants_runs_come_from_its_parent_and_its_outcome_is_in_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    variants.tree_register(name="verify", parent="baseline", parent_node="root", card=_card(), diff="+ a.md")
    (tmp_path / "variants" / "verify").mkdir()
    root = _node(name="root", variant="", parent=None, resolved=[True, False, False, True], created=1.0)
    child = _node(
        name="child",
        variant=str(tmp_path / "variants" / "verify"),
        parent="root",
        resolved=[True, True, True, False],
        created=2.0,
    )
    nodes = [root, child]
    entry = variants.tree_load()["verify"]
    assert [n.name for n in variants.variant_nodes(entry=entry, nodes=nodes)] == ["child"]
    other = _node(
        name="other",
        variant=str(tmp_path / "variants" / "verify"),
        parent="child",
        resolved=[True, True, True, True],
        created=3.0,
    )
    assert [n.name for n in variants.variant_nodes(entry=entry, nodes=[*nodes, other])] == ["child"]
    # nothing moved, so there's no behavior to report
    assert variants.result_line(node=child, nodes=nodes) == "3/4 vs parent 2/4"
    # the teacher learns how a variant did in words, never by how many tasks it solved
    assert variants.outcome(entry=entry, nodes=nodes) == "no clear difference"


def test_outcome_says_why_a_variant_has_no_result_yet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    variants.tree_register(name="verify", parent="baseline", parent_node="root", card=_card(), diff="+ a.md")
    entry = variants.tree_load()["verify"]
    assert variants.outcome(entry=entry, nodes=[]) == "not run"
    started = _node(
        name="started", variant=str(tmp_path / "variants" / "verify"), parent="root", resolved=[], created=2.0
    )
    started.runs["t__t-0"] = core.TaskRun(instance_id="t__t-0", status="ok")
    assert variants.outcome(entry=entry, nodes=[started]) == "not scored yet"
    started.runs["t__t-1"] = core.TaskRun(
        instance_id="t__t-1", status="error", error="pi died:\n  bad | extension in rsi-t__t-1 (T__t-1)"
    )
    # task ids are left out, since the task may be one the teacher never sees
    assert variants.outcome(entry=entry, nodes=[started]) == (
        "not scored yet; first error: pi died: bad / extension in rsi-<task> (<task>)"
    )


def test_activations_say_how_often_a_variant_kicked_in_and_its_parents_harness_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    package = tmp_path / "variants" / "gate"
    (package / "extensions").mkdir(parents=True)
    (package / "extensions" / "gate.ts").write_text('pi.appendEntry("acted", { what: "blocked" });\n')
    variants.tree_register(name="gate", parent="baseline", parent_node="root", card=_card(), diff="+ gate.ts")
    root = _node(name="root", variant="", parent=None, resolved=[True, False], created=1.0)
    first = _node(name="first", variant=str(package), parent="root", resolved=[True, True], created=2.0)
    second = _node(name="second", variant=str(package), parent="root", resolved=[True, False], created=3.0)
    first.runs["t__t-0"].metrics.acted = 3
    second.runs["t__t-1"].metrics.acted = 1
    nodes = [root, first, second]
    entry = variants.tree_load()["gate"]
    assert variants.activations(entry=entry, nodes=nodes) == "kicked in on 2 of 4 task runs"
    # a variant carries its parent's changes, so the parent's harness is counted too when it kicks in
    root.runs["t__t-0"].metrics.acted = 1
    assert variants.activations(entry=entry, nodes=nodes) == (
        "kicked in on 2 of 4 task runs, against 1 of 2 for its parent's harness"
    )


def test_scoreboard_puts_a_variant_against_every_run_of_its_parents_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    package = tmp_path / "variants" / "gate"
    (package / "extensions").mkdir(parents=True)
    (package / "extensions" / "gate.ts").write_text('pi.appendEntry("acted", { what: "blocked" });\n')
    variants.tree_register(name="gate", parent="baseline", parent_node="root", card=_card(), diff="+ gate.ts")
    entry = variants.tree_load()["gate"]
    assert variants.scoreboard(entry=entry, nodes=[]) == ["from root", "not run"]
    root = _node(name="root", variant="", parent=None, resolved=[False, False, False, True], created=1.0)
    repeat = _node(name="repeat", variant="", parent="root", resolved=[False, False, False, True], created=2.0)
    first = _node(name="first", variant=str(package), parent="root", resolved=[True] * 4, created=3.0)
    second = _node(name="second", variant=str(package), parent="root", resolved=[True] * 4, created=4.0)
    for i, spent in enumerate((2000, 3000, 1500, 2500)):
        for node, tokens in ((root, 1000), (repeat, 1000), (first, spent), (second, spent)):
            node.runs[f"t__t-{i}"].metrics.output_tokens = tokens
    first.runs["t__t-0"].metrics.acted = 3
    second.runs["t__t-1"].metrics.acted = 1
    for node, task in ((first, "t__t-0"), (second, "t__t-1"), (root, "t__t-0")):
        node.runs[task].metrics.contract_violations = ["sampling {'top_k': 40}"]
    assert variants.scoreboard(entry=entry, nodes=[first, second]) == [
        "from root",
        "2 runs, 4.0/4 solved per run",
        "no run of the parent's harness on these tasks to compare with yet",
    ]
    assert variants.outcome(entry=entry, nodes=[first, second]) == "no run of its parent's harness to compare with yet"
    runs.write_node(node=root)
    runs.hidden_path(root=root).write_text("# hidden from the teacher\nt__t-2\nt__t-3\n")
    nodes = [root, repeat, first, second]
    assert variants.scoreboard(entry=entry, nodes=nodes) == [
        "from root",
        "2 runs, 4.0/4 solved per run",
        "solved +3.0 ± 1.0 tasks per run, better, beyond noise (2 runs vs 2 of the parent, 4 tasks)",
        (
            "on the 2 tasks hidden from the teacher +1.0 ± 0.6 tasks per run, within noise "
            "(2 runs vs 2 of the parent, 2 tasks)"
        ),
        "tokens per task +118% ± 15%, worse, beyond noise (2 runs vs 2 of the parent, 4 tasks)",
        "broke the contract on 2 task runs, so its result doesn't count",
        "kicked in on 2 of 8 task runs",
        "not shown better than its parent",
    ]
    assert variants.outcome(entry=entry, nodes=nodes) == "it changed what's fixed, so its runs don't count"
    for node in (first, second):
        for run in node.runs.values():
            run.metrics.contract_violations = []
    found = variants.standing(entry=entry, nodes=nodes)
    assert found is not None and found.better_by == "solves"
    assert variants.scoreboard(entry=entry, nodes=nodes)[-1] == "better than its parent: solves more"
    assert variants.outcome(entry=entry, nodes=nodes) == "better than its parent: solves more"
    second.server = {"engine": "fake 1.1"}
    runs.hidden_path(root=root).unlink()
    board = variants.scoreboard(entry=entry, nodes=nodes)
    assert board[2:4] == [
        (
            "these runs don't compare with the parent's, so none of this counts: not the same model server: {} on "
            'root, repeat, first; {"engine": "fake 1.1"} on second'
        ),
        "solved +3.0 ± 1.0 tasks per run, better, beyond noise (2 runs vs 2 of the parent, 4 tasks)",
    ]
    found = variants.standing(entry=entry, nodes=nodes)
    assert found is not None and found.better_by == ""
    assert variants.outcome(entry=entry, nodes=nodes) == (
        "its runs were made differently from its parent's, so they don't compare"
    )
    assert not [line for line in board if "hidden from the teacher" in line]


def test_outcome_says_when_a_variant_did_worse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    package = tmp_path / "variants" / "noisy"
    package.mkdir(parents=True)
    variants.tree_register(name="noisy", parent="baseline", parent_node="root", card=_card(), diff="+ a.md")
    root = _node(name="root", variant="", parent=None, resolved=[True, True, True, False], created=1.0)
    repeat = _node(name="repeat", variant="", parent="root", resolved=[True, True, True, False], created=2.0)
    first = _node(name="first", variant=str(package), parent="root", resolved=[False] * 4, created=3.0)
    second = _node(name="second", variant=str(package), parent="root", resolved=[False] * 4, created=4.0)
    entry = variants.tree_load()["noisy"]
    assert variants.outcome(entry=entry, nodes=[root, repeat, first, second]) == "worse than its parent: solves fewer"
