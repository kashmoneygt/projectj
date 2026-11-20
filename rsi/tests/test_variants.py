import dataclasses
from datetime import UTC, datetime
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
    # "parent" is reserved, since the teacher's /work/runs/parent/ holds the parent's runs
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
    # no behavior changed, so the line only has the scores
    assert variants.result_line(node=child, nodes=nodes) == "3/4 vs parent 2/4"
    # the teacher only gets the outcome in words, never as task counts
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
    # task ids are left out, since the teacher may never see that task
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
    # a variant includes its parent's changes, so also count how often the parent kicked in
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


def _round_entry(name: str, round_id: str, created: str) -> variants.VariantEntry:
    return variants.VariantEntry(
        name=name,
        parent="baseline",
        parent_node="root",
        created=created,
        created_by="teacher:t",
        title=name,
        what="the change",
        why="",
        expect="",
        reuses=[],
        diff="+ a.md",
        round=round_id,
    )


def _probe_lineage() -> core.Node:
    """A root run on 6 tasks, with the first 4 as its probe set"""
    root = _node(name="root", variant="", parent=None, resolved=[True] * 6, created=0.0)
    runs.write_node(node=root)
    runs.probe_path(root=root).write_text("# probe\n" + "\n".join(f"t__t-{i}" for i in range(4)) + "\n")
    return root


def _probe_run(variant: str, seed: int, solved: int, created: float) -> core.Node:
    resolved = [True] * solved + [False] * (4 - solved)
    node = _node(
        name=f"{variant or 'baseline'}-s{seed}", variant=variant, parent="root", resolved=resolved, created=created
    )
    node.seed = seed
    return node


def test_a_round_runs_its_parent_and_each_variant_once_then_reruns_its_best_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    nodes = [_probe_lineage(), _probe_run(variant="", seed=42, solved=2, created=0.5)]
    made = [_round_entry(name="older", round_id="r1", created="2026-01-01T00:00:00+00:00")] + [
        _round_entry(name=name, round_id="r2", created=f"2026-01-02T00:00:0{i}+00:00")
        for i, name in enumerate(("alpha", "beta", "gamma"))
    ]
    entries = {e.name: e for e in made}

    def due() -> tuple[str, int] | None:
        found = variants.next_run(entries=entries, nodes=nodes)
        return found[:2] if found else None

    # the round's parent runs once more first, since its run from before the round doesn't count
    assert due() == ("baseline", 43)
    nodes.append(_probe_run(variant="", seed=43, solved=2, created=datetime(2026, 1, 3, tzinfo=UTC).timestamp()))
    # only the newest round counts, and each variant in it runs once
    assert due() == ("alpha", 42)
    nodes += [
        _probe_run(variant=name, seed=42, solved=solved, created=i + 1.0)
        for i, (name, solved) in enumerate((("alpha", 2), ("beta", 3), ("gamma", 1)))
    ]
    # beta's first run was the best, so it runs twice more
    assert due() == ("beta", 43)
    nodes.append(_probe_run(variant="beta", seed=43, solved=2, created=4.0))
    assert due() == ("beta", 44)
    nodes.append(_probe_run(variant="beta", seed=44, solved=2, created=5.0))
    # alpha's first run was second best, so it runs once more, and a 3rd time only if that puts it ahead of beta
    assert due() == ("alpha", 43)
    nodes.append(_probe_run(variant="alpha", seed=43, solved=2, created=6.0))
    assert due() is None
    nodes[-1] = _probe_run(variant="alpha", seed=43, solved=4, created=6.0)
    assert due() == ("alpha", 44)
    nodes.append(_probe_run(variant="alpha", seed=44, solved=1, created=7.0))
    assert due() is None


def test_the_next_parent_is_the_best_harness_with_two_runs_or_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    entries = {
        name: _round_entry(name=name, round_id="r1", created="2026-01-01T00:00:00+00:00")
        for name in ("alpha", "beta", "gamma", "delta")
    }
    solved = {"": (2, 2), "alpha": (3, 2), "beta": (3, 2), "gamma": (4,), "delta": (4, 4, 4)}
    nodes = [_probe_lineage()] + [
        _probe_run(variant=variant, seed=42 + k, solved=count, created=float(k))
        for variant, counts in solved.items()
        for k, count in enumerate(counts)
    ]
    # beta solves as many as alpha with fewer tokens, gamma ran once, and delta broke the contract
    for node in nodes[1:]:
        for run in node.runs.values():
            run.metrics.output_tokens = 500 if node.variant == "beta" else 1000
    next(n for n in nodes if n.variant == "delta").runs["t__t-0"].metrics.contract_violations = ["top_k 40"]
    assert variants.best(entries=entries, nodes=nodes).name == "beta"


def _two_kinds(variant: str, seed: int, solved: list[bool], created: float) -> core.Node:
    """A run on a probe set of 8 tasks, with the first 4 one kind of work and the last 4 another"""
    node = _node(
        name=f"{variant or 'baseline'}-s{seed}", variant=variant, parent="root", resolved=solved, created=created
    )
    node.seed = seed
    node.kinds = {f"t__t-{i}": "fixes" if i < 4 else "builds" for i in range(8)}
    return node


def test_a_harness_that_does_worse_at_one_kind_of_work_is_never_picked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    root = _node(name="root", variant="", parent=None, resolved=[True] * 10, created=0.0)
    runs.write_node(node=root)
    runs.probe_path(root=root).write_text("\n".join(f"t__t-{i}" for i in range(8)) + "\n")
    (tmp_path / "variants" / "lopsided").mkdir(parents=True)
    variants.tree_register(name="lopsided", parent="baseline", parent_node="root", card=_card(), diff="+ a.md")
    entries = {"lopsided": dataclasses.replace(variants.tree_load()["lopsided"], round="r1")}
    # lopsided solves every fix and no build, while the baseline solves every build and no fix
    nodes = [root] + [
        _two_kinds(variant=variant, seed=42 + k, solved=solved, created=1.0 + k)
        for variant, solved in (("", [False] * 4 + [True] * 4), ("lopsided", [True] * 4 + [False] * 4))
        for k in range(2)
    ]
    found = variants.standing(entry=entries["lopsided"], nodes=nodes)
    assert found.solves.gain == 0 and found.worse_at == ["builds"] and found.kinds["fixes"].z >= 2
    assert (
        variants.outcome(entry=entries["lopsided"], nodes=nodes) == "no clear difference overall, but worse at builds"
    )
    assert variants.contender(name="lopsided", root=root, nodes=nodes, parent="root").regressed == ("builds",)
    assert variants.best(entries=entries, nodes=nodes).name == "baseline"


def test_a_lineage_only_sees_its_own_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    old = _node(name="old-root", variant="", parent=None, resolved=[True], created=0.0)
    new = _node(name="new-root", variant="", parent=None, resolved=[True], created=1.0)
    later = _node(name="new-probe", variant="", parent="new-root", resolved=[True], created=2.0)
    entries = {
        "old-idea": _round_entry(name="old-idea", round_id="r1", created="2026-01-01T00:00:00+00:00"),
        "new-idea": dataclasses.replace(
            _round_entry(name="new-idea", round_id="r2", created="2026-01-02T00:00:00+00:00"), parent_node="new-probe"
        ),
    }
    entries["old-idea"] = dataclasses.replace(entries["old-idea"], parent_node="old-root")
    assert list(variants.lineage_entries(root=new, entries=entries, nodes=[old, new, later])) == ["new-idea"]
    assert list(variants.lineage_entries(root=old, entries=entries, nodes=[old, new, later])) == ["old-idea"]
