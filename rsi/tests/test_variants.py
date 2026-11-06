import dataclasses
from pathlib import Path

import pytest

from rsi import cards, core, paths, runs, variants


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
    long = variants.free_name(proposed="a" * 50, taken={"a" * 40})
    assert long == "a" * 38 + "-2"
    for name in ("keep-a-copy", "run-tests-hook", "v-2-step-plan", "fix-3", long):
        assert variants.safe_variant_name(name=name) == name


def test_variant_name_and_resolution(tmp_path, monkeypatch) -> None:
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


def test_package_comparison_ignores_the_card(tmp_path) -> None:
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


def test_tree_register(tmp_path, monkeypatch) -> None:
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


def test_the_log_shows_how_the_parent_was_built_one_commit_at_a_time(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "package.json").write_text("{}\n")
    for name, parent, line in (("a", "baseline", "Read before editing."), ("b", "a", "Run the tests.")):
        package = paths.variants_dir() / name
        source = baseline if parent == "baseline" else paths.variants_dir() / parent
        package.mkdir(parents=True)
        for file in source.iterdir():
            (package / file.name).write_text(file.read_text())
        appended = (package / "APPEND_SYSTEM.md").read_text() if (package / "APPEND_SYSTEM.md").is_file() else ""
        (package / "APPEND_SYSTEM.md").write_text(f"{appended}{line}\n")
        (package / cards.FILE).write_text(f"system prompt: say {name}\n\nWhy {name}.\n\nPredicts: {name} helps\n")
        variants.tree_register(name=name, parent=parent, parent_node="n", card=_card(title=name), diff="~")
    entries = variants.tree_load()
    assert variants.ancestry(name="b", entries=entries) == ["a", "b"]
    log = variants.lineage_log(parent="b", entries=entries, nodes=[], baseline=baseline)
    assert log.index("## a") < log.index("## b")
    assert "Built on `baseline`. It changed ~ and scored not run." in log and "Built on `a`." in log
    assert "    system prompt: say a\n\n    Why a." in log
    assert "+Read before editing." in log and "+Run the tests." in log
    # each diff is against the variant's own parent, so b's adds only its own line
    assert log.split("## b")[1].count("+Read before editing.") == 0
    first = variants.lineage_log(parent="baseline", entries=entries, nodes=[], baseline=baseline)
    assert "nothing has been added yet" in first
    assert "isn't in the tree" in variants.lineage_log(parent="ghost", entries=entries, nodes=[], baseline=baseline)


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


def test_tree_markdown_compares_with_the_parent_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    variants.tree_register(
        name="verify", parent="baseline", parent_node="root", card=_card(), diff="+ a.md", area="context", round="r1"
    )
    assert (variants.tree_load()["verify"].area, variants.tree_load()["verify"].round) == ("context", "r1")
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
    line = variants.result_line(node=child, nodes=nodes)
    assert line.startswith("3/4 vs parent 2/4 on the same tasks (+2 fixed, -1 broken; a split this lopsided")
    # nothing moved, so there's no behavior to report
    assert line.endswith("happens by chance with p=1.00)")
    assert variants.result_line(node=child, nodes=nodes, detail=False) == "3/4 vs parent 2/4"
    text = variants.tree_markdown(
        entries=variants.tree_load(), nodes=nodes, lineage=["baseline"], results={}, parent_node="root", turned_down=[]
    )
    assert "The parent's lineage: baseline. `log.md` has each of its commits." in text
    assert "Made from this parent so far: `verify`." in text
    assert "| variant | parent | area | title | result |" in text
    assert "## verify\n\nVerify before finishing\n" in text
    assert "- what: Runs settle unverified" in text
    assert (
        "| verify | baseline | context | Verify before finishing | 3.0/4 per run; +1.0 ± 1.7 tasks per run, within "
        in text
    )
    assert "- result: " not in text
    assert "- expect: fewer patched runs fail" in text
    assert "- kicked in" not in text
    empty = variants.tree_markdown(entries={}, nodes=[], lineage=[], results={}, parent_node="root", turned_down=[])
    assert "None yet, since this is the first round." in empty


def test_tree_markdown_shows_this_parents_variants_whole_and_shortens_the_rest(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    long = cards.VariantCard(title="snapshot the repo", what="word " * 300, why="e", expect="less " * 200)
    variants.tree_register(name="snap", parent="baseline", parent_node="root", card=long, diff="+ a.md")
    entries = variants.tree_load()
    mine = variants.tree_markdown(entries=entries, nodes=[], lineage=[], results={}, parent_node="root", turned_down=[])
    assert f"- what: {'word ' * 300}".strip() in mine and "- why: e" in mine
    other = variants.tree_markdown(entries=entries, nodes=[], lineage=[], results={}, parent_node="r", turned_down=[])
    shown = next(line for line in other.splitlines() if line.startswith("- expect: "))
    assert shown.endswith(" … (all of it is in `tree/snap/VARIANT.md`)")
    assert len(shown) < variants.FIELD_SHOWN + 80
    # a variant from another parent shows what it changed and what it expects, but not why
    assert "- what: word" in other and "- why: " not in other
    assert variants.tree_load()["snap"].what == "word " * 300


def test_tree_markdown_shows_a_note_from_a_person(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    variants.tree_register(name="snap", parent="baseline", parent_node="root", card=_card(), diff="+ a.md")
    plain = variants.tree_markdown(
        entries=variants.tree_load(), nodes=[], lineage=[], results={}, parent_node="r", turned_down=[]
    )
    assert "- note" not in plain
    entries = variants.tree_load()
    entries["snap"] = dataclasses.replace(entries["snap"], note="it never fired")
    variants.tree_save(entries=entries)
    text = variants.tree_markdown(
        entries=variants.tree_load(), nodes=[], lineage=[], results={}, parent_node="r", turned_down=[]
    )
    assert "- note from the people running the variants: it never fired" in text


def test_tree_markdown_says_why_a_variant_has_no_result_yet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    variants.tree_register(name="verify", parent="baseline", parent_node="root", card=_card(), diff="+ a.md")
    entry = variants.tree_load()["verify"]
    assert variants.pooled_result(entry=entry, nodes=[]) == ("not run", "")
    started = _node(
        name="started", variant=str(tmp_path / "variants" / "verify"), parent="root", resolved=[], created=2.0
    )
    started.runs["t__t-0"] = core.TaskRun(instance_id="t__t-0", status="ok")
    assert variants.pooled_result(entry=entry, nodes=[started]) == ("not scored yet", "")
    started.runs["t__t-1"] = core.TaskRun(
        instance_id="t__t-1", status="error", error="pi died:\n  bad | extension in rsi-t_1776_t-1 (T__t-1)"
    )
    text = variants.tree_markdown(
        entries=variants.tree_load(), nodes=[started], lineage=[], results={}, parent_node="root", turned_down=[]
    )
    assert (
        "| verify | baseline |  | Verify before finishing | not scored yet; first error: pi died: bad / extension in "
        "rsi-<task> (<task>) |"
    ) in text


def test_records_activations_looks_for_the_acted_entry_in_code(tmp_path) -> None:
    package = tmp_path / "package"
    (package / "extensions").mkdir(parents=True)
    (package / "APPEND_SYSTEM.md").write_text('Never call pi.appendEntry("acted").\n')
    (package / cards.FILE).write_text('gate\nWhat: it calls pi.appendEntry("acted", ...)\n')
    (package / "extensions" / "gate.ts").write_text('pi.appendEntry("noted", { what: "x" });\n')
    assert not variants.records_activations(package=package)
    (package / "extensions" / "gate.ts").write_text("pi.appendEntry(\n  'acted', { what: 'blocked a call' });\n")
    assert variants.records_activations(package=package)


def test_tree_markdown_says_how_often_a_variant_that_records_it_kicked_in(
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
    text = variants.tree_markdown(
        entries=variants.tree_load(),
        nodes=[root, first, second],
        lineage=[],
        results={},
        parent_node="root",
        turned_down=[],
    )
    assert "- kicked in on 2 of 4 task runs\n" in text


def test_compose_candidates_came_out_ahead_in_the_lineage_and_are_not_built_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    made_from = {
        "ahead": "root",
        "cheaper": "root",
        "behind": "root",
        "picked": "root",
        "unscored": "root",
        "unrun": "root",
        "elsewhere": "root",
        "next": "picked-run",
        "stranger": "other-root",
    }
    for name, parent_node in made_from.items():
        variants.tree_register(name=name, parent="baseline", parent_node=parent_node, card=_card(), diff="+ a.md")

    def run(name: str, resolved: list[bool]) -> core.Node:
        package = str(tmp_path / "variants" / name)
        return _node(name=f"{name}-run", variant=package, parent=made_from[name], resolved=resolved, created=3.0)

    root = _node(name="root", variant="", parent=None, resolved=[False, False, True, True], created=1.0)
    repeat = _node(name="repeat", variant="", parent="root", resolved=[False, False, True, True], created=2.0)
    cheaper = run(name="cheaper", resolved=[False, False, True, True])
    for i, (theirs, ours) in enumerate([(1000, 500), (2000, 900), (3000, 1600), (4000, 2100)]):
        root.runs[f"t__t-{i}"].metrics.output_tokens = repeat.runs[f"t__t-{i}"].metrics.output_tokens = theirs
        cheaper.runs[f"t__t-{i}"].metrics.output_tokens = ours
    picked = run(name="picked", resolved=[True, False, True, True])
    unscored = run(name="unscored", resolved=[True, True, True, True])
    unscored.total = 0
    nodes = [
        root,
        repeat,
        run(name="ahead", resolved=[True, False, True, True]),
        cheaper,
        run(name="behind", resolved=[False, False, False, True]),
        picked,
        unscored,
        run(name="elsewhere", resolved=[True, True, True]),
        run(name="next", resolved=[True, True, True, True]),
        run(name="stranger", resolved=[True, True, True, True]),
    ]
    found = variants.compose_candidates(record=picked, entries=variants.tree_load(), nodes=nodes)
    assert found == ["ahead", "cheaper", "next"]


def test_scoreboard_puts_a_variant_against_every_run_of_its_parents_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    package = tmp_path / "variants" / "gate"
    (package / "extensions").mkdir(parents=True)
    (package / "extensions" / "gate.ts").write_text('pi.appendEntry("acted", { what: "blocked" });\n')
    variants.tree_register(
        name="gate", parent="baseline", parent_node="root", card=_card(), diff="+ gate.ts", area="tools"
    )
    entry = variants.tree_load()["gate"]
    assert variants.scoreboard(entry=entry, nodes=[]) == ["from root, area tools", "not run"]
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
        "from root, area tools",
        "2 runs, 4.0/4 solved per run",
        "no run of the parent's harness on these tasks to compare with yet",
    ]
    runs.write_node(node=root)
    runs.hidden_path(root=root).write_text("# hidden from the teacher\nt__t-2\nt__t-3\n")
    nodes = [root, repeat, first, second]
    assert variants.scoreboard(entry=entry, nodes=nodes) == [
        "from root, area tools",
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
    for node in (first, second):
        for run in node.runs.values():
            run.metrics.contract_violations = []
    found = variants.standing(entry=entry, nodes=nodes)
    assert found is not None and found.better_by == "solves"
    assert variants.scoreboard(entry=entry, nodes=nodes)[-1] == "better than its parent: solves more"
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
    assert not [line for line in board if "hidden from the teacher" in line]
