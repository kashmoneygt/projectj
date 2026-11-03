import time

from rsi import core, tree


def _node(
    name: str,
    parent: str | None = None,
    model: str = "gemma-4-12b",
    resolved: int = 0,
    total: int = 0,
    created: float = 0.0,
) -> core.Node:
    return core.Node(
        name=name,
        parent=parent,
        dataset=core.DatasetSpec(name="swebench-verified", dataset_id="princeton-nlp/SWE-bench_Verified", split="test"),
        harness="pi",
        harness_version="1.3.0",
        model=core.ModelSpec(name=model, source="hf:x", model_id="m"),
        budget=core.Budget(),
        tasks=10,
        seed=42,
        resolved=resolved,
        total=total,
        score=round(100.0 * resolved / total, 2) if total else 0.0,
        created=created or time.time(),
    )


def test_derive_node_name() -> None:
    prefix = "gemma-4-12b--pi--swebench-verified--"
    assert (
        tree.derive_node_name(
            model_name="gemma-4-12b",
            harness_name="pi",
            benchmark_name="swebench-verified",
            variant_name="baseline",
            tasks=10,
            seed=42,
        )
        == f"{prefix}baseline--10t--s42"
    )
    assert (
        tree.derive_node_name(
            model_name="gemma-4-12b",
            harness_name="pi",
            benchmark_name="swebench-verified",
            variant_name="test-before-edit",
            tasks=50,
            seed=42,
        )
        == f"{prefix}test-before-edit--50t--s42"
    )
    assert (
        tree.derive_node_name(
            model_name="qwen-3.8-27b",
            harness_name="pi",
            benchmark_name="swebench-verified",
            variant_name="run_tests_hook",
            tasks=0,
            seed=7,
        )
        == "qwen-3.8-27b--pi--swebench-verified--run_tests_hook--all--s7"
    )


def test_best_child_policy() -> None:
    assert tree.best_child(children=[_node(name="a")]) is None
    a = _node(name="a", resolved=1, total=10, created=100.0)
    b = _node(name="b", resolved=3, total=10, created=200.0)
    c = _node(name="c", resolved=3, total=10, created=50.0)
    assert tree.best_child(children=[a, b, c]).name == "c"
    d = _node(name="d", resolved=5, total=10, created=300.0)
    assert tree.best_child(children=[a, b, c, d]).name == "d"


def test_render_tree() -> None:
    prefix = "gemma-4-12b--pi--swebench-verified--"
    root = _node(name=prefix + "baseline--10t--s42", created=1.0, resolved=1, total=10)
    c1 = _node(name=prefix + "test-before-edit--10t--s42", parent=root.name, created=2.0, resolved=3, total=10)
    c2 = _node(name=prefix + "run-tests-hook--10t--s42", parent=root.name, created=3.0, resolved=2, total=10)
    other = _node(name="qwen-3.8-27b--pi--swebench-verified--baseline--10t--s42", created=4.0)
    lines = tree.render_tree(nodes=[c2, root, c1, other]).splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("├─ " + root.name)
    assert lines[1].startswith("│   ├─ " + c1.name) and lines[1].rstrip().endswith("*")
    assert lines[2].startswith("│   └─ " + c2.name) and "*" not in lines[2]
    assert lines[3].startswith("└─ " + other.name) and "*" not in lines[3]


def _scored(name: str, parent: str | None, variant: str, outcomes: dict[str, bool], created: float) -> core.Node:
    node = _node(name=name, parent=parent, created=created)
    node.variant = variant
    node.runs = {iid: core.TaskRun(instance_id=iid, status="ok", resolved=ok) for iid, ok in outcomes.items()}
    node.task_ids = sorted(outcomes)
    node.total, node.resolved = len(outcomes), sum(outcomes.values())
    return node


def test_lineage_walks_to_the_root() -> None:
    root = _node(name="root")
    mid = _node(name="mid", parent="root")
    leaf = _node(name="leaf", parent="mid")
    assert [n.name for n in tree.lineage(name="leaf", nodes=[leaf, root, mid])] == ["root", "mid", "leaf"]
    orphan = _node(name="orphan", parent="gone")
    assert [n.name for n in tree.lineage(name="orphan", nodes=[orphan])] == ["orphan"]
    assert tree.lineage(name="missing", nodes=[root]) == []


def test_reference_run_is_the_parent_variant_on_the_same_tasks() -> None:
    probe = {"a": False, "b": False, "c": True}
    root = _scored(name="root", parent=None, variant="", outcomes={**probe, "d": True}, created=1.0)
    child = _scored(
        name="child", parent="root", variant="/v/fix", outcomes={"a": True, "b": False, "c": True}, created=3.0
    )
    assert tree.reference_run(child=child, nodes=[root, child]) is None
    repeat = _scored(name="repeat", parent="root", variant="", outcomes=probe, created=2.0)
    assert tree.reference_run(child=child, nodes=[root, repeat, child]).name == "repeat"
    other_model = _scored(name="other-model", parent="root", variant="", outcomes=probe, created=4.0)
    other_model.model = core.ModelSpec(name="qwen-3.8-27b", source="hf:x", model_id="q")
    assert tree.reference_run(child=child, nodes=[root, repeat, child, other_model]).name == "repeat"
    c = tree.compare(child=child, reference=repeat)
    assert (c.resolved, c.reference_resolved, c.fixed, c.broken) == (2, 1, 1, 0)
    assert tree.compare(child=child, reference=None).reference is None


def test_the_reference_is_every_run_of_the_parents_harness_on_the_same_tasks() -> None:
    probe = {"a": False, "b": True}
    parent = _scored(name="parent", parent="root", variant="/v/fix", outcomes=probe, created=1.0)
    repeat = _scored(name="repeat", parent="parent", variant="/v/fix", outcomes=probe, created=2.0)
    sibling = _scored(name="sibling", parent="root", variant="/v/fix", outcomes=probe, created=3.0)
    child = _scored(name="child", parent="parent", variant="/v/next", outcomes=probe, created=4.0)
    fewer = _scored(name="fewer", parent="parent", variant="/v/fix", outcomes={"a": True}, created=5.0)
    unscored = _scored(name="unscored", parent="parent", variant="/v/fix", outcomes=probe, created=6.0)
    unscored.total = 0
    nodes = [parent, repeat, sibling, child, fewer, unscored]
    found = tree.reference_runs(parent=parent, tasks=("a", "b"), nodes=nodes)
    assert [n.name for n in found] == ["parent", "repeat", "sibling"]
    assert tree.reference_run(child=child, nodes=nodes).name == "sibling"
    assert tree.reference_run(child=repeat, nodes=[parent, repeat, child]).name == "parent"


def test_runs_compare_across_ports_but_not_across_model_settings() -> None:
    probe = {"a": False}
    root = _scored(name="root", parent=None, variant="", outcomes={**probe, "d": True}, created=1.0)
    moved = _scored(name="moved", parent="root", variant="", outcomes=probe, created=2.0)
    moved.model = moved.model.model_copy(update={"port": 9999})
    hotter = _scored(name="hotter", parent="root", variant="", outcomes=probe, created=3.0)
    hotter.model = hotter.model.model_copy(update={"sampling": {"temperature": 1.5}})
    found = tree.reference_runs(parent=root, tasks=("a",), nodes=[root, moved, hotter])
    assert [n.name for n in found] == ["moved"]
    reordered = _scored(name="reordered", parent="root", variant="", outcomes=probe, created=4.0)
    reordered.model = reordered.model.model_copy(update={"sampling": {"top_p": 0.9, "temperature": 1.0}})
    root.model = root.model.model_copy(update={"sampling": {"temperature": 1.0, "top_p": 0.9}})
    assert tree.comparable(a=root, b=reordered)


def test_runs_made_differently_say_what_differs() -> None:
    fix = _scored(name="fix", parent="root", variant="/v/fix", outcomes={"a": True}, created=1.0)
    fix.server, fix.contract, fix.variant_digest = {"engine": "1.0"}, "c1", "d1"
    again = fix.model_copy(update={"name": "again"})
    base = fix.model_copy(update={"name": "base", "variant": "", "variant_digest": ""})
    assert tree.made_differently(compared=[fix, again, base], packages=[[fix, again], [base]]) == []
    again.server, again.variant_digest = {"engine": "1.1"}, "d2"
    base.contract = "c2"
    assert tree.made_differently(compared=[fix, again, base], packages=[[fix, again], [base]]) == [
        'not the same model server: {"engine": "1.0"} on fix, base; {"engine": "1.1"} on again',
        'not the same harness contract: "c1" on fix, again; "c2" on base',
        'not the same package files: "d1" on fix; "d2" on again',
    ]


def test_sign_test() -> None:
    assert tree.sign_test(fixed=0, broken=0) == 1.0
    assert tree.sign_test(fixed=3, broken=3) == 1.0
    assert abs(tree.sign_test(fixed=10, broken=2) - 0.0386) < 1e-3
    assert tree.sign_test(fixed=2, broken=10) == tree.sign_test(fixed=10, broken=2)


def test_profile_counts_the_observables() -> None:
    node = _scored(name="n", parent=None, variant="", outcomes={"a": True, "b": False}, created=1.0)
    node.runs["a"].patch = "diff"
    node.runs["a"].seconds = 60.0
    node.runs["a"].metrics = core.TaskMetrics(output_tokens=1000, reasoning_tokens=500)
    node.runs["b"].metrics = core.TaskMetrics(
        stop_reason="length", requests=4, output_tokens=3000, reasoning_tokens=2500, contract_violations=["model x"]
    )
    node.runs["b"].status = "timeout"
    node.runs["b"].seconds = 180.0
    p = tree.profile(node=node)
    assert (p.capped, p.empty, p.stopped, p.requests, p.contract) == (1, 1, 1, 2.0, 1)
    assert (p.output_tokens, p.thinking, p.minutes) == (2000.0, 0.75, 2.0)
    change = tree.behavior_change(
        before=p, after=tree.profile(node=_scored(name="m", parent=None, variant="", outcomes={"a": True}, created=1.0))
    )
    assert change.startswith("runs ending cut at the output cap 1→0, runs without a patch 1→1")
    assert "thinking share 75%→0%" in change and change.endswith("minutes/task 2.0→0.0")


def test_render_tree_marks_best_per_task_set() -> None:
    root = _node(name="root", created=1.0, resolved=1, total=10)
    a = _scored(name="a", parent="root", variant="/v/a", outcomes={"x": True, "y": True}, created=2.0)
    b = _scored(name="b", parent="root", variant="/v/b", outcomes={"z": False, "w": True}, created=3.0)
    lines = tree.render_tree(nodes=[root, a, b]).splitlines()
    assert all(line.rstrip().endswith("*") for line in lines)
