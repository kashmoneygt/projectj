import json
from pathlib import Path

import pydantic
import pytest

from rsi import benchmarks, core, paths, runs


def _node() -> core.Node:
    return core.Node(
        name="t-node",
        dataset=core.DatasetSpec(name="swebench-verified", dataset_id="princeton-nlp/SWE-bench_Verified"),
        harness="pi",
        harness_version="1.3.0",
        model=core.ModelSpec(name="gemma-4-12b", source="path:/tmp/x.gguf", model_id="gemma-4-12b-it"),
        budget=core.Budget(),
        tasks=3,
        seed=1,
    )


def test_node_roundtrip_and_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    node = _node()
    runs.write_node(node=node)
    assert runs.read_node(name="t-node").name == "t-node"
    names = [n.name for n in runs.list_nodes()]
    assert names == ["t-node"]


def test_record_task_run_and_apply_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    node = _node()
    runs.write_node(node=node)
    run = core.TaskRun(instance_id="a__b-1", status="ok", patch="diff", seconds=1.0)
    runs.record_task_run(node_name="t-node", run=run)
    assert (paths.instances_dir(node_name="t-node") / "a__b-1" / "result.json").exists()

    node.runs["a__b-1"] = run
    runs.apply_score(node=node, per_instance={"a__b-1": True, "c__d-2": False})
    assert (node.total, node.resolved, node.score) == (2, 1, 50.0)
    assert node.runs["c__d-2"].resolved is False
    assert runs.read_node(name="t-node").score == 50.0


def test_write_preds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    runs.write_preds(
        path=tmp_path / "preds.json",
        runs={
            "a__b-1": core.TaskRun(instance_id="a__b-1", status="ok", patch="diff a\n"),
            "c__d-2": core.TaskRun(instance_id="c__d-2", status="error", patch="diff c\n"),
        },
        model_name="gemma-4-12b",
    )
    data = json.loads((tmp_path / "preds.json").read_text())
    assert data["a__b-1"]["model_name_or_path"] == "gemma-4-12b"
    assert data["a__b-1"]["model_patch"] == "diff a\n"
    assert data["a__b-1"]["instance_id"] == "a__b-1"
    assert data["c__d-2"]["model_patch"] == ""


def test_history_reads_an_archive_and_says_which_records_it_cant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    runs.write_node(node=_node())
    (tmp_path / "runs" / "gold").mkdir()
    (tmp_path / "runs" / "older").mkdir()
    older = json.loads((tmp_path / "runs" / "t-node" / "node.json").read_text())
    older["model"]["thinking"] = "medium"  # how records said it before thinking was on or off
    (tmp_path / "runs" / "older" / "node.json").write_text(json.dumps(older))
    nodes, unread = runs.history_in(folder=tmp_path / "runs")
    assert ([n.name for n in nodes], unread) == (["t-node"], ["older"])
    assert runs.history_in(folder=tmp_path / "missing") == ([], [])
    # the live runs folder is read strictly, so a record it can't read isn't skipped quietly
    with pytest.raises(pydantic.ValidationError):
        runs.list_nodes()


def _scored(name: str, outcomes: dict[str, bool], status: str = "ok") -> core.Node:
    node = _node().model_copy(update={"name": name})
    node.runs = {iid: core.TaskRun(instance_id=iid, status=status, resolved=ok) for iid, ok in outcomes.items()}
    node.total, node.resolved = len(outcomes), sum(outcomes.values())
    return node


def _repos(node: core.Node) -> dict[str, str]:
    """Each task's repo, from the part of its id before the first dash"""
    return {iid: iid.split("-")[0] for iid in node.runs}


def test_probe_is_the_failures_plus_sampled_successes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    monkeypatch.setattr(runs, "PROBE_SUCCESSES", 2)
    root = _scored(name="root", outcomes={"a__b-1": False, "b__c-2": True, "c__d-3": True, "d__e-4": True})
    probe = runs.write_probe(root=root, nodes=[root], repos=_repos(node=root))
    assert probe[0] == "a__b-1" and len(probe) == 3
    assert set(probe[1:]) <= {"b__c-2", "c__d-3", "d__e-4"}
    assert runs.lineage_probe(root=root) == probe
    assert runs.write_probe(root=root, nodes=[root], repos=_repos(node=root)) == probe


def test_probe_leaves_out_tasks_the_model_never_solves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    root = _scored(name="root", outcomes={"hard-1": False, "flip-2": False, "easy-3": True})
    history = [_scored(name=f"run-{i}", outcomes={"hard-1": False, "flip-2": i == 0}) for i in range(8)]
    history.append(_scored(name="errored", outcomes={"hard-1": False}, status="error"))
    unscored = _scored(name="unscored", outcomes={"flip-2": False})
    unscored.total = 0
    nodes = [root, *history, unscored]
    assert runs.never_solved(model="gemma-4-12b", dataset="swebench-verified", nodes=nodes) == {"hard-1"}
    assert runs.write_probe(root=root, nodes=nodes, repos=_repos(node=root)) == ["flip-2", "easy-3"]
    text = (paths.node_dir(name="root") / "probe.txt").read_text()
    assert text.startswith("# probe set of the lineage rooted at root: 2 tasks (seed 1)\n")
    assert "never solved: hard-1\n" in text
    assert runs.lineage_probe(root=root) == ["flip-2", "easy-3"]


def test_a_lineage_without_a_probe_set_says_how_to_write_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    root = _scored(name="root", outcomes={"a__b-1": False})
    with pytest.raises(FileNotFoundError, match="`rsi probe --root root"):
        runs.lineage_probe(root=root)
    assert not (paths.node_dir(name="root") / "probe.txt").exists()


def test_a_probe_set_picked_elsewhere_is_kept_as_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    root = _scored(name="root", outcomes={"a-1": True, "b-2": False, "c-3": True})
    with pytest.raises(ValueError, match="didn't run 1 of these tasks: z-9"):
        runs.carry_probe(root=root, probe=["a-1", "z-9"], hidden=["a-1"])
    with pytest.raises(ValueError, match="aren't in the probe set: c-3"):
        runs.carry_probe(root=root, probe=["a-1", "b-2"], hidden=["c-3"])
    assert not runs.probe_path(root=root).exists()
    assert runs.carry_probe(root=root, probe=["c-3", "a-1"], hidden=["c-3"]) == ["c-3", "a-1"]
    assert runs.lineage_probe(root=root) == ["c-3", "a-1"]
    assert runs.lineage_hidden(root=root) == {"c-3"}
    assert "# picked before this lineage started\n" in runs.probe_path(root=root).read_text()


def test_the_probe_hides_half_its_failures_and_half_its_successes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    monkeypatch.setattr(runs, "PROBE_SUCCESSES", 4)
    passed = [f"pass-{i}" for i in range(6)]
    root = _scored(name="root", outcomes={f"fail-{i}": False for i in range(5)} | dict.fromkeys(passed, True))
    assert runs.lineage_hidden(root=root) == set()
    probe = runs.write_probe(root=root, nodes=[root], repos=_repos(node=root))
    assert probe[5:] == benchmarks.spread(ids=passed, repos=_repos(node=root), count=4, seed=root.seed)
    hidden = runs.lineage_hidden(root=root)
    assert hidden <= set(probe)
    assert sum(t.startswith("fail") for t in hidden) == 2 and sum(t.startswith("pass") for t in hidden) == 2
    text = (paths.node_dir(name="root") / "hidden.txt").read_text()
    assert text.startswith("# hidden from the teacher: 4 tasks of root's probe set (seed 1)\n")
    runs.write_probe(root=root, nodes=[root], repos=_repos(node=root))
    assert runs.lineage_hidden(root=root) == hidden


def test_the_probe_spreads_its_successes_and_hidden_half_across_the_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    monkeypatch.setattr(runs, "PROBE_SUCCESSES", 8)
    # one big repo that passes a lot, and three small ones, like Django against the rest
    outcomes = {f"big-{i}": True for i in range(30)} | {f"{r}-{i}": True for r in ("x", "y", "z") for i in range(2)}
    outcomes |= {f"big-f{i}": False for i in range(4)} | {f"x-f{i}": False for i in range(2)}
    root = _scored(name="root", outcomes=outcomes)
    probe = runs.write_probe(root=root, nodes=[root], repos=_repos(node=root))
    successes = [iid for iid in probe if root.runs[iid].resolved]
    # by size alone the big repo would take nearly all 8, and a small one could be left out
    assert {iid.split("-")[0] for iid in successes} == {"big", "x", "y", "z"}
    hidden = runs.lineage_hidden(root=root)
    for repo in ("big", "x"):
        failures = [iid for iid in probe if iid.startswith(f"{repo}-f")]
        assert sum(iid in hidden for iid in failures) == len(failures) // 2
