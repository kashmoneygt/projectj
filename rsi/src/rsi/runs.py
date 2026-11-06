import json
import random
from collections import Counter
from pathlib import Path

import pydantic

from rsi import benchmarks, core, paths


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_node(node: core.Node) -> None:
    _atomic_write(path=paths.node_dir(name=node.name) / "node.json", text=node.model_dump_json(indent=2) + "\n")


def read_node(name: str) -> core.Node:
    return core.Node.model_validate_json((paths.node_dir(name=name) / "node.json").read_text())


def list_nodes() -> list[core.Node]:
    return [core.Node.model_validate_json(path.read_text()) for path in _records(folder=paths.runs_dir())]


def history_in(folder: Path) -> tuple[list[core.Node], list[str]]:
    """An archived runs/ folder's node records, and the names of any in a format this code no longer reads"""
    nodes: list[core.Node] = []
    unread: list[str] = []
    for path in _records(folder=folder):
        try:
            nodes.append(core.Node.model_validate_json(path.read_text()))
        except pydantic.ValidationError:
            unread.append(path.parent.name)
    return nodes, unread


def _records(folder: Path) -> list[Path]:
    return sorted(folder.glob("*/node.json")) if folder.is_dir() else []


def record_task_run(node_name: str, run: core.TaskRun) -> None:
    path = paths.instances_dir(node_name=node_name) / run.instance_id / "result.json"
    _atomic_write(path=path, text=run.model_dump_json(indent=2) + "\n")


def apply_score(node: core.Node, per_instance: dict[str, bool]) -> None:
    for instance_id, resolved in per_instance.items():
        run = node.runs.setdefault(instance_id, core.TaskRun(instance_id=instance_id))
        run.resolved = resolved
        record_task_run(node_name=node.name, run=run)
    node.total = len(per_instance)
    node.resolved = sum(per_instance.values())
    node.score = round(100.0 * node.resolved / node.total, 2) if node.total else 0.0
    write_node(node=node)


# enough successes to catch regressions
PROBE_SUCCESSES = 30
# these won't flip, and they're the slowest to run
NEVER_SOLVED_TRIES = 8


def probe_path(root: core.Node) -> Path:
    return paths.node_dir(name=root.name) / "probe.txt"


def hidden_path(root: core.Node) -> Path:
    return paths.node_dir(name=root.name) / "hidden.txt"


def _task_ids(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def lineage_probe(root: core.Node) -> list[str]:
    path = probe_path(root=root)
    if not path.is_file():
        raise FileNotFoundError(
            f"the lineage rooted at {root.name} has no probe set yet; write it once with "
            f"`rsi probe --root {root.name} --history <a folder of older runs>`"
        )
    return _task_ids(path=path)


def lineage_hidden(root: core.Node) -> set[str]:
    """Probe tasks hidden from the teacher, to check that gains hold out"""
    path = hidden_path(root=root)
    return set(_task_ids(path=path)) if path.is_file() else set()


def write_probe(root: core.Node, nodes: list[core.Node], repos: dict[str, str]) -> list[str]:
    """Write the probe set once: the root's failures and successes spread across the repos, half of each hidden"""
    skipped = never_solved(model=root.model.name, dataset=root.dataset.name, nodes=nodes)
    failed = sorted(iid for iid, run in root.runs.items() if not run.resolved)
    passed = sorted(iid for iid, run in root.runs.items() if run.resolved)
    failures = [iid for iid in failed if iid not in skipped]
    successes = benchmarks.spread(ids=passed, repos=repos, count=PROBE_SUCCESSES, seed=root.seed)
    probe = failures + successes
    pick = random.Random(root.seed)
    hidden = sorted(_half(ids=failures, repos=repos, pick=pick) + _half(ids=successes, repos=repos, pick=pick))
    left_out = [iid for iid in failed if iid in skipped]
    note = f"# left out, failed in {NEVER_SOLVED_TRIES}+ runs and never solved: {', '.join(left_out)}\n"
    _write_probe_lists(root=root, probe=probe, hidden=hidden, note=note if left_out else "")
    return probe


def _half(ids: list[str], repos: dict[str, str], pick: random.Random) -> list[str]:
    """Every other task within each repo, so the hidden half is spread across the repos like the rest"""
    return sorted(ids, key=lambda iid: (repos[iid], pick.random()))[1::2]


def carry_probe(root: core.Node, probe: list[str], hidden: list[str]) -> list[str]:
    """Write a probe set picked elsewhere, like another lineage's, with the tasks it hides from the teacher"""
    missing = [iid for iid in probe if iid not in root.runs]
    if missing:
        raise ValueError(f"{root.name} didn't run {len(missing)} of these tasks: {', '.join(missing[:5])}")
    stray = sorted(set(hidden) - set(probe))
    if stray:
        raise ValueError(f"hidden tasks that aren't in the probe set: {', '.join(stray[:5])}")
    _write_probe_lists(root=root, probe=probe, hidden=sorted(hidden), note="# picked before this lineage started\n")
    return probe


def _write_probe_lists(root: core.Node, probe: list[str], hidden: list[str], note: str) -> None:
    header = f"# probe set of the lineage rooted at {root.name}: {len(probe)} tasks (seed {root.seed})\n{note}"
    # hidden half first, so a crash can't leave a probe set without one
    hidden_header = f"# hidden from the teacher: {len(hidden)} tasks of {root.name}'s probe set (seed {root.seed})\n"
    _atomic_write(path=hidden_path(root=root), text=hidden_header + "\n".join(hidden) + "\n")
    _atomic_write(path=probe_path(root=root), text=header + "\n".join(probe) + "\n")


def never_solved(model: str, dataset: str, nodes: list[core.Node]) -> set[str]:
    """Tasks failed in NEVER_SOLVED_TRIES or more scored runs and never solved"""
    tries: Counter[str] = Counter()
    solved: set[str] = set()
    for node in nodes:
        if node.model.name != model or node.dataset.name != dataset or node.total == 0:
            continue
        for iid, run in node.runs.items():
            if run.status in ("ok", "timeout"):
                tries[iid] += 1
                if run.resolved:
                    solved.add(iid)
    return {iid for iid, count in tries.items() if count >= NEVER_SOLVED_TRIES and iid not in solved}


def write_preds(path: Path, runs: dict[str, core.TaskRun], model_name: str) -> None:
    # an errored run submits nothing
    preds = {
        instance_id: {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": run.patch if run.status != "error" else "",
        }
        for instance_id, run in runs.items()
    }
    _atomic_write(path=path, text=json.dumps(preds, indent=2) + "\n")
