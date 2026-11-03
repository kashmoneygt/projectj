import json
import random
from collections import Counter
from pathlib import Path

from rsi import core, paths


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
    return nodes_in(folder=paths.runs_dir())


def nodes_in(folder: Path) -> list[core.Node]:
    """Node records in a runs/ folder or an archived copy"""
    if not folder.is_dir():
        return []
    return [
        core.Node.model_validate_json((d / "node.json").read_text())
        for d in sorted(folder.iterdir())
        if (d / "node.json").is_file()
    ]


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


def write_probe(root: core.Node, nodes: list[core.Node]) -> list[str]:
    """Write the probe set once: the root's failures and sampled successes, half of each hidden"""
    skipped = never_solved(model=root.model.name, dataset=root.dataset.name, nodes=nodes)
    failed = sorted(iid for iid, run in root.runs.items() if not run.resolved)
    passed = sorted(iid for iid, run in root.runs.items() if run.resolved)
    pick = random.Random(root.seed)
    failures = [iid for iid in failed if iid not in skipped]
    successes = pick.sample(passed, min(PROBE_SUCCESSES, len(passed)))
    probe = failures + successes
    hidden = sorted(pick.sample(failures, len(failures) // 2) + pick.sample(successes, len(successes) // 2))
    left_out = [iid for iid in failed if iid in skipped]
    header = f"# probe set of the lineage rooted at {root.name}: {len(probe)} tasks (seed {root.seed})\n"
    if left_out:
        header += f"# left out, failed in {NEVER_SOLVED_TRIES}+ runs and never solved: {', '.join(left_out)}\n"
    # hidden half first, so a crash can't leave a probe set without one
    hidden_header = f"# hidden from the teacher: {len(hidden)} tasks of {root.name}'s probe set (seed {root.seed})\n"
    _atomic_write(path=hidden_path(root=root), text=hidden_header + "\n".join(hidden) + "\n")
    _atomic_write(path=probe_path(root=root), text=header + "\n".join(probe) + "\n")
    return probe


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
