import base64
import dataclasses
import functools
import gzip
import json
import logging
import math
import re
import time
import urllib.parse
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rsi import benchmarks, cards, core, generation, harness, paths, runs, stats, teacher, tree, variants, views

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("viewer.html")
LIVE_S = 600  # files changed this recently mean it's still running
# max characters sent per block
PROMPT_MAX = 40_000
DIFF_MAX = 80_000
_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def tree_data() -> dict:
    """Each lineage as a tree of harnesses, variants under the harness they came from"""
    now = time.time()
    nodes = _nodes()
    entries = variants.tree_load()
    by_name = {n.name: n for n in nodes}
    members: dict[str, list[core.Node]] = {}
    for n in sorted(nodes, key=lambda n: n.created):
        members.setdefault(tree.lineage(name=n.name, nodes=nodes)[0].name, []).append(n)
    roots = sorted((by_name[name] for name in members), key=lambda n: -n.created)
    lineages = [_lineage(root=r, members=members[r.name], entries=entries, nodes=nodes, now=now) for r in roots]
    return {"now": now, "areas": generation.area_texts(), "lineages": lineages}


def _lineage(
    root: core.Node,
    members: list[core.Node],
    entries: dict[str, variants.VariantEntry],
    nodes: list[core.Node],
    now: float,
) -> dict:
    probe = _probe(root=root)
    by_name = {n.name: n for n in members}
    groups: dict[str, list[core.Node]] = {}
    for n in members:
        groups.setdefault(variants.variant_name(reference=n.variant), []).append(n)
    made = {name: e for name, e in entries.items() if e.parent_node in by_name}
    top = variants.variant_name(reference=root.variant)
    cards = {
        name: _harness(name=name, own=groups.get(name, []), entry=entries.get(name), probe=probe, nodes=nodes, now=now)
        for name in [*groups, *(name for name in made if name not in groups)]
    }
    for name, card in sorted(cards.items(), key=lambda kv: kv[1]["created"]):
        if name == top:
            continue
        made_from = made[name].parent_node if name in made else groups[name][0].parent
        parent = variants.variant_name(reference=by_name[made_from].variant) if made_from in by_name else top
        cards[parent if parent in cards and parent != name else top]["children"].append(card)
    for n in members:
        live = [{"node": n.name, **r} for r in _rounds(node=n, entries=entries, now=now) if r["live"]]
        cards[variants.variant_name(reference=n.variant)]["teacher"] += live
    others = [card for name, card in cards.items() if name != top]
    hidden = runs.lineage_hidden(root=root)
    return {
        "model": root.model.name,
        "harness": f"{root.harness} {root.harness_version}",
        "benchmark": root.dataset.name,
        "root": _run_chip(node=root, probe=probe, now=now),
        "probe": {"tasks": len(probe), "hidden": len(hidden & set(probe))} if probe else None,
        "made": len(others),
        "ran": sum(any(run["total"] for run in card["runs"]) for card in others),
        "better": sum(bool(card["standing"] and card["standing"]["better_by"]) for card in others),
        "best": _best(groups=groups, top=top, probe=probe),
        "hours": sum(run.seconds for n in members for run in n.runs.values()) / 3600,
        "tree": cards[top],
    }


def _harness(
    name: str,
    own: list[core.Node],
    entry: variants.VariantEntry | None,
    probe: tuple[str, ...],
    nodes: list[core.Node],
    now: float,
) -> dict:
    """A harness card: its runs, solves per run and standing against its parent"""
    on_probe = [n for n in own if n.total and probe and tree.task_set(node=n) == probe]
    scored = on_probe or [n for n in own if n.total][-1:]
    found = variants.standing(entry=entry, nodes=nodes) if entry is not None else None
    return {
        "name": name,
        "title": _title(entry=entry, node=own[0] if own else None, name=name),
        "area": entry.area if entry is not None else "",
        "round": entry.round if entry is not None else "",
        "reuses": list(entry.reuses) if entry is not None else [],
        "changed": entry.diff if entry is not None else "",  # the files it changed, in one line
        "created": datetime.fromisoformat(entry.created).timestamp() if entry is not None else own[0].created,
        "href": f"#node/{own[0].name}" if own else f"#variant/{name}",
        "runs": [_run_chip(node=n, probe=probe, now=now) for n in own],
        "solved": sum(n.resolved for n in scored) / len(scored) if scored else None,
        "of": scored[0].total if scored else None,
        "scored": len(scored),
        "profile": _mean_profile(nodes=scored),
        "standing": _standing(found=found) if found is not None else None,
        "acted": variants.activations(entry=entry, nodes=nodes) if entry is not None else "",
        "children": [],
        "teacher": [],
    }


def _best(groups: dict[str, list[core.Node]], top: str, probe: tuple[str, ...]) -> dict | None:
    """The variant with the most solves per run on the probe set, vs the root's harness"""
    on_probe = {
        name: scored
        for name, own in groups.items()
        if (scored := [n for n in own if n.total and probe and tree.task_set(node=n) == probe])
    }
    baseline = on_probe.pop(top, [])
    if not baseline or not on_probe:
        return None
    name, best = max(on_probe.items(), key=lambda kv: sum(n.resolved for n in kv[1]) / len(kv[1]))
    return {
        "name": name,
        "solved": sum(n.resolved for n in best) / len(best),
        "baseline": sum(n.resolved for n in baseline) / len(baseline),
        "of": len(probe),
        "solves": _evidence(found=stats.evidence(variant=best, reference=baseline, tasks=list(probe))),
    }


def _standing(found: variants.Standing) -> dict:
    hidden = found.hidden
    return {
        "runs": found.runs,
        "reference_runs": found.reference_runs,
        "solved": found.solved,
        "reference_solved": found.reference_solved,
        "tasks": found.tasks,
        "solves": _evidence(found=found.solves),
        "hidden": {**_evidence(found=hidden), "tasks": hidden.tasks} if hidden is not None else None,
        "tokens": {"change": math.exp(-found.tokens.saving) - 1, "se": found.tokens.se, "z": found.tokens.z},
        "problems": found.problems,
        "broke": found.broke,
        "better_by": found.better_by,
    }


def _evidence(found: stats.Evidence) -> dict:
    return {"gain": found.gain, "se": found.se, "z": found.z, "verdict": stats.verdict(z=found.z)}


def _run_chip(node: core.Node, probe: tuple[str, ...], now: float) -> dict:
    return {
        "name": node.name,
        "label": variants.run_label(node=node),
        "count": len(node.task_ids) or node.tasks,
        "probe": bool(probe) and tree.task_set(node=node) == probe,
        "done": len(node.runs),
        "resolved": node.resolved,
        "total": node.total,
        "status": _status(node=node, live=_live_tasks(node=node, now=now), now=now),
    }


def _title(entry: variants.VariantEntry | None, node: core.Node | None, name: str) -> str:
    if entry is not None:
        # the card has the whole title, while tree.json keeps it cut short
        card = cards.read(package=paths.variants_dir() / entry.name, title_max=None)
        return card.title if card is not None else entry.title
    if node is not None and not node.variant:
        return f"{node.harness} {node.harness_version}, no changes"
    return (node.title if node is not None else None) or name


def _mean_profile(nodes: list[core.Node]) -> dict | None:
    profiles = [dataclasses.asdict(tree.profile(node=n)) for n in nodes]
    if not profiles:
        return None
    return {key: sum(p[key] for p in profiles) / len(profiles) for key in profiles[0]}


def _probe(root: core.Node) -> tuple[str, ...]:
    try:
        return tuple(sorted(runs.lineage_probe(root=root)))
    except FileNotFoundError:
        return ()


_PARSED: dict[Path, tuple[tuple[int, int], core.Node]] = {}


def _nodes() -> list[core.Node]:
    """Every node record, reparsing a file only when it changes"""
    folder = paths.runs_dir()
    found = []
    for path in sorted(folder.glob("*/node.json")) if folder.is_dir() else []:
        try:
            stat = path.stat()
        except OSError:
            continue
        key = (stat.st_mtime_ns, stat.st_size)
        if path not in _PARSED or _PARSED[path][0] != key:
            _PARSED[path] = (key, core.Node.model_validate_json(path.read_text()))
        found.append(_PARSED[path][1])
    return found


def node_data(name: str) -> dict:
    """Everything the page shows for a run"""
    now = time.time()
    nodes = _nodes()
    by_name = {n.name: n for n in nodes}
    node = by_name.get(name)
    if node is None:
        raise KeyError(f"no node {name!r}")
    entries = variants.tree_load()
    root = tree.lineage(name=node.name, nodes=nodes)[0]
    probe = _probe(root=root)
    variant = variants.variant_name(reference=node.variant)
    entry = entries.get(variant)
    parent = by_name.get(node.parent or "")
    reference = []
    if parent is not None:
        found = tree.reference_runs(parent=parent, tasks=tree.task_set(node=node), nodes=nodes)
        reference = [n for n in found if n.name != node.name]
    lineage = {n.name for n in nodes if tree.lineage(name=n.name, nodes=nodes)[0].name == root.name}
    card = _card(entry=entry) if entry is not None else None
    if entry is not None and card is not None:
        card["result"] = variants.scoreboard(entry=entry, nodes=nodes)
    hidden = runs.lineage_hidden(root=root)
    live = _live_tasks(node=node, now=now)
    return {
        **_run_chip(node=node, probe=probe, now=now),
        "variant": variant,
        "title": _title(entry=entry, node=node, name=variant),
        "area": entry.area if entry is not None else "",
        "parent": node.parent,
        "parent_label": variants.run_label(node=parent) if parent is not None else "",
        "model": node.model.name,
        "backend": node.model.backend,
        "harness": f"{node.harness} {node.harness_version}",
        "benchmark": node.dataset.name,
        "seed": node.seed,
        "created": node.created,
        "live": sorted(live),
        "hours": sum(run.seconds for run in node.runs.values()) / 3600,
        "profile": dataclasses.asdict(tree.profile(node=node)),
        "reference": {
            "runs": [variants.run_label(node=n) for n in reference],
            "solved": sum(n.resolved for n in reference) / len(reference),
            "profile": _mean_profile(nodes=reference),
        }
        if reference
        else None,
        "card": card,
        # a run of its parent's own harness has no diff
        "diff": _diff(variant=node.variant, parent=parent, harness_name=node.harness)
        if node.variant and (parent is None or parent.variant != node.variant)
        else "",
        "harness_runs": [
            _run_chip(node=n, probe=probe, now=now)
            for n in nodes
            if n.variant == node.variant and n.name != node.name and n.name in lineage
        ],
        "children": [
            {
                "name": e.name,
                "title": e.title,
                "area": e.area,
                "runs": [_run_chip(node=n, probe=probe, now=now) for n in variants.variant_nodes(entry=e, nodes=nodes)],
            }
            for e in sorted(entries.values(), key=lambda e: e.created)
            if e.parent_node == node.name
        ],
        "rounds": _rounds(node=node, entries=entries, now=now),
        "tasks": [
            _task_row(node=node, iid=iid, reference=reference, hidden=hidden, live=live, now=now)
            for iid in (node.task_ids or sorted(node.runs))
        ],
    }


def variant_data(name: str) -> dict:
    """A variant's card and diff, even before it runs"""
    entry = variants.tree_load().get(name)
    if entry is None:
        raise KeyError(f"no variant {name!r} in the tree")
    nodes = _nodes()
    parent = next((n for n in nodes if n.name == entry.parent_node), None)
    return {
        "name": name,
        "card": _card(entry=entry),
        "parent_label": variants.run_label(node=parent) if parent is not None else entry.parent_node,
        "diff": _diff(variant=name, parent=parent, harness_name=parent.harness if parent else "pi"),
        "nodes": [n.name for n in variants.variant_nodes(entry=entry, nodes=nodes)],
    }


def flips_data() -> dict:
    """Outcomes task by task for runs that share a task set, disagreements first"""
    nodes = sorted((n for n in _nodes() if n.total), key=lambda n: n.created)
    sets = dict.fromkeys((n.model.name, n.dataset.name, tree.task_set(node=n)) for n in nodes)
    groups = []
    for model, benchmark, tasks in sets:
        columns = [
            n for n in nodes if (n.model.name, n.dataset.name) == (model, benchmark) and set(tasks) <= n.runs.keys()
        ]
        if len(columns) < 2:
            continue
        rows = []
        for iid in tasks:
            cells = [_outcome(run=n.runs[iid]) for n in columns]
            rows.append({"iid": iid, "cells": cells, "flips": len({cell == "passed" for cell in cells}) > 1})
        groups.append(
            {
                "model": model,
                "benchmark": benchmark,
                "nodes": [
                    {
                        "name": n.name,
                        "variant": variants.variant_name(reference=n.variant),
                        "seed": n.seed,
                        "tasks": len(tree.task_set(node=n)),
                        "resolved": sum(n.runs[iid].resolved for iid in tasks),
                    }
                    for n in columns
                ],
                "tasks": sorted(rows, key=lambda row: (not row["flips"], row["iid"])),
            }
        )
    return {"groups": groups}


def _outcome(run: core.TaskRun) -> str:
    if run.resolved:
        return "passed"
    return run.status if run.status in ("error", "timeout") else "failed"


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _age(path: Path, now: float) -> float:
    mtime = _mtime(path=path)
    return float("inf") if mtime is None else now - mtime


def _quiet_for(folder: Path, now: float) -> float:
    """Seconds since anything directly in folder changed"""
    times = [_mtime(path=p) for p in folder.iterdir()] if folder.is_dir() else []
    return now - max((t for t in times if t is not None), default=-math.inf)


def _live_tasks(node: core.Node, now: float) -> set[str]:
    """Tasks whose agent is still writing"""
    folder = paths.instances_dir(node_name=node.name)
    if not folder.is_dir():
        return set()
    return {
        d.name
        for d in folder.iterdir()
        if (d.name not in node.runs or node.runs[d.name].status == "error") and _quiet_for(folder=d, now=now) < LIVE_S
    }


def _status(node: core.Node, live: set[str], now: float) -> str:
    if live:
        return "running"
    if node.total:
        return "done"
    busy = _age(path=paths.node_dir(name=node.name) / "rsi.log", now=now) < LIVE_S
    if len(node.runs) >= (len(node.task_ids) or node.tasks):
        return "scoring" if busy else "unscored"
    return "running" if busy else "stopped"


def _card(entry: variants.VariantEntry) -> dict:
    card = dataclasses.asdict(entry)
    card["changed"] = card.pop("diff")  # the one-line summary (the full diff is sent separately)
    return card


def _diff(variant: str, parent: core.Node | None, harness_name: str) -> str:
    """A variant package's diff against its parent's package (or the harness's own)"""
    try:
        package = variants.resolve_variant(reference=variant)
    except FileNotFoundError:
        return ""
    base = harness.load(name=harness_name).package_dir
    if parent is not None:
        base = variants.node_package(node=parent, baseline=base)
    return views.clip(text=variants.unified_diff(base=base, variant=package), limit=DIFF_MAX)


def _task_row(
    node: core.Node, iid: str, reference: list[core.Node], hidden: set[str], live: set[str], now: float
) -> dict:
    run = node.runs.get(iid)
    problem = paths.instances_dir(node_name=node.name) / iid / harness.TASK_FILE
    started = _mtime(path=problem)
    if iid in live:
        state = "running"
    elif run is None:
        state = "pending"
    elif run.status == "error":
        state = "error"
    elif not node.total:
        state = "unscored"
    else:
        state = "passed" if run.resolved else "failed"
    # fixed if every run of the parent's harness failed it, broken if every one passed it
    parent: list[int] | None = None
    flip = ""
    outcomes = [n.runs[iid].resolved for n in reference if iid in n.runs]
    if run is not None and node.total and outcomes:
        parent = [sum(outcomes), len(outcomes)]
        flip = "fixed" if run.resolved and not any(outcomes) else "broken" if not run.resolved and all(outcomes) else ""
    metrics = run.metrics if run is not None else core.TaskMetrics()
    if state == "running":
        seconds = now - started if started is not None else 0.0
    else:
        seconds = run.seconds if run is not None else 0.0
    lead = node.dataset.request.partition("{task}")[0]
    return {
        "iid": iid,
        "brief": _brief(path=str(problem), stamp=started, lead=lead) if started is not None else "",
        "state": state,
        "timeout": bool(run is not None and run.status == "timeout"),
        "flip": flip,
        "parent": parent,
        "hidden": iid in hidden,
        "seconds": round(seconds, 1),
        "requests": metrics.requests,
        "output_tokens": metrics.output_tokens,
        "stop": metrics.stop_reason,
        "patch_lines": metrics.patch_lines,
        "error": run.error if run is not None else None,
    }


@functools.lru_cache(maxsize=4096)
def _brief(path: str, stamp: float, lead: str) -> str:
    """The task's first real line after the benchmark's lead-in, usually its title (stamp is for the cache)"""
    for line in Path(path).read_text(errors="replace").removeprefix(lead).splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "```", "===")):
            return line[:140]
    return ""


def task_data(node_name: str, iid: str) -> dict:
    """A task's trajectory, system prompt and timing, and once done its diff and eval"""
    nodes = _nodes()
    node = next((n for n in nodes if n.name == _safe(name=node_name)), None)
    folder = paths.instances_dir(node_name=node_name) / _safe(name=iid)
    if node is None or not folder.is_dir():
        raise KeyError(f"no task {iid!r} in {node_name!r}")
    data = dataclasses.asdict(_view(axis="harness", name=node.harness, folder=str(folder), stamp=_stamp(folder=folder)))
    data["iid"] = iid
    told = harness.load(name=node.harness).system_context(instance_dir=folder)
    data["told"] = views.clip(text=told, limit=PROMPT_MAX)
    run = node.runs.get(iid)
    data["finished"] = run is not None
    data["patch"] = _parse_patch(text=run.patch) if run is not None and run.patch.strip() else None
    data["metrics"] = run.metrics.model_dump() if run is not None else None
    data["verdict"] = None
    if run is not None:
        judged = benchmarks.get(name=node.dataset.name).verdict(task_dir=folder) if node.total else None
        dot, label = _classify(run=run, judged=judged)
        shown = dataclasses.asdict(judged) if judged is not None else None
        data["verdict"] = {"dot": dot, "label": label, "seconds": run.seconds, "eval": shown, "error": run.error}
    # the same task in other runs of this model and benchmark, this harness first
    data["others"] = [
        {
            "name": n.name,
            "label": variants.run_label(node=n),
            "outcome": _task_outcome(node=n, iid=iid),
            "same": n.variant == node.variant,
        }
        for n in sorted(nodes, key=lambda n: (n.variant != node.variant, n.created))
        if n.name != node.name
        and iid in (n.task_ids or n.runs)
        and (n.model.name, n.dataset.name) == (node.model.name, node.dataset.name)
    ]
    return data


def _task_outcome(node: core.Node, iid: str) -> str:
    run = node.runs.get(iid)
    if run is None:
        return "pending"
    if run.status == "error" or node.total:
        return _outcome(run=run)
    return "unscored"


def _classify(run: core.TaskRun, judged: views.Verdict | None) -> tuple[str, str]:
    """A finished task's status dot and label"""
    if run.status == "error":
        return "err", f"error: {run.error}"
    if run.status == "timeout":
        return "warn", "timed out"
    if run.metrics.patch_lines == 0:
        return "warn", "no patch"
    if judged is None:
        return "pend", "not scored yet"
    if judged.infra:
        return "warn", "the grader failed to run"
    if not judged.applied:
        return "err", "patch didn't apply"
    if judged.resolved:
        return "ok", "resolved"
    return "bad", "not resolved"


def _parse_patch(text: str) -> dict:
    """A unified diff as a list of lines per file"""
    files: list[dict] = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1] if " b/" in line else line.split()[-1]
            files.append({"path": path.strip(), "added": 0, "removed": 0, "lines": []})
        elif not files:
            continue
        elif line.startswith("@@"):
            files[-1]["lines"].append(["hunk", line.strip()])
        elif line.startswith("+") and not line.startswith("+++"):
            files[-1]["lines"].append(["add", line[1:]])
            files[-1]["added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            files[-1]["lines"].append(["del", line[1:]])
            files[-1]["removed"] += 1
        elif line.startswith(" "):
            files[-1]["lines"].append(["ctx", line[1:]])
    added = sum(f["added"] for f in files)
    removed = sum(f["removed"] for f in files)
    return {"files": files, "stats": f"{len(files)} file{'' if len(files) == 1 else 's'} · +{added} −{removed}"}


def _rounds(node: core.Node, entries: dict[str, variants.VariantEntry], now: float) -> list[dict]:
    folder = paths.node_dir(name=node.name) / "generations"
    if not folder.is_dir():
        return []
    rounds = []
    for round_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        sessions = [
            _session_summary(round_dir=round_dir, session_dir=s, entries=entries, now=now)
            for s in sorted(round_dir.glob("session-*"), key=lambda p: int(p.name.split("-")[1]))
        ]
        rounds.append(
            {
                "id": round_dir.name,
                "live": any(s["status"] == "running" for s in sessions),
                "registered": sum(len(s["registered"]) for s in sessions),
                "sessions": sessions,
            }
        )
    return rounds


def _session_summary(round_dir: Path, session_dir: Path, entries: dict[str, variants.VariantEntry], now: float) -> dict:
    checks = round_dir / "checks" / session_dir.name
    checked = sorted(p.name for p in checks.iterdir() if p.is_dir()) if checks.is_dir() else []
    names = {c: variants.free_name(proposed=c, taken=set()) for c in checked}  # the names they got registered under
    registered = [c for c in checked if names[c] in entries]
    rejected = [c for c in checked if c not in registered]
    if registered:
        status = "registered"
    elif rejected:
        status = "rejected"
    elif _none(session_dir=session_dir) is not None:
        status = "declined"
    elif _quiet_for(folder=session_dir, now=now) < LIVE_S:
        status = "running"
    else:
        status = "no variant"
    _, n, area = session_dir.name.split("-", 2)
    return {
        "n": int(n),
        "area": area,
        "status": status,
        "registered": [names[c] for c in registered],
        "rejected": rejected,
    }


def _rejection(folder: Path) -> str:
    """Why a candidate was turned down, then the end of each check's log"""
    problems = folder / "PROBLEMS.md"
    parts = [problems.read_text(errors="replace")] if problems.is_file() else []
    for log_file in [folder / "check.log", *sorted((folder / "question").glob("*.log"))]:
        if log_file.is_file():
            parts.append(f"{log_file.relative_to(folder)}:\n{log_file.read_text(errors='replace')[-3_000:]}")
    return "\n\n".join(parts)


def _none(session_dir: Path) -> str | None:
    """What the session said when it made nothing, or None if it didn't write NONE.md"""
    out = session_dir / generation.TEACHER_OUT
    return generation.teacher_text(path=out / "variants" / generation.NONE, out=out)


def session_data(node_name: str, round_id: str, session: int) -> dict:
    """A teacher session's trajectory, prompt and check results"""
    round_dir = paths.node_dir(name=_safe(name=node_name)) / "generations" / _safe(name=round_id)
    folder = next(round_dir.glob(f"session-{session}-*"), None)
    if folder is None or not folder.is_dir():
        raise KeyError(f"no session {session} in round {round_id!r}")
    agent = json.loads((round_dir / generation.ROUND).read_text())["teacher"]["agent"]
    data = dataclasses.asdict(_view(axis="teacher", name=agent, folder=str(folder), stamp=_stamp(folder=folder)))
    prompt = folder / "prompt.md"
    shown = views.clip(text=prompt.read_text(errors="replace"), limit=PROMPT_MAX) if prompt.is_file() else ""
    data["instructions"] = shown
    outcome = _session_summary(round_dir=round_dir, session_dir=folder, entries=variants.tree_load(), now=time.time())
    checks = round_dir / "checks" / folder.name
    outcome["checks"] = {name: _rejection(folder=checks / name) for name in outcome["rejected"]}
    outcome["none"] = (_none(session_dir=folder) or "")[:4_000]
    data["outcome"] = outcome
    data["finished"] = outcome["status"] != "running"
    return data


def _stamp(folder: Path) -> tuple:
    """When each file directly in folder last changed, so a run's view is read again only once it has"""
    found = []
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        try:
            stat = path.stat()
        except OSError:
            continue  # gone since the listing
        found.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(found)


@functools.lru_cache(maxsize=64)
def _view(axis: str, name: str, folder: str, stamp: tuple) -> views.Trajectory:
    """A run (from its harness) or a teacher session (from its teacher), as of stamp"""
    if axis == "harness":
        return harness.load(name=name).trajectory(instance_dir=Path(folder))
    return teacher.load(name=name).trajectory(session_dir=Path(folder))


def _safe(name: str) -> str:
    """Allow only plain file names, since names come from the URL"""
    if not _SAFE.match(name) or name.startswith("."):
        raise KeyError(f"bad name {name!r}")
    return name


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # http.server needs this name
        url = urllib.parse.urlparse(self.path)
        query = {key: values[0] for key, values in urllib.parse.parse_qs(url.query).items()}
        routes = {
            "/api/tree": lambda: tree_data(),
            "/api/flips": lambda: flips_data(),
            "/api/node": lambda: node_data(name=query["name"]),
            "/api/variant": lambda: variant_data(name=query["name"]),
            "/api/task": lambda: task_data(node_name=query["node"], iid=query["iid"]),
            "/api/session": lambda: session_data(
                node_name=query["node"], round_id=query["round"], session=int(query["session"])
            ),
        }
        try:
            if url.path in ("/", "/index.html"):
                self._send(code=200, body=PAGE.read_bytes(), kind="text/html; charset=utf-8")
            elif url.path in routes:
                body = json.dumps(routes[url.path](), ensure_ascii=False).encode()
                self._send(code=200, body=body, kind="application/json")
            else:
                self._send(code=404, body=b"not found", kind="text/plain")
        except (KeyError, ValueError, FileNotFoundError) as exc:
            self._send(code=404, body=json.dumps({"error": str(exc)}).encode(), kind="application/json")

    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("content-type", kind)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # http.server needs this signature
        log.debug(f"{self.address_string()} {format % args}")


def _page(path: str, query: dict[str, object]) -> str:
    """The URL the page asks for, encoded like encodeURIComponent"""
    return f"{path}?{urllib.parse.urlencode(query, quote_via=urllib.parse.quote)}"


def save(path: Path, node_names: list[str]) -> int:
    """Save the viewer as one HTML file to share (trajectories only for node_names, if given)"""
    nodes = _nodes()
    known = {n.name for n in nodes}
    unknown = sorted(set(node_names) - known)
    if unknown:
        raise KeyError(f"no node {', '.join(unknown)}")
    chosen = set(node_names) or known
    pages: dict[str, object] = {"/api/tree": tree_data(), "/api/flips": flips_data()}
    for name in sorted(known):
        node = node_data(name=name)
        pages[_page(path="/api/node", query={"name": name})] = node
        if name not in chosen:
            continue
        for row in node["tasks"]:
            if row["state"] != "pending":
                key = _page(path="/api/task", query={"node": name, "iid": row["iid"]})
                pages[key] = task_data(node_name=name, iid=row["iid"])
        for r in node["rounds"]:
            for session in r["sessions"]:
                key = _page(path="/api/session", query={"node": name, "round": r["id"], "session": session["n"]})
                pages[key] = session_data(node_name=name, round_id=r["id"], session=session["n"])
    for name in variants.tree_load():
        pages[_page(path="/api/variant", query={"name": name})] = variant_data(name=name)
    packed = {
        key: base64.b64encode(gzip.compress(json.dumps(value, ensure_ascii=False).encode(), mtime=0)).decode()
        for key, value in pages.items()
    }
    data = json.dumps({"saved": time.time(), "pages": packed})
    embedded = f'<script type="application/json" id="saved">{data}</script>\n<script>'
    page = PAGE.read_text().replace("<script>", embedded, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path.stat().st_size


def serve(host: str, port: int, open_browser: bool) -> None:
    """Serve the viewer live, reading runs/ on every request"""
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{server.server_port}/"
    log.info(f"viewer at {url} (ctrl-c to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
