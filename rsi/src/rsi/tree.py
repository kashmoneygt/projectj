import dataclasses
import json
import math

from rsi import core


def derive_node_name(
    model_name: str,
    harness_name: str,
    benchmark_name: str,
    variant_name: str,
    tasks: int,
    seed: int,
) -> str:
    scale = f"{tasks}t" if tasks > 0 else "all"
    return f"{model_name}--{harness_name}--{benchmark_name}--{variant_name}--{scale}--s{seed}"


def task_set(node: core.Node) -> tuple[str, ...]:
    return tuple(sorted(node.task_ids or node.runs))


def _setup(node: core.Node) -> tuple:
    """Settings runs must share to be compared, ignoring the port"""
    model = json.dumps(node.model.model_dump(mode="json", exclude={"port"}), sort_keys=True)
    return (model, node.harness, node.harness_version, node.dataset.name, node.dataset.revision, node.budget)


def comparable(a: core.Node, b: core.Node) -> bool:
    return _setup(node=a) == _setup(node=b)


def lineage(name: str, nodes: list[core.Node]) -> list[core.Node]:
    """The node and its ancestors, root first (up to a missing parent)"""
    by_name = {n.name: n for n in nodes}
    chain: list[core.Node] = []
    current = by_name.get(name)
    while current is not None and current.name not in {n.name for n in chain}:
        chain.append(current)
        current = by_name.get(current.parent) if current.parent else None
    return chain[::-1]


def runs_of(variant: str, tasks: tuple[str, ...], like: core.Node, nodes: list[core.Node]) -> list[core.Node]:
    """A variant's scored runs on exactly these tasks that compare with like"""
    setup = _setup(node=like)
    return [
        n
        for n in nodes
        if n.variant == variant and n.total > 0 and task_set(node=n) == tasks and _setup(node=n) == setup
    ]


def reference_runs(parent: core.Node, tasks: tuple[str, ...], nodes: list[core.Node]) -> list[core.Node]:
    """Every run of the parent's harness on these tasks (what its children are compared with)"""
    return runs_of(variant=parent.variant, tasks=tasks, like=parent, nodes=nodes)


def reference_run(child: core.Node, nodes: list[core.Node]) -> core.Node | None:
    """The latest reference run, for one-line comparisons"""
    parent = next((n for n in nodes if n.name == child.parent), None)
    if parent is None:
        return None
    found = [n for n in reference_runs(parent=parent, tasks=task_set(node=child), nodes=nodes) if n.name != child.name]
    return max(found, key=lambda n: n.created, default=None)


def _differences(field: str, values: dict[str, object]) -> str:
    """Which runs recorded which value, or "" if they all match"""
    by_value: dict[str, list[str]] = {}
    for name, value in values.items():
        by_value.setdefault(json.dumps(value, sort_keys=True), []).append(name)
    if len(by_value) < 2:
        return ""
    return f"not the same {field}: " + "; ".join(f"{value} on {', '.join(names)}" for value, names in by_value.items())


def made_differently(compared: list[core.Node], packages: list[list[core.Node]]) -> list[str]:
    """Why these runs can't be compared (server, contract or package files differ)"""
    problems = [
        _differences(field="model server", values={n.name: n.server for n in compared}),
        _differences(field="harness contract", values={n.name: n.contract for n in compared}),
    ]
    for group in packages:
        problems.append(_differences(field="package files", values={n.name: n.variant_digest for n in group}))
    return [p for p in problems if p]


def sign_test(fixed: int, broken: int) -> float:
    """Two-sided exact sign test p-value for fixed vs broken"""
    flips = fixed + broken
    tail = sum(math.comb(flips, k) for k in range(min(fixed, broken) + 1)) / 2**flips
    return min(1.0, 2 * tail)


@dataclasses.dataclass(frozen=True)
class Comparison:
    node: str
    reference: str | None  # the parent's run on the same tasks, if any
    resolved: int
    reference_resolved: int
    fixed: int  # resolved here, not in the reference
    broken: int  # resolved in the reference, not here
    p_value: float  # sign test on fixed vs broken


def compare(child: core.Node, reference: core.Node | None) -> Comparison:
    if reference is None:
        return Comparison(
            node=child.name,
            reference=None,
            resolved=child.resolved,
            reference_resolved=0,
            fixed=0,
            broken=0,
            p_value=1.0,
        )
    shared = set(child.runs) & set(reference.runs)
    fixed = sum(1 for t in shared if child.runs[t].resolved and not reference.runs[t].resolved)
    broken = sum(1 for t in shared if reference.runs[t].resolved and not child.runs[t].resolved)
    return Comparison(
        node=child.name,
        reference=reference.name,
        resolved=child.resolved,
        reference_resolved=sum(1 for t in shared if reference.runs[t].resolved),
        fixed=fixed,
        broken=broken,
        p_value=sign_test(fixed=fixed, broken=broken),
    )


@dataclasses.dataclass(frozen=True)
class Profile:
    """How a node's runs behaved, apart from the score"""

    capped: int  # runs whose last request hit the output cap
    empty: int  # runs without a patch
    stopped: int  # runs that timed out or errored
    requests: float  # mean per task
    output_tokens: float  # mean per task
    thinking: float  # thinking share of output tokens
    minutes: float  # mean per task
    contract: int  # tasks with requests off the pinned model settings


def profile(node: core.Node) -> Profile:
    runs = list(node.runs.values())
    count = max(len(runs), 1)
    output = sum(r.metrics.output_tokens for r in runs)
    return Profile(
        capped=sum(r.metrics.stop_reason == "length" for r in runs),
        empty=sum(not r.patch.strip() for r in runs),
        stopped=sum(r.status in ("timeout", "error") for r in runs),
        requests=sum(r.metrics.requests for r in runs) / count,
        output_tokens=output / count,
        thinking=sum(r.metrics.reasoning_tokens for r in runs) / max(output, 1),
        minutes=sum(r.seconds for r in runs) / 60 / count,
        contract=sum(bool(r.metrics.contract_violations) for r in runs),
    )


def behavior_change(before: Profile, after: Profile) -> str:
    """How behavior changed from the reference's runs to the child's"""
    return ", ".join(
        [
            f"runs ending cut at the output cap {before.capped}→{after.capped}",
            f"runs without a patch {before.empty}→{after.empty}",
            f"timeouts/errors {before.stopped}→{after.stopped}",
            f"requests/task {before.requests:.0f}→{after.requests:.0f}",
            f"output tokens/task {before.output_tokens / 1000:.1f}k→{after.output_tokens / 1000:.1f}k",
            f"thinking share {before.thinking:.0%}→{after.thinking:.0%}",
            f"minutes/task {before.minutes:.1f}→{after.minutes:.1f}",
        ]
    )


def best_child(children: list[core.Node]) -> core.Node | None:
    """Highest score, then most resolved, then earliest"""
    return max((n for n in children if n.total > 0), key=lambda n: (n.score, n.resolved, -n.created), default=None)


def render_tree(nodes: list[core.Node], model_filter: str = "") -> str:
    """One line per node with its score, '*' marking the best of each comparable group"""
    shown = sorted((n for n in nodes if n.model.name.startswith(model_filter)), key=lambda n: n.created)
    names = {n.name for n in shown}
    children: dict[str | None, list[core.Node]] = {}
    groups: dict[tuple, list[core.Node]] = {}
    for n in shown:
        children.setdefault(n.parent if n.parent in names else None, []).append(n)
        if n.total > 0:
            groups.setdefault((task_set(node=n), _setup(node=n)), []).append(n)
    best = {b.name for group in groups.values() if (b := best_child(children=group))}

    lines: list[str] = []
    width = max((len(n.name) for n in shown), default=0) + 2

    def walk(parent: str | None, prefix: str) -> None:
        group = children.get(parent, [])
        for i, n in enumerate(group):
            last = i == len(group) - 1
            star = " *" if n.name in best else ""
            score = f"{n.resolved}/{n.total} ({n.score:.0f}%)" if n.total > 0 else "-"
            lines.append(f"{prefix}{'└─ ' if last else '├─ '}{n.name:<{width}} {score}{star}")
            walk(parent=n.name, prefix=prefix + ("    " if last else "│   "))

    walk(parent=None, prefix="")
    return "\n".join(lines)
