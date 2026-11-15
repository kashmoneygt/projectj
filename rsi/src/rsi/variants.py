import dataclasses
import difflib
import hashlib
import json
import re
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path

from rsi import cards, core, harness, paths, runs, stats, tree


def safe_variant_name(name: str) -> str:
    # '--' separates the parts of a node name
    if not cards.NAME.match(name) or "--" in name:
        raise ValueError(
            f"variant name {name!r} must be a short lowercase name (hyphens/underscores ok, no '--'), "
            "e.g. 'test-before-edit'"
        )
    return name


def free_name(proposed: str, taken: Collection[str]) -> str:
    """Make the teacher's name for a variant valid and unique (-2, -3, ...)"""
    # baseline is the harness as it ships, and parent holds the parent's runs in the teacher's /work/runs/
    reserved = {*taken, "baseline", "parent"}
    base = "-".join(re.findall(r"[a-z0-9]+", proposed.lower()))[:40].strip("-") or "variant"
    if not base[0].isalpha():
        base = f"v-{base}"[:40].rstrip("-")
    name, n = base, 2
    while name in reserved:
        suffix = f"-{n}"
        name, n = base[: 40 - len(suffix)].rstrip("-") + suffix, n + 1
    return name


def is_name(reference: str) -> bool:
    """Check if a --variant value is a name rather than a path"""
    return bool(reference) and "/" not in reference and "\\" not in reference and not Path(reference).suffix


def variant_name(reference: str) -> str:
    """The variant name for a --variant value ('baseline' if empty)"""
    if not reference:
        return "baseline"
    if is_name(reference=reference):
        return safe_variant_name(name=reference)
    return Path(reference).name


def resolve_variant(reference: str) -> Path:
    """The package folder for a --variant value"""
    if not is_name(reference=reference):
        if not Path(reference).is_dir():
            raise FileNotFoundError(f"variant package dir {reference!r} does not exist")
        return Path(reference).resolve()
    registry = paths.variants_dir()
    variant = registry / safe_variant_name(name=reference)
    if not variant.is_dir():
        known = sorted(p.name for p in registry.iterdir() if p.is_dir()) if registry.is_dir() else []
        raise FileNotFoundError(f"unknown variant {reference!r}; known variants: {', '.join(known) or 'none'}")
    return variant


def reference(package: Path) -> str:
    """What a node records for its variant (a registered one's name, so records survive a move, or else its path)"""
    package = package.resolve()
    return package.name if package.parent == paths.variants_dir().resolve() else str(package)


def run_label(node: core.Node) -> str:
    """A run by its variant and seed, without the model and benchmark names its node has"""
    return f"{variant_name(reference=node.variant)} s{node.seed}"


def node_package(node: core.Node, baseline: Path) -> Path:
    return resolve_variant(reference=node.variant) if node.variant else baseline


def package_files(root: Path) -> dict[str, Path]:
    """A package's files, without its card and dotfiles"""
    return {
        p.relative_to(root).as_posix(): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != cards.FILE and not any(part.startswith(".") for part in p.relative_to(root).parts)
    }


def same_package(a: Path, b: Path) -> bool:
    files_a, files_b = package_files(root=a), package_files(root=b)
    return files_a.keys() == files_b.keys() and all(
        files_a[name].read_bytes() == files_b[name].read_bytes() for name in files_a
    )


def package_digest(package: Path) -> str:
    """Hash of the files a run gets from a package"""
    digest = hashlib.sha256()
    files = package_files(root=package)
    for name in sorted(files):
        content = files[name].read_bytes()
        digest.update(f"{name}\0{len(content)}\0".encode())
        digest.update(content)
    return digest.hexdigest()


DIFF_ENTRIES = 12  # files listed in a diff summary before "+N more"


def diff_summary(base: Path, variant: Path) -> str:
    """One line of the files a variant added (+), removed (-) or changed (~)"""
    base_files, variant_files = package_files(root=base), package_files(root=variant)
    entries: list[str] = []
    for name in sorted(set(base_files) | set(variant_files)):
        if name not in base_files:
            entries.append(f"+ {name}")
        elif name not in variant_files:
            entries.append(f"- {name}")
        else:
            changed = _changed_lines(before=base_files[name].read_bytes(), after=variant_files[name].read_bytes())
            if changed is not None:
                entries.append(f"~ {name} (+{changed[0]} -{changed[1]})")
    if not entries:
        return "identical"
    if len(entries) > DIFF_ENTRIES:
        entries = [*entries[:DIFF_ENTRIES], f"+{len(entries) - DIFF_ENTRIES} more"]
    return ", ".join(entries)


def _changed_lines(before: bytes, after: bytes) -> tuple[int, int] | None:
    if before == after:
        return None
    diff = difflib.unified_diff(
        before.decode("utf-8", "replace").splitlines(), after.decode("utf-8", "replace").splitlines(), lineterm=""
    )
    lines = [line for line in diff if line[:3] not in ("+++", "---") and not line.startswith("@@")]
    return sum(line.startswith("+") for line in lines), sum(line.startswith("-") for line in lines)


def unified_diff(base: Path, variant: Path) -> str:
    base_files, variant_files = package_files(root=base), package_files(root=variant)
    chunks: list[str] = []
    for name in sorted(set(base_files) | set(variant_files)):
        before = base_files[name].read_text(errors="replace").splitlines() if name in base_files else []
        after = variant_files[name].read_text(errors="replace").splitlines() if name in variant_files else []
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{name}" if name in base_files else "/dev/null",
                tofile=f"b/{name}" if name in variant_files else "/dev/null",
                lineterm="",
            )
        )
    return "\n".join(chunks)


@dataclasses.dataclass(frozen=True)
class VariantEntry:
    """A variant in the tree: its card and origin (results live in node records)"""

    name: str
    parent: str  # parent variant ('baseline' for the first round)
    parent_node: str  # node it was generated from
    created: str  # ISO-8601 timestamp
    created_by: str  # "teacher:<name>", or "human"
    title: str
    what: str
    why: str
    expect: str
    reuses: list[str]  # variants whose ideas it builds on
    diff: str  # diff_summary vs the parent's package
    round: str = ""  # a folder in the parent node's generations/
    session: int = 0  # its session's number in that round
    # added by hand in tree.json
    note: str = ""


def tree_load() -> dict[str, VariantEntry]:
    path = paths.tree_path()
    if not path.is_file():
        return {}
    return {name: VariantEntry(**raw) for name, raw in json.loads(path.read_text()).items()}


def tree_save(entries: dict[str, VariantEntry]) -> None:
    data = {name: dataclasses.asdict(entry) for name, entry in sorted(entries.items())}
    runs.atomic_write(path=paths.tree_path(), text=json.dumps(data, indent=2) + "\n")


def tree_register(
    name: str,
    parent: str,
    parent_node: str,
    card: cards.VariantCard,
    diff: str,
    created_by: str = "human",
    round: str = "",
    session: int = 0,
) -> VariantEntry:
    safe_variant_name(name=name)
    entries = tree_load()
    if name in entries:
        raise ValueError(f"variant {name!r} is already in the tree; variants are never overwritten")
    unknown = [name for name in card.reuses if name not in entries]
    if unknown:
        raise ValueError(f"it reuses {unknown}, which aren't in the tree")
    entry = VariantEntry(
        name=name,
        parent=parent,
        parent_node=parent_node,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        created_by=created_by,
        title=card.title,
        what=card.what,
        why=card.why,
        expect=card.expect,
        reuses=list(card.reuses),
        diff=diff,
        round=round,
        session=session,
    )
    entries[name] = entry
    tree_save(entries=entries)
    return entry


def variant_nodes(entry: VariantEntry, nodes: list[core.Node]) -> list[core.Node]:
    """A variant's runs, oldest first"""
    return sorted(
        (
            n
            for n in nodes
            if n.variant and n.parent == entry.parent_node and variant_name(reference=n.variant) == entry.name
        ),
        key=lambda n: n.created,
    )


def result_line(node: core.Node, nodes: list[core.Node]) -> str:
    """A run's score vs its parent's run on the same tasks, plus any change in behavior"""
    if node.total == 0:
        return "not scored yet"
    line = f"{node.resolved}/{node.total}"
    reference = tree.reference_run(child=node, nodes=nodes)
    if reference is None:
        return line
    solved = sum(reference.runs[task].resolved for task in set(node.runs) & set(reference.runs))
    change = tree.behavior_change(before=tree.profile(node=reference), after=tree.profile(node=node))
    return f"{line} vs parent {solved}/{node.total}" + (f"; {change}" if change else "")


def pooled_runs(entry: VariantEntry, nodes: list[core.Node]) -> tuple[list[core.Node], list[core.Node], list[str]]:
    """A variant's scored runs on its first run's tasks, the parent harness's runs on them, and the tasks"""
    ran = [n for n in variant_nodes(entry=entry, nodes=nodes) if n.total > 0]
    if not ran:
        return [], [], []
    tasks = tree.task_set(node=ran[0])
    parent = next((n for n in nodes if n.name == entry.parent_node), None)
    reference = tree.reference_runs(parent=parent, tasks=tasks, nodes=nodes) if parent else []
    return [n for n in ran if tree.task_set(node=n) == tasks], reference, list(tasks)


def unscored(entry: VariantEntry, nodes: list[core.Node]) -> str:
    """Why a variant has no scored run: it wasn't run, or its runs aren't scored yet (with the first error)"""
    made = variant_nodes(entry=entry, nodes=nodes)
    if not made:
        return "not run"
    errors = [(task, run.error) for n in made for task, run in n.runs.items() if run.error]
    if not errors:
        return "not scored yet"
    task, error = errors[0]
    # drop the task id, which may be a hidden task
    error = re.sub(re.escape(task), "<task>", error, flags=re.IGNORECASE)
    first = " ".join(error.split())[:200].replace("|", "/")
    return f"not scored yet; first error: {first}"


def activations(entry: VariantEntry, nodes: list[core.Node]) -> str:
    """How many scored task runs a variant kicked in on, if it logs that, and as many for its parent's harness"""
    same, reference, _ = pooled_runs(entry=entry, nodes=nodes)
    package = resolve_variant(reference=same[0].variant) if same else None
    if package is None or not harness.load(name=same[0].harness).records_activations(package=package):
        return ""
    mine = [run.metrics.acted > 0 for n in same for run in n.runs.values()]
    theirs = [run.metrics.acted > 0 for n in reference for run in n.runs.values()]
    line = f"kicked in on {sum(mine)} of {len(mine)} task runs"
    # a variant carries its parent's changes, which count the same way
    return line + (f", against {sum(theirs)} of {len(theirs)} for its parent's harness" if any(theirs) else "")


@dataclasses.dataclass(frozen=True)
class Standing:
    """A variant's scored runs against every run of its parent's harness"""

    runs: int
    tasks: int
    solved: float  # tasks solved per run
    solves: stats.Evidence
    hidden: stats.Evidence | None  # on the tasks hidden from the teacher
    tokens: stats.TokenEvidence
    problems: list[str]  # why these runs can't be compared, if they can't
    broke: int  # task runs that broke the contract

    @property
    def better_by(self) -> str:
        """Return "solves" (beyond noise, hidden tasks too), "tokens" (fewer tokens, as many solves) or "" if neither"""
        if self.problems or self.broke:
            return ""
        holds = self.hidden is None or self.hidden.gain > 0
        if self.solves.z >= stats.NOISE_Z and holds:
            return "solves"
        if self.tokens.z >= stats.NOISE_Z and self.solves.gain >= 0 and (self.hidden is None or self.hidden.gain >= 0):
            return "tokens"
        return ""


BETTER = {"solves": "solves more", "tokens": "solves as many with fewer tokens"}  # Standing.better_by, in words


def standing(entry: VariantEntry, nodes: list[core.Node]) -> Standing | None:
    """Compare a variant with its parent's harness, or None until both ran these tasks"""
    same, reference, tasks = pooled_runs(entry=entry, nodes=nodes)
    if not same or not reference:
        return None
    lineage = tree.lineage(name=entry.parent_node, nodes=nodes)
    hidden = sorted(set(tasks) & runs.lineage_hidden(root=lineage[0])) if lineage else []
    return Standing(
        runs=len(same),
        tasks=len(tasks),
        solved=sum(n.resolved for n in same) / len(same),
        solves=stats.evidence(variant=same, reference=reference, tasks=tasks),
        hidden=stats.evidence(variant=same, reference=reference, tasks=hidden) if hidden else None,
        tokens=stats.token_evidence(variant=same, reference=reference, tasks=tasks),
        problems=tree.made_differently(compared=[*reference, *same], packages=[same, reference]),
        broke=sum(bool(run.metrics.contract_violations) for n in same for run in n.runs.values()),
    )


def scoreboard(entry: VariantEntry, nodes: list[core.Node]) -> list[str]:
    """How a variant did against its parent's harness, as lines of text"""
    lines = [f"from {entry.parent_node}"]
    same, _, tasks = pooled_runs(entry=entry, nodes=nodes)
    if not same:
        return [*lines, unscored(entry=entry, nodes=nodes)]
    solved = sum(n.resolved for n in same) / len(same)
    lines.append(f"{len(same)} run{'s' if len(same) != 1 else ''}, {solved:.1f}/{len(tasks)} solved per run")
    found = standing(entry=entry, nodes=nodes)
    if found is None:
        return [*lines, "no run of the parent's harness on these tasks to compare with yet"]
    if found.problems:
        lines.append(f"these runs don't compare with the parent's, so none of this counts: {'; '.join(found.problems)}")
    lines.append(f"solved {found.solves.line()}")
    if found.hidden is not None:
        lines.append(f"on the {found.hidden.tasks} tasks hidden from the teacher {found.hidden.line()}")
    lines.append(found.tokens.line())
    if found.broke:
        lines.append(f"broke the contract on {found.broke} task runs, so its result doesn't count")
    acted = activations(entry=entry, nodes=nodes)
    if acted:
        lines.append(acted)
    better = BETTER.get(found.better_by)
    lines.append(f"better than its parent: {better}" if better else "not shown better than its parent")
    return lines


def outcome(entry: VariantEntry, nodes: list[core.Node]) -> str:
    """How a variant did against its parent's harness, in words and without numbers, for the teacher"""
    found = standing(entry=entry, nodes=nodes)
    if found is None:
        same, _, _ = pooled_runs(entry=entry, nodes=nodes)
        return "no run of its parent's harness to compare with yet" if same else unscored(entry=entry, nodes=nodes)
    if found.problems:
        return "its runs were made differently from its parent's, so they don't compare"
    if found.broke:
        return "it changed what's fixed, so its runs don't count"
    if found.better_by:
        return f"better than its parent: {BETTER[found.better_by]}"
    return "worse than its parent: solves fewer" if found.solves.z <= -stats.NOISE_Z else "no clear difference"
