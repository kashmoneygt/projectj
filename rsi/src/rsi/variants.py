import dataclasses
import difflib
import hashlib
import itertools
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
    """Make the teacher's name for a variant valid and unique, adding -2, -3, ... if it's taken"""
    # reserve "baseline" (the harness as it ships) and "parent" (holds the parent's runs in the teacher's /work/runs/)
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
    """What a node records as its variant, i.e. a registered variant's name (so records survive a move) or a path"""
    package = package.resolve()
    return package.name if package.parent == paths.variants_dir().resolve() else str(package)


def run_label(node: core.Node) -> str:
    """Short label for a run, i.e. its variant and seed without the model and benchmark from its node name"""
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
    """One-line summary of the files a variant added (+), removed (-) or changed (~)"""
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
    """A variant's entry in the tree, i.e. its card and where it came from (results live in node records)"""

    name: str
    parent: str  # parent variant ('baseline' for the first round)
    parent_node: str  # node it was generated from
    created: str  # ISO-8601 timestamp
    created_by: str  # "teacher:<name>", or "human"
    title: str
    what: str
    why: str
    expect: str
    reuses: list[str]  # names of variants it builds on
    diff: str  # diff_summary vs the parent's package
    round: str = ""  # folder name under runs/<parent_node>/generations/
    session: int = 0  # which session of the round made it (1, 2, ...)
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


def lineage_entries(
    root: core.Node, entries: dict[str, VariantEntry], nodes: list[core.Node]
) -> dict[str, VariantEntry]:
    """Variants made in root's lineage, so the teacher never sees another lineage's variants"""
    members = {n.name for n in tree.members(root=root.name, nodes=nodes)}
    return {name: entry for name, entry in entries.items() if entry.parent_node in members}


def by_kind(variant: list[core.Node], reference: list[core.Node], tasks: list[str]) -> dict[str, stats.Evidence]:
    """Evidence on solves vs the reference for each kind of work, or {} unless the tasks cover 2+ kinds"""
    kinds = variant[0].kinds if variant else {}
    groups: dict[str, list[str]] = {}
    for task in tasks:
        groups.setdefault(kinds.get(task, ""), []).append(task)
    if len(groups) < 2:
        return {}
    return {
        kind: stats.evidence(variant=variant, reference=reference, tasks=group)
        for kind, group in sorted(groups.items())
    }


def regressions(ran: list[core.Node], parent: core.Node | None, nodes: list[core.Node]) -> list[str]:
    """The kinds of work these runs do worse at than the parent's harness, beyond noise"""
    if not ran or parent is None:
        return []
    tasks = tree.task_set(node=ran[0])
    reference = tree.reference_runs(parent=parent, tasks=tasks, nodes=nodes)
    if not reference:
        return []
    return stats.worse_kinds(kinds=by_kind(variant=ran, reference=reference, tasks=list(tasks)))


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
    """A variant's scored runs on the same tasks as its first one, the parent harness's runs on them, and the tasks"""
    ran = [n for n in variant_nodes(entry=entry, nodes=nodes) if n.total > 0]
    if not ran:
        return [], [], []
    tasks = tree.task_set(node=ran[0])
    parent = next((n for n in nodes if n.name == entry.parent_node), None)
    reference = tree.reference_runs(parent=parent, tasks=tasks, nodes=nodes) if parent else []
    return [n for n in ran if tree.task_set(node=n) == tasks], reference, list(tasks)


def unscored(entry: VariantEntry, nodes: list[core.Node]) -> str:
    """Why a variant has no scored run, i.e. it wasn't run or isn't scored yet (with the first error if any)"""
    made = variant_nodes(entry=entry, nodes=nodes)
    if not made:
        return "not run"
    errors = [(task, run.error) for n in made for task, run in n.runs.items() if run.error]
    if not errors:
        return "not scored yet"
    task, error = errors[0]
    # mask the task id, since the teacher sees this error and the task may be a hidden one
    error = re.sub(re.escape(task), "<task>", error, flags=re.IGNORECASE)
    first = " ".join(error.split())[:200].replace("|", "/")
    return f"not scored yet; first error: {first}"


def activations(entry: VariantEntry, nodes: list[core.Node]) -> str:
    """How many scored task runs a variant kicked in on, if it logs that, and the same for its parent's harness"""
    same, reference, _ = pooled_runs(entry=entry, nodes=nodes)
    package = resolve_variant(reference=same[0].variant) if same else None
    if package is None or not harness.load(name=same[0].harness).records_activations(package=package):
        return ""
    mine = [run.metrics.acted > 0 for n in same for run in n.runs.values()]
    theirs = [run.metrics.acted > 0 for n in reference for run in n.runs.values()]
    line = f"kicked in on {sum(mine)} of {len(mine)} task runs"
    # the variant keeps its parent's changes, and they can kick in too, so show the parent's count to compare with
    return line + (f", against {sum(theirs)} of {len(theirs)} for its parent's harness" if any(theirs) else "")


@dataclasses.dataclass(frozen=True)
class Standing:
    """How a variant's scored runs compare with every run of its parent's harness"""

    runs: int
    tasks: int
    solved: float  # tasks solved per run
    solves: stats.Evidence
    hidden: stats.Evidence | None  # on the tasks hidden from the teacher
    kinds: dict[str, stats.Evidence]  # on each kind of work, when there's more than one
    tokens: stats.TokenEvidence
    problems: list[str]  # why these runs can't be compared, if they can't
    broke: int  # task runs that broke the contract

    @property
    def worse_at(self) -> list[str]:
        """The kinds of work it does worse at, beyond noise"""
        return stats.worse_kinds(kinds=self.kinds)

    @property
    def better_by(self) -> str:
        """Return "solves" (beyond noise, ahead on hidden tasks), "tokens" (no fewer solves, fewer tokens) or "" """
        # not better if it's worse at any kind of work, however it does overall
        if self.problems or self.broke or self.worse_at:
            return ""
        holds = self.hidden is None or self.hidden.gain > 0
        if self.solves.z >= stats.NOISE_Z and holds:
            return "solves"
        if self.tokens.z >= stats.NOISE_Z and self.solves.gain >= 0 and (self.hidden is None or self.hidden.gain >= 0):
            return "tokens"
        return ""


BETTER = {"solves": "solves more", "tokens": "solves as many with fewer tokens"}  # Standing.better_by, in words


def standing(entry: VariantEntry, nodes: list[core.Node]) -> Standing | None:
    """Compare a variant with its parent's harness, or None until both have scored runs on the same tasks"""
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
        kinds=by_kind(variant=same, reference=reference, tasks=tasks),
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
    for kind, evidence in found.kinds.items():
        lines.append(f"on {kind or 'other work'} {evidence.line()}")
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
    worse = f"worse at {' and '.join(found.worse_at)}" if found.worse_at else ""
    if found.solves.z <= -stats.NOISE_Z:
        return "worse than its parent: solves fewer" + (f", and {worse}" if worse else "")
    return f"no clear difference overall, but {worse}" if worse else "no clear difference"


FIRST_SEED = 42  # seed of a variant's first run, and each later run takes the next free seed
MAX_RUNS = 3  # the most runs a variant gets in its own round (a parent also runs once more in each round made from it)


@dataclasses.dataclass(frozen=True)
class Contender:
    """A harness and its scored runs on the lineage's probe set, oldest first"""

    name: str  # a variant's name, or "baseline"
    runs: list[core.Node]
    regressed: tuple[str, ...] = ()  # kinds of work it does worse at than its parent's harness, beyond noise

    @property
    def solved(self) -> float:
        return sum(n.resolved for n in self.runs) / len(self.runs)

    @property
    def tokens(self) -> float:
        """Output tokens per task, over all its runs"""
        return sum(tree.profile(node=n).output_tokens for n in self.runs) / len(self.runs)

    @property
    def broke(self) -> bool:
        return any(run.metrics.contract_violations for n in self.runs for run in n.runs.values())

    @property
    def eligible(self) -> bool:
        """Whether it can be picked, i.e. it kept the contract and isn't worse than its parent at any kind of work"""
        return not self.broke and not self.regressed

    def next_seed(self) -> int:
        used = {n.seed for n in self.runs}
        return next(seed for seed in itertools.count(start=FIRST_SEED) if seed not in used)


def contender(name: str, root: core.Node, nodes: list[core.Node], parent: str = "") -> Contender:
    """A harness's runs on the probe set of root's lineage with root's settings (parent is the node it came from)"""
    tasks = tuple(sorted(runs.lineage_probe(root=root)))
    found = sorted(
        (
            n
            for n in nodes
            if variant_name(reference=n.variant) == name
            and n.total > 0
            and tree.task_set(node=n) == tasks
            and tree.comparable(a=n, b=root)
        ),
        key=lambda n: n.created,
    )
    made_from = next((n for n in nodes if n.name == parent), None)
    return Contender(name=name, runs=found, regressed=tuple(regressions(ran=found, parent=made_from, nodes=nodes)))


def newest_round(entries: dict[str, VariantEntry]) -> list[VariantEntry]:
    """The variants of the newest round, in the order they were made"""
    if not entries:
        return []
    newest = max(entries.values(), key=lambda e: e.created)
    return sorted((e for e in entries.values() if e.round == newest.round), key=lambda e: e.created)


def next_run(entries: dict[str, VariantEntry], nodes: list[core.Node]) -> tuple[str, int, str] | None:
    """The newest round's next run as (variant, seed, why), or None once the round is narrowed down"""
    batch = newest_round(entries=entries)
    if not batch:
        return None
    root = tree.lineage(name=batch[0].parent_node, nodes=nodes)[0]
    made_from = next((n for n in nodes if n.name == batch[0].parent_node), None)
    if made_from is not None:
        # a harness becomes the parent on runs that were likely lucky, so it runs once more in each round made from it.
        # Otherwise its mean stays at that lucky high, and better variants rarely beat it
        own = contender(name=variant_name(reference=made_from.variant), root=root, nodes=nodes)
        started = datetime.fromisoformat(batch[0].created).timestamp()
        if not any(n.created >= started for n in own.runs):
            return own.name, own.next_seed(), f"{own.name} is the round's parent, so it runs once more"
    made = [contender(name=e.name, root=root, nodes=nodes, parent=e.parent_node) for e in batch]
    for new in made:
        if not new.runs:
            return new.name, new.next_seed(), f"{new.name} is new, so it runs once"
    # rank by tasks solved in the first run, then by fewer output tokens on a tie
    ranked = sorted(
        (c for c in made if c.eligible),
        key=lambda c: (-c.runs[0].resolved, tree.profile(node=c.runs[0]).output_tokens),
    )
    if not ranked:
        return None
    best = ranked[0]
    if len(best.runs) < MAX_RUNS:
        return best.name, best.next_seed(), f"{best.name} had the round's best first run, so it runs {MAX_RUNS} times"
    if len(ranked) == 1:
        return None
    second = ranked[1]
    if len(second.runs) < 2:
        return second.name, second.next_seed(), f"{second.name} had the round's second best first run, so it runs again"
    if len(second.runs) < MAX_RUNS and second.solved > best.solved:
        why = (
            f"{second.name} solved {second.solved:.1f} per run over {len(second.runs)} runs, more than "
            f"{best.name}'s {best.solved:.1f}, so it runs a third time"
        )
        return second.name, second.next_seed(), why
    return None


def best(entries: dict[str, VariantEntry], nodes: list[core.Node]) -> Contender | None:
    """The next round's parent, i.e. the best harness with 2+ runs in the lineage of the newest round"""
    batch = newest_round(entries=entries)
    if not batch:
        return None
    root = tree.lineage(name=batch[0].parent_node, nodes=nodes)[0]
    # every harness in the lineage counts, earlier parents too, so a round with no winner keeps the current parent
    mine = lineage_entries(root=root, entries=entries, nodes=nodes)
    pool = [contender(name="baseline", root=root, nodes=nodes)] + [
        contender(name=name, root=root, nodes=nodes, parent=entry.parent_node) for name, entry in mine.items()
    ]
    settled = [c for c in pool if len(c.runs) >= 2 and c.eligible]
    return min(settled, key=lambda c: (-c.solved, c.tokens), default=None)
