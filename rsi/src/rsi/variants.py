import dataclasses
import difflib
import hashlib
import json
import re
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path

from rsi import cards, core, paths, runs, stats, tree


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
    reserved = {*taken, "baseline"}
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


# a variant logs that it acted with pi.appendEntry("acted", ...)
ACTED_CALL = re.compile(r"""appendEntry\(\s*["']acted["']""")
CODE_SUFFIXES = (".ts", ".mts", ".cts", ".js", ".mjs", ".cjs")


def records_activations(package: Path) -> bool:
    """Check if the package logs when it acts"""
    return any(
        path.suffix in CODE_SUFFIXES and ACTED_CALL.search(path.read_text(errors="replace"))
        for path in package_files(root=package).values()
    )


def diff_summary(base: Path, variant: Path, max_entries: int = 12) -> str:
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
    if len(entries) > max_entries:
        entries = [*entries[:max_entries], f"+{len(entries) - max_entries} more"]
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


LOG_DIFF_LINES = 150  # max lines of each variant's diff in log.md (tree/<name>/ has every file)
LOG_DIFFS = 2  # the newest commits in log.md that show their diff


FIELD_SHOWN = 400  # max characters per card field in tree.md, for variants from other parents


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
    area: str = ""  # see prompts/areas.md
    round: str = ""  # a folder in the parent node's generations/
    # added by hand in tree.json
    note: str = ""


def tree_load() -> dict[str, VariantEntry]:
    path = paths.tree_path()
    if not path.is_file():
        return {}
    return {name: VariantEntry(**raw) for name, raw in json.loads(path.read_text()).items()}


def tree_save(entries: dict[str, VariantEntry]) -> None:
    path = paths.tree_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {name: dataclasses.asdict(entry) for name, entry in sorted(entries.items())}
    path.write_text(json.dumps(data, indent=2) + "\n")


def tree_register(
    name: str,
    parent: str,
    parent_node: str,
    card: cards.VariantCard,
    diff: str,
    created_by: str = "human",
    area: str = "",
    round: str = "",
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
        area=area,
        round=round,
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


def result_line(node: core.Node, nodes: list[core.Node], detail: bool = True) -> str:
    """A node's result vs its parent's run on the same tasks (and behavior, with detail)"""
    if node.total == 0:
        return "not scored yet"
    reference = tree.reference_run(child=node, nodes=nodes)
    line = f"{node.resolved}/{node.total}"
    if reference is not None:
        comparison = tree.compare(child=node, reference=reference)
        line += f" vs parent {comparison.reference_resolved}/{node.total}"
        if detail:
            line += (
                f" on the same tasks (+{comparison.fixed} fixed, -{comparison.broken} broken; a split this "
                f"lopsided happens by chance with p={comparison.p_value:.2f})"
            )
            change = tree.behavior_change(before=tree.profile(node=reference), after=tree.profile(node=node))
            line += f"; {change}" if change else ""
    return line


def pooled_runs(entry: VariantEntry, nodes: list[core.Node]) -> tuple[list[core.Node], list[core.Node], list[str]]:
    """A variant's scored runs on its first run's tasks, the parent harness's runs on them, and the tasks"""
    ran = [n for n in variant_nodes(entry=entry, nodes=nodes) if n.total > 0]
    if not ran:
        return [], [], []
    tasks = tree.task_set(node=ran[0])
    parent = next((n for n in nodes if n.name == entry.parent_node), None)
    reference = tree.reference_runs(parent=parent, tasks=tasks, nodes=nodes) if parent else []
    return [n for n in ran if tree.task_set(node=n) == tasks], reference, list(tasks)


def pooled_result(entry: VariantEntry, nodes: list[core.Node]) -> tuple[str, str]:
    """A variant's pooled result vs its parent, and its latest run's behavior change"""
    same, reference, tasks = pooled_runs(entry=entry, nodes=nodes)
    if not same:
        made = variant_nodes(entry=entry, nodes=nodes)
        if not made:
            return "not run", ""
        errors = [(task, run.error) for n in made for task, run in n.runs.items() if run.error]
        if not errors:
            return "not scored yet", ""
        task, error = errors[0]
        # drop the task id, which may be a hidden task
        for name in (task, task.replace("__", "_1776_")):
            error = re.sub(re.escape(name), "<task>", error, flags=re.IGNORECASE)
        first = " ".join(error.split())[:200].replace("|", "/")
        return f"not scored yet; first error: {first}", ""
    line = f"{sum(n.resolved for n in same) / len(same):.1f}/{len(tasks)} per run"
    behavior = ""
    if reference:
        line += f"; {stats.evidence(variant=same, reference=reference, tasks=tasks).line()}"
        latest = max(reference, key=lambda n: n.created)
        behavior = tree.behavior_change(before=tree.profile(node=latest), after=tree.profile(node=same[-1]))
    return line, behavior


def activations(entry: VariantEntry, nodes: list[core.Node]) -> str:
    """How many scored task runs a variant acted on, if it logs that"""
    ran = [n for n in variant_nodes(entry=entry, nodes=nodes) if n.total > 0]
    if not ran or not records_activations(package=resolve_variant(reference=ran[0].variant)):
        return ""
    task_runs = [run for n in ran for run in n.runs.values()]
    return f"kicked in on {sum(run.metrics.acted > 0 for run in task_runs)} of {len(task_runs)} task runs"


@dataclasses.dataclass(frozen=True)
class Standing:
    """A variant's scored runs against every run of its parent's harness"""

    runs: int
    tasks: int
    solved: float  # tasks solved per run
    reference_runs: int
    reference_solved: float
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
        reference_runs=len(reference),
        reference_solved=sum(n.resolved for n in reference) / len(reference),
        solves=stats.evidence(variant=same, reference=reference, tasks=tasks),
        hidden=stats.evidence(variant=same, reference=reference, tasks=hidden) if hidden else None,
        tokens=stats.token_evidence(variant=same, reference=reference, tasks=tasks),
        problems=tree.made_differently(compared=[*reference, *same], packages=[same, reference]),
        broke=sum(bool(run.metrics.contract_violations) for n in same for run in n.runs.values()),
    )


def scoreboard(entry: VariantEntry, nodes: list[core.Node]) -> list[str]:
    """How a variant did against its parent's harness, as lines of text"""
    lines = [f"from {entry.parent_node}" + (f", area {entry.area}" if entry.area else "")]
    same, _, tasks = pooled_runs(entry=entry, nodes=nodes)
    if not same:
        return [*lines, pooled_result(entry=entry, nodes=nodes)[0]]
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
    better = {"solves": "solves more", "tokens": "solves as many with fewer tokens"}.get(found.better_by)
    lines.append(f"better than its parent: {better}" if better else "not shown better than its parent")
    return lines


def compose_candidates(record: core.Node, entries: dict[str, VariantEntry], nodes: list[core.Node]) -> list[str]:
    """Variants in record's lineage that did a bit better but nothing builds on yet"""
    lineage = tree.lineage(name=record.name, nodes=nodes)
    made_from = {n.name for n in lineage}
    built_on = {variant_name(reference=n.variant) for n in lineage if n.variant}
    found: list[str] = []
    for name, entry in entries.items():
        if entry.parent_node not in made_from or name in built_on:
            continue
        same, reference, tasks = pooled_runs(entry=entry, nodes=nodes)
        if (
            same
            and reference
            and (
                stats.evidence(variant=same, reference=reference, tasks=tasks).gain > 0
                or stats.token_evidence(variant=same, reference=reference, tasks=tasks).saving > 0
            )
        ):
            found.append(name)
    return found


def ancestry(name: str, entries: dict[str, VariantEntry]) -> list[str]:
    """The variants from the baseline up to name, oldest first"""
    chain: list[str] = []
    while name in entries and name not in chain:
        chain.append(name)
        name = entries[name].parent
    return chain[::-1]


def lineage_log(parent: str, entries: dict[str, VariantEntry], nodes: list[core.Node], baseline: Path) -> str:
    """How the parent was built, like `git log -p` from the baseline, with each variant's card and diff"""
    lines = ["# How the parent was built", ""]
    if parent == "baseline":
        return "\n".join(
            [*lines, "The parent is the baseline, the harness as it ships, so nothing has been added yet.", ""]
        )
    chain = ancestry(name=parent, entries=entries)
    if not chain:
        return "\n".join(
            [*lines, f"The parent, `{parent}`, isn't in the tree, so only `parent/` shows what it holds.", ""]
        )
    shown = f"Only the newest {LOG_DIFFS} show their diff."
    where = "`parent/` has the code they all add up to, and `tree/<name>/` has the package of each."
    lines += [f"The parent is `{parent}`, the baseline with these variants on top, oldest first. {shown} {where}", ""]
    for index, name in enumerate(chain):
        entry = entries[name]
        package = resolve_variant(reference=name)
        card = package / cards.FILE
        message = card.read_text(errors="replace").strip() if card.is_file() else entry.title
        scored = pooled_result(entry=entry, nodes=nodes)[0]
        lines += [
            f"## {name}",
            "",
            f"Built on `{entry.parent}`. It changed {entry.diff} and scored {scored}.",
            "",
            *(f"    {line}" if line else "" for line in message.splitlines()),
            "",
        ]
        if index < len(chain) - LOG_DIFFS:
            continue
        before = baseline if entry.parent == "baseline" else resolve_variant(reference=entry.parent)
        diff = unified_diff(base=before, variant=package).splitlines()
        if len(diff) > LOG_DIFF_LINES:
            diff = [*diff[:LOG_DIFF_LINES], f"... {len(diff) - LOG_DIFF_LINES} more lines in tree/{name}/"]
        # four backticks, so a fence inside the diff doesn't end this one
        lines += ["````diff", *diff, "````", ""]
    return "\n".join(lines)


def tree_markdown(
    entries: dict[str, VariantEntry],
    nodes: list[core.Node],
    lineage: list[str],
    results: dict[str, str],
    parent_node: str,
    turned_down: list[str],
) -> str:
    """The teacher's view of the tree (tree.md)"""
    lines = ["# Variants tried so far", ""]
    if lineage:
        lines += [f"The parent's lineage: {' → '.join(lineage)}. `log.md` has each of its commits.", ""]
    if turned_down:
        lines += ["Turned down from this parent so far, so don't build these again as they were:", "", *turned_down, ""]
    if not entries:
        return "\n".join([*lines, "None yet, since this is the first round.", ""])
    by_age = sorted(entries.items(), key=lambda kv: kv[1].created)
    siblings = [name for name, entry in by_age if entry.parent_node == parent_node]
    if siblings:
        made = ", ".join(f"`{name}`" for name in siblings)
        others = "For the others, `tree/<name>/VARIANT.md` has the whole card."
        lines += [f"Made from this parent so far: {made}. Their sections come first, with whole cards. {others}", ""]
    lines += [
        "A result counts every task in the fixed set, the ones you never see included.",
        "",
        "| variant | parent | area | title | result |",
        "|---|---|---|---|---|",
    ]
    pooled = {name: pooled_result(entry=entry, nodes=nodes) for name, entry in by_age}
    for name, entry in by_age:
        lines.append(f"| {name} | {entry.parent} | {entry.area} | {entry.title} | {pooled[name][0]} |")
    for name, entry in sorted(by_age, key=lambda kv: kv[0] not in siblings):
        lines += ["", f"## {name}", "", entry.title, ""]
        fields = [("what", entry.what), ("why", entry.why), ("expect", entry.expect)]
        if name not in siblings:
            fields = [field for field in fields if field[0] != "why"]
        for key, value in fields:
            if name not in siblings and len(value) > FIELD_SHOWN:
                value = f"{value[:FIELD_SHOWN].rstrip()} … (all of it is in `tree/{name}/VARIANT.md`)"
            if value:
                lines.append(f"- {key}: {value}")
        if entry.reuses:
            lines.append(f"- reuses: {', '.join(entry.reuses)}")
        lines.append(f"- changed: {entry.diff}")
        acted = activations(entry=entry, nodes=nodes)
        if acted:
            lines.append(f"- {acted}")
        if pooled[name][1]:
            lines.append(f"- behavior, latest run vs the parent's: {pooled[name][1]}")
        if entry.note:
            lines.append(f"- note from the people running the variants: {entry.note}")
        if results.get(name):
            lines.append(f"- runs: `results/{name}/` ({results[name]})")
    return "\n".join(lines) + "\n"
