import dataclasses
import json
import logging
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path

from rsi import benchmarks, cards, core, harness, paths, runs, teacher, tree, variants

log = logging.getLogger(__name__)

DIFF_MAX_CHARS = 3_000  # so a task file fits one read_file call
TEACHER_OUT = "out"  # mounted at /out in the teacher's container
NONE = "NONE.md"  # written when the teacher has nothing worth running
ROUND = "round.json"  # the round's teacher and areas, so the viewer knows whose sessions they are
AREAS = ("tools", "middleware", "context")  # besides open and compose


@dataclasses.dataclass(frozen=True)
class Round:
    out: Path  # the round's folder
    registered: list[variants.VariantEntry]
    declined: dict[str, str]  # area -> why its session wrote no variant


class TeacherUnavailable(Exception):
    """A teacher session failed before doing any work (service down or out of quota)"""


def teacher_text(path: Path, out: Path) -> str | None:
    """A file the teacher wrote under out, or None if it isn't there, isn't a regular file or a symlink leads to it"""
    # the teacher writes as root, so a symlink could point at any file on the host machine
    try:
        real = path.resolve(strict=True)
    except OSError:
        return None
    if real != out.resolve() / path.relative_to(out) or not real.is_file():
        return None
    return real.read_text(errors="replace")


def _tools_line(metrics: core.TaskMetrics) -> str:
    calls = [
        f"{name} {stat.count}" + (f" ({stat.errors} errors)" if stat.errors else "")
        for name, stat in sorted(metrics.tool_stats.items(), key=lambda kv: -kv[1].count)
    ]
    return ", ".join(calls) or "none"


def _ended(run: core.TaskRun) -> str:
    """How a run ended ("never started", its status, or its last stop reason)"""
    if run.status == "ok":
        return run.metrics.stop_reason or "-"
    return "never started" if run.metrics.requests == 0 else run.status


def _share(part: float, whole: float) -> str:
    return f"{part / whole:.0%}" if whole else "-"


def _behavior_column(runs: list[core.TaskRun]) -> list[str]:
    """One column of overview.md's behavior table (medians per task, shares of totals)"""
    ms = [run.metrics for run in runs]
    if not ms:
        return ["0"] + ["-"] * 11
    calls = sum(stat.count for m in ms for stat in m.tool_stats.values())
    errors = sum(stat.errors for m in ms for stat in m.tool_stats.values())
    wall = sum(run.seconds for run in runs) * 1000
    model, tools = sum(m.model_ms for m in ms), sum(m.tool_ms for m in ms)
    output = sum(m.output_tokens for m in ms)
    return [
        str(len(runs)),
        f"{statistics.median(run.seconds for run in runs) / 60:.1f}",
        f"{statistics.median(m.requests for m in ms):.0f}",
        f"{statistics.median(m.output_tokens for m in ms):.0f}",
        _share(part=sum(m.reasoning_tokens for m in ms), whole=output),
        f"{statistics.median(m.peak_output_tokens for m in ms):.0f}",
        str(sum(1 for m in ms if m.capped_requests)),
        f"{calls / max(sum(m.requests for m in ms), 1):.2f}",
        _share(part=errors, whole=calls),
        " / ".join(_share(part=part, whole=wall) for part in (model, tools, wall - model - tools)),
        str(sum(1 for m in ms if m.patch_lines == 0)),
        f"{statistics.median(m.context_pct for m in ms):.0f}%",
    ]


_BEHAVIOR_ROWS = (
    "tasks",
    "minutes per task (median)",
    "requests per task (median)",
    "output tokens per task (median)",
    "share of output tokens spent thinking",
    "largest single response, tokens (median)",
    "runs with a response cut at the output cap",
    "tool calls per request",
    "tool calls that returned an error",
    "wall time: model / tools / other",
    "runs that changed no file",
    "peak context (median)",
)


def overview_markdown(record: core.Node, passes: dict[str, tuple[int, int]], diverged: list[str]) -> str:
    runs = sorted(record.runs.values(), key=lambda run: run.instance_id)
    groups = ([run for run in runs if run.resolved], [run for run in runs if not run.resolved], runs)
    columns = [_behavior_column(runs=group) for group in groups]
    lines = [
        "# The parent run",
        "",
        f"- harness: {variants.variant_name(reference=record.variant)}",
        (
            f"- model: {record.model.name}, thinking {'on' if record.model.thinking else 'off'}, context window "
            f"{record.model.ctx} tokens, at most {record.model.output_cap} tokens in one response (thinking included)"
        ),
        (
            f"- each task gets {record.budget.timeout_s} s, with no turn limit, and a run still going then is "
            "stopped and scored as it stands"
        ),
        f"- result: {len(groups[0])} of {len(runs)} tasks passed",
    ]
    if diverged:
        lines += [
            "",
            "## Where the parent's runs disagree",
            "",
            (
                "Tasks the parent's harness passed in some runs and failed in others. Same task, same harness, so "
                "what differs is how the runs went. `tasks/diverged/<task>.md` puts one passing and one failing run "
                "side by side."
            ),
            "",
            "| task | passed |",
            "|---|---|",
        ]
        lines += [f"| {task} | {passes[task][0]} of {passes[task][1]} |" for task in diverged]
    lines += [
        "",
        "## How the agent works",
        "",
        "Across runs that passed, runs that failed, and all runs.",
        "",
        "| | passed | failed | all |",
        "|---|---|---|---|",
    ]
    lines += [f"| {row} | {' | '.join(column[i] for column in columns)} |" for i, row in enumerate(_BEHAVIOR_ROWS)]
    tools: dict[str, core.ToolStat] = {}
    for run in runs:
        for name, stat in run.metrics.tool_stats.items():
            total = tools.setdefault(name, core.ToolStat())
            total.count += stat.count
            total.errors += stat.errors
            total.ms += stat.ms
    lines += ["", "| tool | calls | errors | total seconds |", "|---|---|---|---|"]
    lines += [
        f"| {name} | {total.count} | {total.errors} | {total.ms / 1000:.0f} |"
        for name, total in sorted(tools.items(), key=lambda kv: -kv[1].count)
    ]
    endings: dict[str, list[bool]] = {}
    for run in runs:
        endings.setdefault(_ended(run=run), []).append(run.resolved)
    lines += [
        "",
        "## How runs ended",
        "",
        (
            "The last request's stop reason, or the run's status if it didn't end on its own. stop means the "
            "agent finished, length means a response hit the output cap, toolUse means it ended in the middle of "
            "a tool call, and never started means it failed before its first model request (its transcript has "
            "the error)."
        ),
        "",
        "| ending | tasks | passed |",
        "|---|---|---|",
    ]
    for ending, outcomes in sorted(endings.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {ending} | {len(outcomes)} | {sum(outcomes)} |")
    lines += [
        "",
        "## Tasks",
        "",
        (
            "all runs: how many of this harness's runs passed the task, over every run of it so far. The same "
            "harness passes and fails some tasks by chance."
        ),
        "",
        (
            "| task | passed | all runs | ended | minutes | requests | output tokens | thinking | largest response "
            "| responses cut | peak context | tool calls | patch lines |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        m = run.metrics
        passed, tries = passes.get(run.instance_id, (int(run.resolved), 1))
        lines.append(
            f"| {run.instance_id} | {'yes' if run.resolved else 'no'} | {passed} of {tries} | "
            f"{_ended(run=run)} | {run.seconds / 60:.1f} | {m.requests} | {m.output_tokens} | "
            f"{_share(part=m.reasoning_tokens, whole=m.output_tokens)} | {m.peak_output_tokens} | "
            f"{m.capped_requests} | {m.context_pct}% | {_tools_line(metrics=m)} | {m.patch_lines} |"
        )
    return "\n".join(lines) + "\n"


def harness_runs(record: core.Node, nodes: list[core.Node]) -> list[core.Node]:
    """Every scored run of the parent's harness that compares with the parent"""
    return [n for n in nodes if n.variant == record.variant and n.total > 0 and tree.comparable(a=n, b=record)]


def pass_counts(record: core.Node, nodes: list[core.Node], hidden: set[str]) -> dict[str, tuple[int, int]]:
    """Passes and runs per task over the harness's runs, hidden tasks left out"""
    counts: dict[str, tuple[int, int]] = {}
    for node in harness_runs(record=record, nodes=nodes):
        for instance_id, run in node.runs.items():
            if run.status != "error" and instance_id not in hidden:
                passed, tries = counts.get(instance_id, (0, 0))
                counts[instance_id] = (passed + run.resolved, tries + 1)
    return counts


def task_markdown(run: core.TaskRun, transcript: str) -> str:
    verdict = "passed" if run.resolved else "failed"
    metrics = run.metrics
    thinking = _share(part=metrics.reasoning_tokens, whole=metrics.output_tokens)
    lines = [
        f"# {run.instance_id}: {verdict}",
        "",
        (
            f"ended {_ended(run=run)} · {run.seconds / 60:.1f} min · {metrics.requests} requests · "
            f"{metrics.output_tokens} output tokens ({thinking} thinking) · peak context {metrics.context_pct}% · "
            f"tools: {_tools_line(metrics=metrics)}"
        ),
    ]
    if run.error:
        lines.append(f"error: {run.error[:300]}")
    diff = run.patch.strip()
    if len(diff) > DIFF_MAX_CHARS:
        diff = diff[:DIFF_MAX_CHARS] + "\n… (diff truncated)"
    lines += ["", "## Turns", "", transcript.rstrip(), "", "## Final diff", ""]
    lines.append(f"```diff\n{diff}\n```" if diff else "(none: the run changed nothing)")
    return "\n".join(lines) + "\n"


def stage(record: core.Node, h: harness.Harness, dest: Path, nodes: list[core.Node]) -> Path:
    """Stage the round's fixed files: the parent's runs, package and history, its system prompt and the harness doc"""
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    hidden = runs.lineage_hidden(root=tree.lineage(name=record.name, nodes=nodes)[0])
    shown = record.model_copy(update={"runs": {iid: run for iid, run in record.runs.items() if iid not in hidden}})
    (dest / "tasks" / "full").mkdir(parents=True)
    shutil.copy2(h.doc, dest / "harness.md")
    shutil.copytree(
        variants.node_package(node=record, baseline=h.package_dir),
        dest / "parent",
        ignore=shutil.ignore_patterns(cards.FILE, ".*"),
    )
    (dest / "log.md").write_text(
        variants.lineage_log(
            parent=variants.variant_name(reference=record.variant),
            entries=variants.tree_load(),
            nodes=nodes,
            baseline=h.package_dir,
        )
    )
    told, told_in = "", ""
    for instance_id, run in sorted(shown.runs.items()):
        instance_dir = paths.instances_dir(node_name=record.name) / instance_id
        for folder, full in ((dest / "tasks", False), (dest / "tasks" / "full", True)):
            transcript = h.transcript(instance_dir=instance_dir, full=full)
            (folder / f"{instance_id}.md").write_text(task_markdown(run=run, transcript=transcript))
        if not told:
            told, told_in = h.system_context(instance_dir=instance_dir), instance_id
    if told:
        (dest / "system-prompt.md").write_text(
            f"# What the model gets before the task\n\nAs the first request of the parent's run on {told_in} sent "
            f"it.\n\n{told}"
        )
    passes = pass_counts(record=record, nodes=nodes, hidden=hidden)
    diverged = stage_divergences(dest=dest, h=h, record=shown, nodes=nodes, passes=passes)
    (dest / "overview.md").write_text(overview_markdown(record=shown, passes=passes, diverged=diverged))
    return dest


def stage_divergences(
    dest: Path, h: harness.Harness, record: core.Node, nodes: list[core.Node], passes: dict[str, tuple[int, int]]
) -> list[str]:
    """Put a passing and a failing run side by side for each task the parent's harness split on"""
    sides: dict[str, dict[bool, core.Node]] = {}
    for node in sorted(harness_runs(record=record, nodes=nodes), key=lambda n: (n.name == record.name, n.created)):
        for task, run in node.runs.items():
            if task in record.runs and run.status != "error":
                sides.setdefault(task, {})[run.resolved] = node
    diverged = sorted(task for task, side in sides.items() if len(side) == 2)
    folder = dest / "tasks" / "diverged"
    if diverged:
        folder.mkdir()
    for task in diverged:
        passed, tries = passes[task]
        sections = [f"# {task}: passed in {passed} of {tries} runs of the parent's harness"]
        for label, node in (("passed", sides[task][True]), ("failed", sides[task][False])):
            sections.append(
                _run_section(h=h, node=node, task=task, label=f"{label} (run {variants.run_label(node=node)})")
            )
        (folder / f"{task}.md").write_text("\n\n".join(sections) + "\n")
    return diverged


def _run_section(h: harness.Harness, node: core.Node, task: str, label: str) -> str:
    """One run of a task under its own heading"""
    run = node.runs[task]
    instance_dir = paths.instances_dir(node_name=node.name) / task
    body = task_markdown(run=run, transcript=h.transcript(instance_dir=instance_dir, full=False))
    return f"## {label}\n\n" + body.split("\n", 1)[1].strip()


def stage_results(
    dest: Path,
    h: harness.Harness,
    entries: dict[str, variants.VariantEntry],
    nodes: list[core.Node],
    hidden: set[str],
) -> dict[str, str]:
    """Each scored variant's fixed and broken tasks, next to the parent's runs"""
    staged: dict[str, str] = {}
    for name, entry in entries.items():
        runs_of_variant = [n for n in variants.variant_nodes(entry=entry, nodes=nodes) if n.total > 0]
        child = runs_of_variant[-1] if runs_of_variant else None
        reference = tree.reference_run(child=child, nodes=nodes) if child else None
        if child is None or reference is None:
            continue
        flips = {
            task: "fixed" if child.runs[task].resolved else "broken"
            for task in sorted(set(child.runs) & set(reference.runs) - hidden)
            if child.runs[task].resolved != reference.runs[task].resolved
        }
        folder = dest / "results" / name
        folder.mkdir(parents=True)
        for task, flip in flips.items():
            runs_of = f"run {variants.run_label(node=child)}, parent run {variants.run_label(node=reference)}"
            sections = [f"# {task}: {flip} by {name} ({runs_of})"]
            for label, node in ((name, child), ("parent", reference)):
                sections.append(_run_section(h=h, node=node, task=task, label=label))
            (folder / f"{flip}-{task}.md").write_text("\n\n".join(sections) + "\n")
        fixed = sum(flip == "fixed" for flip in flips.values())
        staged[name] = f"{fixed} fixed and {len(flips) - fixed} broken tasks, each run next to the parent's"
    return staged


def turned_down(record: core.Node) -> list[str]:
    """A line for each candidate from this parent the checks rejected, and each area a session declined"""
    found: list[str] = []
    for out in sorted((paths.node_dir(name=record.name) / "generations").glob("*")):
        for problems in sorted(out.glob("checks/*/*/PROBLEMS.md")):
            session, name = problems.parent.parent.name, problems.parent.name
            teacher_out = out / session / TEACHER_OUT
            card = teacher_text(path=teacher_out / "variants" / name / cards.FILE, out=teacher_out)
            title = cards.parse(text=card or "").title or "with no title"
            why = problems.read_text(errors="replace").splitlines()[0].removeprefix("- ")
            found.append(f"- `{name}`, {title}. Rejected, since {why}")
        for none in sorted(out.glob(f"session-*/{TEACHER_OUT}/variants/{NONE}")):
            said = teacher_text(path=none, out=none.parents[1])
            if said is None:
                continue
            area = none.parents[2].name.split("-", 2)[-1]
            said = " ".join(said.split())[:300]
            found.append(f"- The {area} session made nothing and " + (f'said "{said}"' if said else "gave no reason"))
    return found


def refresh_tree(
    dest: Path, h: harness.Harness, record: core.Node, entries: dict[str, variants.VariantEntry], nodes: list[core.Node]
) -> None:
    """Update tree.md, tree/ and results/ right before a session"""
    for folder in ("tree", "results"):
        shutil.rmtree(dest / folder, ignore_errors=True)
    (dest / "tree").mkdir()
    for name in entries:
        shutil.copytree(variants.resolve_variant(reference=name), dest / "tree" / name)
    lineage = tree.lineage(name=record.name, nodes=nodes)
    results = stage_results(dest=dest, h=h, entries=entries, nodes=nodes, hidden=runs.lineage_hidden(root=lineage[0]))
    chain = [variants.variant_name(reference=n.variant) for n in lineage]
    (dest / "tree.md").write_text(
        variants.tree_markdown(
            entries=entries,
            nodes=nodes,
            lineage=chain,
            results=results,
            parent_node=record.name,
            turned_down=turned_down(record=record),
        )
    )


def area_texts() -> dict[str, str]:
    """Each area's text from prompts/areas.md"""
    text = (paths.project_dir() / "prompts" / "areas.md").read_text()
    sections = (section.partition("\n") for section in text.split("\n## ")[1:])
    return {area.strip(): " ".join(body.split()) for area, _, body in sections}


def default_areas(record: core.Node, nodes: list[core.Node]) -> list[str]:
    """Open first, then the least-covered areas, then compose once two variants are worth combining"""
    lineage = tree.lineage(name=record.name, nodes=nodes)
    root = lineage[0].name if lineage else record.name
    members = [n.name for n in tree.members(root=root, nodes=nodes)]
    # a session's folder is session-<n>-<area>
    given = Counter(
        session.name.split("-", 2)[2]
        for name in members
        for session in (paths.node_dir(name=name) / "generations").glob("*/session-*-*")
    )
    entries = variants.tree_load()
    compose = len(variants.compose_candidates(record=record, entries=entries, nodes=nodes)) >= 2
    least = sorted(AREAS, key=lambda area: given[area])  # ties keep the order in AREAS
    return ["open", *least[: 2 if compose else 3], *(["compose"] if compose else [])]


def prompt_text(harness_name: str, model_name: str, area: str, timeout_s: int, start: float) -> str:
    text = (paths.project_dir() / "prompts" / "variant-generator.md").read_text()
    return (
        text.replace("{harness}", harness_name)
        .replace("{model}", model_name)
        .replace("{area}", area_texts()[area])
        .replace("{budget}", f"{timeout_s // 60} minutes")
        .replace("{deadline}", time.strftime("%H:%M UTC", time.gmtime(start + timeout_s)))  # the container's clock
    )


def package_problems(package: Path, h: harness.Harness, terms: tuple[str, ...], log_dir: Path) -> list[str]:
    """A variant's problems: a missing card, or the harness's own checks"""
    return cards.check(package=package) + h.validate_variant(variant=package, out_dir=log_dir, terms=terms)


def candidate_problems(
    candidate: Path,
    parent_package: Path,
    entries: dict[str, variants.VariantEntry],
    h: harness.Harness,
    terms: tuple[str, ...],
    log_dir: Path,
) -> list[str]:
    # the teacher writes as root, so a symlink could reach outside the package
    links = sorted(p.relative_to(candidate).as_posix() for p in candidate.rglob("*") if p.is_symlink())
    if links:
        return [f"{rel} is a symlink; a package holds regular files only" for rel in links]
    problems: list[str] = []
    if variants.same_package(a=candidate, b=parent_package):
        problems.append("the package is identical to the parent's: it changes nothing")
    problems += [
        f"the package is identical to variant {name!r}"
        for name in entries
        if variants.same_package(a=candidate, b=variants.resolve_variant(reference=name))
    ]
    return problems + package_problems(package=candidate, h=h, terms=terms, log_dir=log_dir)


def collect(
    produced: Path,
    record: core.Node,
    parent_package: Path,
    created_by: str,
    area: str,
    round: str,
    h: harness.Harness,
    terms: tuple[str, ...],
    checks: Path,
    endpoint: core.ModelEndpoint,
) -> list[variants.VariantEntry]:
    """Register the session's variant if it passes the checks (the others' problems go in checks/)"""
    # a symlink here would make every folder it leads to a candidate
    found = sorted(produced.iterdir()) if produced.is_dir() and not produced.is_symlink() else []
    candidates = [d for d in found if d.is_dir() and not d.is_symlink()]
    if not candidates:
        log.warning(f"the session wrote no variant under {produced}")
    entries = variants.tree_load()
    passed: list[Path] = []
    for candidate in candidates:
        problems = candidate_problems(
            candidate=candidate,
            parent_package=parent_package,
            entries=entries,
            h=h,
            terms=terms,
            log_dir=checks / candidate.name,
        )
        if not problems:
            log.info(f"asking {candidate.name} the questions that need no change")
            problems = h.question_problems(
                variant=candidate, out_dir=checks / candidate.name / "question", model=record.model, endpoint=endpoint
            )
        if problems:
            _reject(candidate=candidate, problems=problems, checks=checks)
        else:
            passed.append(candidate)
    if len(passed) > 1:
        # each one takes hours to run, and a session is asked for one
        names = ", ".join(candidate.name for candidate in passed)
        for candidate in passed:
            problem = f"the session handed in {len(passed)} variants that pass the checks ({names}), not one"
            _reject(candidate=candidate, problems=[problem], checks=checks)
        return []
    if not passed:
        return []
    candidate = passed[0]
    card = cards.read(package=candidate)
    assert card is not None  # cards.check found a card
    unknown = [name for name in card.reuses if name not in entries]
    if unknown:
        log.warning(f"{candidate.name}: it reuses {unknown}, which aren't in the tree; dropped")
        card = dataclasses.replace(card, reuses=tuple(name for name in card.reuses if name in entries))
    registry = paths.variants_dir()
    taken = set(entries) | ({p.name for p in registry.iterdir()} if registry.is_dir() else set())
    name = variants.free_name(proposed=candidate.name, taken=taken)
    if name != candidate.name:
        log.info(f"registering {candidate.name} as {name}")
    diff = variants.diff_summary(base=parent_package, variant=candidate)
    # package first, so a crash never leaves a tree entry without one
    registry.mkdir(parents=True, exist_ok=True)
    shutil.move(str(candidate), str(registry / name))
    entry = variants.tree_register(
        name=name,
        parent=variants.variant_name(reference=record.variant),
        parent_node=record.name,
        card=card,
        diff=diff,
        created_by=created_by,
        area=area,
        round=round,
    )
    log.info(f"registered {entry.name}: {entry.title} ({entry.diff})")
    return [entry]


def _reject(candidate: Path, problems: list[str], checks: Path) -> None:
    (checks / candidate.name).mkdir(parents=True, exist_ok=True)
    (checks / candidate.name / "PROBLEMS.md").write_text("".join(f"- {p}\n" for p in problems))
    log.warning(f"rejected {candidate.name}: {'; '.join(problems)}")


def generate(
    record: core.Node,
    h: harness.Harness,
    teacher_spec: core.TeacherSpec,
    endpoint: core.ModelEndpoint,
    areas: list[str],
    timeout_s: int,
) -> Round:
    """Run a generation round: one teacher session per area, each writing a variant or NONE.md"""
    out = paths.node_dir(name=record.name) / "generations" / time.strftime("%Y%m%d-%H%M%S")
    t = teacher.load(name=teacher_spec.agent)
    described = t.describe(model=teacher_spec.model, effort=teacher_spec.effort)
    log.info(
        f"generation round for {record.name} in {out} (areas {', '.join(areas)}; "
        f"teacher {teacher_spec.name}: {described})"
    )
    out.mkdir(parents=True)
    (out / ROUND).write_text(json.dumps({"teacher": teacher_spec.model_dump(), "areas": areas}, indent=2) + "\n")
    workspace = stage(record=record, h=h, dest=out / "stage", nodes=runs.list_nodes())
    parent_package = variants.node_package(node=record, baseline=h.package_dir)
    terms = benchmarks.banned_terms(benchmark=benchmarks.get(name=record.dataset.name)) + h.protocol_terms
    registered: list[variants.VariantEntry] = []
    declined: dict[str, str] = {}
    for session, area in enumerate(areas, start=1):
        refresh_tree(dest=workspace, h=h, record=record, entries=variants.tree_load(), nodes=runs.list_nodes())
        session_dir = out / f"session-{session}-{area}"
        teacher_out = session_dir / TEACHER_OUT
        session_dir.mkdir(parents=True)
        log.info(f"session {session}: area {area}")
        with h.teacher_container(
            workspace=workspace,
            out_dir=teacher_out,
            model=record.model,
            endpoint=endpoint,
            name=f"{record.name}-{out.name}-{session}",
            terms=terms,
        ) as container:
            # once the container is up, since the deadline counts from here
            prompt = prompt_text(
                harness_name=h.name, model_name=record.model.name, area=area, timeout_s=timeout_s, start=time.time()
            )
            (session_dir / "prompt.md").write_text(prompt)
            result = t.run(
                prompt=prompt,
                model=teacher_spec.model,
                effort=teacher_spec.effort,
                container=container,
                out_dir=session_dir,
                timeout_s=timeout_s,
            )
        (session_dir / "transcript.md").write_text(t.transcript(session_dir=session_dir))
        error = f" ({result.error})" if result.error else ""
        log.info(
            f"session {session} ended {result.status} after {result.seconds:.0f} s: {result.model_calls} model "
            f"calls, {result.tool_calls} tool calls, {result.output_tokens} output tokens, "
            f"{result.cost or 'cost unknown'}{error}"
        )
        produced = teacher_out / "variants"
        said = teacher_text(path=produced / NONE, out=teacher_out)
        if said is not None and not any(d.is_dir() for d in produced.iterdir()):
            declined[area] = " ".join(said.split())[:600] or "it gave no reason"
            log.info(f"session {session} found nothing worth running in area {area}: {declined[area]}")
            continue
        found = collect(
            produced=produced,
            record=record,
            parent_package=parent_package,
            created_by=f"teacher:{teacher_spec.name}",
            area=area,
            round=out.name,
            h=h,
            terms=terms,
            checks=out / "checks" / session_dir.name,
            endpoint=endpoint,
        )
        registered += found
        if not found and (result.status == "error" or result.tool_calls == 0):
            left = ", ".join(areas[session - 1 :])
            raise TeacherUnavailable(
                f"session {session} failed before doing any work ({result.error or 'no tool calls'}), so the round "
                f"stops; the areas left are {left}, and the round is in {out}"
            )
    return Round(out=out, registered=registered, declined=declined)
