import dataclasses
import json
import logging
import shutil
import time
from pathlib import Path

from rsi import benchmarks, cards, core, harness, paths, runs, teacher, tree, variants

log = logging.getLogger(__name__)

TEACHER_OUT = "out"  # mounted at /out in the teacher's container
ROUND = "round.json"  # the round's teacher and how many sessions it asked for
SESSIONS = 4  # teacher sessions in a round, one after another, and each sees the variants made before it
PROBLEMS = "problems.json"  # why the checks turned a candidate down, in its folder under checks/
# what the teacher sees of each task's run, which leaves out whether it passed
TASK_FIELDS = {"status", "seconds", "patch", "error", "metrics"}


@dataclasses.dataclass(frozen=True)
class Round:
    out: Path  # the round's folder
    registered: list[variants.VariantEntry]


class TeacherUnavailable(Exception):
    """A teacher session failed before doing any work (service down or out of quota)"""


def regular_file(path: Path, folder: Path) -> bool:
    """Check that path is a regular file in folder that no symlink leads to"""
    # runs and the teacher write as root, so a symlink could lead to any file on the host machine
    try:
        real = path.resolve(strict=True)
        return real == folder.resolve() / path.relative_to(folder) and real.is_file()
    except (OSError, ValueError):
        return False


def harness_runs(record: core.Node, nodes: list[core.Node]) -> list[core.Node]:
    """Every run of the record's harness in its lineage that compares with it, oldest first"""
    root = tree.lineage(name=record.name, nodes=nodes)[0]
    same = [
        n
        for n in tree.members(root=root.name, nodes=nodes)
        if n.variant == record.variant and tree.comparable(a=n, b=record)
    ]
    return sorted(same, key=lambda n: n.created)


def stage_runs(dest: Path, ran: list[core.Node], shown: set[str], h: harness.Harness) -> None:
    """Copy each run's shown tasks to dest/<k>/, oldest first, without whether they passed or the node's name"""
    for k, node in enumerate([n for n in ran if shown & n.runs.keys()], start=1):
        folder = dest / str(k)
        tasks = {task: run for task, run in sorted(node.runs.items()) if task in shown}
        record = {
            "model": node.model.model_dump(),
            "tasks": {task: run.model_dump(include=TASK_FIELDS) for task, run in tasks.items()},
        }
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        for task in tasks:
            instance_dir = paths.instances_dir(node_name=node.name) / task
            for name, source in h.run_files(instance_dir=instance_dir).items():
                if regular_file(path=source, folder=instance_dir):
                    (folder / task).mkdir(exist_ok=True)
                    shutil.copyfile(source, folder / task / name)


def stage(record: core.Node, h: harness.Harness, dest: Path, nodes: list[core.Node]) -> Path:
    """Stage the parent's package and runs, the runs of the variants made from it and the harness doc"""
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    root = tree.lineage(name=record.name, nodes=nodes)[0]
    hidden = runs.lineage_hidden(root=root)
    # a session staged before the lineage's probe set is written would see tasks that get hidden later
    if root.parent is not None or not hidden:
        raise ValueError(f"{record.name}'s lineage has no tasks hidden from the teacher yet (its root is {root.name})")
    # only the root's tasks, so a run on other tasks, like held-out ones, never reaches the teacher
    shown = set(tree.task_set(node=root)) - hidden
    dest.mkdir(parents=True)
    shutil.copy2(h.doc, dest / "harness.md")
    shutil.copytree(
        variants.node_package(node=record, baseline=h.package_dir),
        dest / "parent",
        ignore=shutil.ignore_patterns(cards.FILE, ".*"),
    )
    parent_runs = harness_runs(record=record, nodes=nodes)
    stage_runs(dest=dest / "runs" / "parent", ran=parent_runs, shown=shown, h=h)
    # a variant made from any run of the parent's harness is one of its children
    parents = {n.name for n in parent_runs}
    for name, entry in variants.tree_load().items():
        made = [n for n in variants.variant_nodes(entry=entry, nodes=nodes) if n.total > 0]
        if entry.parent_node in parents and made:
            stage_runs(dest=dest / "runs" / name, ran=made, shown=shown, h=h)
    return dest


def refresh_tree(dest: Path, entries: dict[str, variants.VariantEntry], nodes: list[core.Node]) -> None:
    """Update tree.json and tree/ right before a session"""
    shutil.rmtree(dest / "tree", ignore_errors=True)
    (dest / "tree").mkdir()
    for name in entries:
        shutil.copytree(
            variants.resolve_variant(reference=name), dest / "tree" / name, ignore=shutil.ignore_patterns(cards.FILE)
        )
    shown = {
        name: {
            # node names give the benchmark away
            **{key: value for key, value in dataclasses.asdict(entry).items() if key != "parent_node"},
            "outcome": variants.outcome(entry=entry, nodes=nodes),
            "kicked_in": variants.activations(entry=entry, nodes=nodes),
        }
        for name, entry in entries.items()
    }
    (dest / "tree.json").write_text(json.dumps(shown, indent=2) + "\n")


def prompt_text(harness_name: str, model_name: str, parent: str, work: str, timeout_s: int) -> str:
    text = (paths.project_dir() / "prompts" / "variant-generator.md").read_text()
    return (
        text.replace("{harness}", harness_name)
        .replace("{model}", model_name)
        .replace("{parent}", parent)
        .replace("{work}", work)
        .replace("{budget}", f"{timeout_s // 60} minutes")
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
    round: str,
    session: int,
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
        round=round,
        session=session,
    )
    log.info(f"registered {entry.name}: {entry.title} ({entry.diff})")
    return [entry]


def _reject(candidate: Path, problems: list[str], checks: Path) -> None:
    (checks / candidate.name).mkdir(parents=True, exist_ok=True)
    (checks / candidate.name / PROBLEMS).write_text(json.dumps(problems, indent=2) + "\n")
    log.warning(f"rejected {candidate.name}: {'; '.join(problems)}")


def generate(
    record: core.Node,
    h: harness.Harness,
    teacher_spec: core.TeacherSpec,
    endpoint: core.ModelEndpoint,
    sessions: int,
    timeout_s: int,
) -> Round:
    """Run a generation round: teacher sessions one after another, each writing a variant"""
    out = paths.node_dir(name=record.name) / "generations" / time.strftime("%Y%m%d-%H%M%S")
    t = teacher.load(name=teacher_spec.agent)
    described = t.describe(model=teacher_spec.model, effort=teacher_spec.effort)
    log.info(
        f"generation round for {record.name} in {out} ({sessions} sessions; teacher {teacher_spec.name}: {described})"
    )
    # staged first, so a lineage that can't be staged leaves no empty round behind
    workspace = stage(record=record, h=h, dest=out / "stage", nodes=runs.list_nodes())
    (out / ROUND).write_text(json.dumps({"teacher": teacher_spec.model_dump(), "sessions": sessions}, indent=2) + "\n")
    parent_package = variants.node_package(node=record, baseline=h.package_dir)
    bench = benchmarks.get(name=record.dataset.name)
    terms = benchmarks.banned_terms(benchmark=bench) + h.protocol_terms
    prompt = prompt_text(
        harness_name=h.name,
        model_name=record.model.name,
        parent=variants.variant_name(reference=record.variant),
        work=bench.work,
        timeout_s=timeout_s,
    )
    registered: list[variants.VariantEntry] = []
    for session in range(1, sessions + 1):
        refresh_tree(dest=workspace, entries=variants.tree_load(), nodes=runs.list_nodes())
        session_dir = out / f"session-{session}"
        teacher_out = session_dir / TEACHER_OUT
        session_dir.mkdir(parents=True)
        log.info(f"session {session} of {sessions}")
        (session_dir / "prompt.md").write_text(prompt)
        with h.teacher_container(
            workspace=workspace,
            out_dir=teacher_out,
            model=record.model,
            endpoint=endpoint,
            name=f"{record.name}-{out.name}-{session}",
            terms=terms,
        ) as container:
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
        found = collect(
            produced=teacher_out / "variants",
            record=record,
            parent_package=parent_package,
            created_by=f"teacher:{teacher_spec.name}",
            round=out.name,
            session=session,
            h=h,
            terms=terms,
            checks=out / "checks" / session_dir.name,
            endpoint=endpoint,
        )
        registered += found
        if not found and (result.status == "error" or result.tool_calls == 0):
            raise TeacherUnavailable(
                f"session {session} failed before doing any work ({result.error or 'no tool calls'}), so the round "
                f"stops with {sessions - session + 1} of its {sessions} sessions left; the round is in {out}"
            )
    return Round(out=out, registered=registered)
