import importlib.metadata

from jym import core


def discover() -> dict[str, core.EnvironmentSpec]:
    specs = (entry.load() for entry in importlib.metadata.entry_points(group="jym.environments"))
    return {spec.id: spec for spec in specs}


ENVIRONMENTS = discover()


def default() -> str:
    return next(iter(ENVIRONMENTS))


def tasks(env: str) -> dict[str, core.Task]:
    return {task.id: task for task in ENVIRONMENTS[env].tasks}


def all_tasks() -> list[core.Task]:
    return [task for spec in ENVIRONMENTS.values() for task in spec.tasks]


def get_task(env: str, task: str | None = None) -> core.Task:
    if env not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment {env!r}; choose from {list(ENVIRONMENTS)}")
    found, chosen = tasks(env), task or ENVIRONMENTS[env].objective
    if chosen not in found:
        raise ValueError(f"Unknown {env} task {chosen!r}; choose from {list(found)}")
    return found[chosen]


def create(env: str) -> core.Environment:
    return ENVIRONMENTS[env].environment()


def describe(env: str) -> dict:
    spec = ENVIRONMENTS[env]
    objectives = [
        {key: getattr(task, key) for key in ("id", "title", "goal", "milestones", "max_steps", "max_seconds")}
        for task in spec.tasks
    ]
    return {"id": env, "title": spec.title, "about": spec.about, "objectives": objectives, **spec.describe()}


def markdown(spec: dict) -> str:
    lines = [f"# {spec['title']}", "", spec["about"], "", "## Objectives", ""]
    lines += [f"- **{o['title']}** (`{o['id']}`): {o['goal']}" for o in spec["objectives"]]
    lines += ["", "## State", "", "Every decision the model is shown a state with these fields:", ""]
    lines += [f"- `{field}`: {about}" for field, about in spec["state"].items()]
    lines += ["", "## Questions", "", "It answers each question with one of its options.", ""]
    for phase, questions in spec["phases"].items():
        lines += [f"### {phase[0].upper()}{phase[1:]}", ""]
        for qid, question in questions.items():
            lines.append(f"- `{qid}`: {question['instructions']}")
            lines += [
                f"  - `{option}`" + (f": {about}" if about else "") for option, about in question["criteria"].items()
            ]
        lines.append("")
    return "\n".join([*lines, "## Timing", "", spec["timing"], ""])
