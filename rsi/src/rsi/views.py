import dataclasses
import json
import re

# max characters the viewer shows of thinking, text, tool output and tool args
THINK_MAX = 30_000
TEXT_MAX = 20_000
OUT_MAX = 8_000
ARGS_MAX = 8_000


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… [{len(text) - limit} more characters]"


@dataclasses.dataclass
class Call:
    """A tool call, named bash, read, write or edit for those kinds of tool, else by the tool's own name"""

    id: str
    name: str
    label: str
    args: dict  # clipped args as one of command, content, edits or json
    out: str | None = None
    ok: bool | None = None
    ms: float | None = None

    def ended(self, out: str, ok: bool, ms: float | None) -> None:
        self.out, self.ok, self.ms = clip(text=out, limit=OUT_MAX), ok, ms


@dataclasses.dataclass
class Step:
    """A model response and the tool calls it made"""

    t: float  # ms since the run started
    ms: float | None  # how long the response took
    intent: str  # first sentence of the thinking (or the text), used as the step title
    thinking: str
    thinking_chars: int
    text: str
    calls: list[Call]
    stop: str | None
    out_tokens: int | None
    error: str | None
    kind: str = "step"


@dataclasses.dataclass
class Note:
    """Something that happened between responses, like a retry or a compaction"""

    text: str
    kind: str = "note"


@dataclasses.dataclass
class Timing:
    """Timings in ms for one response, and the size in tokens of the request it answered"""

    ttft: float | None
    total: float | None
    think: float | None
    context: int | None
    stop: str


@dataclasses.dataclass
class Trajectory:
    user: str = ""  # the first user message
    items: list[Step | Note] = dataclasses.field(default_factory=list)
    inflight_at: float | None = None  # when the response still in progress started, in epoch ms
    timing: list[Timing] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Checks:
    """A group of benchmark checks, e.g. the tests a fix should make pass"""

    label: str
    failing: str  # word for a failing check in this group, e.g. "still failing"
    total: int
    passed: int
    failed: list[str]


@dataclasses.dataclass
class Verdict:
    """How the benchmark judged a task"""

    resolved: bool
    applied: bool  # the agent's changes could be applied for grading
    infra: bool  # the grader itself failed
    checks: list[Checks]
    log: str


def call(call_id: str, name: str, args: object) -> Call:
    """Make a Call from args shaped like bash {command}, read {path}, write {path, content} or edit {path, edits}"""
    return Call(id=call_id, name=name, label=_label(name=name, args=args), args=_shown(name=name, args=args))


def step(
    t: float,
    ms: float | None,
    thinking: str,
    text: str,
    calls: list[Call],
    stop: str | None,
    out_tokens: int | None,
    error: str | None,
) -> Step:
    thinking, text = thinking.strip(), text.strip()
    return Step(
        t=t,
        ms=ms,
        intent=_intent(thinking=thinking, text=text),
        thinking=clip(text=thinking, limit=THINK_MAX),
        thinking_chars=len(thinking),
        text=clip(text=text, limit=TEXT_MAX),
        calls=calls,
        stop=stop,
        out_tokens=out_tokens,
        error=error,
    )


def _intent(thinking: str, text: str) -> str:
    for source in (thinking, text):
        for line in source.splitlines():
            line = line.strip().lstrip("#*->").strip()
            if len(line) > 3:
                return re.split(r"(?<=[.!?])\s", line, maxsplit=1)[0][:160]
    return ""


def _shown(name: str, args: object) -> dict:
    """Clipped tool args for the viewer, keeping old and new text for edits so the viewer can show a diff"""
    if isinstance(args, dict) and name == "edit":
        edits = [e for e in args.get("edits") or [] if isinstance(e, dict)]
        return {
            "edits": [
                {
                    "old": clip(text=str(e.get("old", "")), limit=ARGS_MAX),
                    "new": clip(text=str(e.get("new", "")), limit=ARGS_MAX),
                }
                for e in edits
            ]
        }
    if isinstance(args, dict) and name == "write":
        return {"content": clip(text=str(args.get("content", "")), limit=ARGS_MAX)}
    if isinstance(args, dict) and name == "bash":
        return {"command": clip(text=str(args.get("command", "")), limit=ARGS_MAX)}
    return {"json": clip(text=json.dumps(args, indent=2), limit=ARGS_MAX)}


def _label(name: str, args: object) -> str:
    if not isinstance(args, dict):
        return name
    if name == "bash":
        command = str(args.get("command", "")).strip()
        return command.splitlines()[0][:120] if command else "bash"
    if name in ("read", "write", "edit"):
        path = str(args.get("path", "?"))
        edits = args.get("edits")
        if name == "edit" and isinstance(edits, list) and len(edits) > 1:
            return f"{path} · {len(edits)} edits"
        return path
    return f"{name} {json.dumps(args)[:100]}"
