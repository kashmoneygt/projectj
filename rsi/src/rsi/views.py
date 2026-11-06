import dataclasses
import json
import re

# What the viewer shows of an agent's run and of how a task was judged. Each harness, teacher and benchmark builds
# these from its own files, so the viewer never reads them itself.

# how much of each part the viewer shows, in characters
THINK_MAX = 30_000
TEXT_MAX = 20_000
OUT_MAX = 8_000
ARGS_MAX = 8_000


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… [{len(text) - limit} more characters]"


@dataclasses.dataclass
class Call:
    """A tool call. Its name is bash, read, write or edit for those kinds of tool, or else the tool's own"""

    id: str
    name: str
    label: str
    args: dict  # what it sent, trimmed: its command, content, edits (each old and new) or json
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
    intent: str  # the first thing the model said to itself, as the step's title
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
    """A response's timing in ms, and the size of the request it answered"""

    ttft: float | None
    total: float | None
    think: float | None
    context: int | None
    stop: str


@dataclasses.dataclass
class Trajectory:
    user: str = ""  # the first message
    items: list[Step | Note] = dataclasses.field(default_factory=list)
    inflight_at: float | None = None  # when a response still being written started, in ms
    timing: list[Timing] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Checks:
    """A group of the benchmark's checks, like the tests a fix should make pass"""

    label: str
    failing: str  # what a failing check in this group means, like "still failing"
    total: int
    passed: int
    failed: list[str]


@dataclasses.dataclass
class Verdict:
    """How the benchmark judged a task"""

    resolved: bool
    applied: bool  # the change could be put in place
    infra: bool  # the grader itself failed
    checks: list[Checks]
    log: str


def call(call_id: str, name: str, args: object) -> Call:
    """A tool call. Bash takes a command, read a path, write a path and content, and edit a path and edits"""
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
    """What a tool call sent, trimmed (an edit keeps its old and new text for a diff)"""
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
