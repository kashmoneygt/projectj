import bisect
import dataclasses
import json
import time
from collections.abc import Iterator
from pathlib import Path

from rsi import core, views


@dataclasses.dataclass
class TaskEvents:
    output_tokens: int = 0
    # tokens in the largest prompt (input + cacheRead + cacheWrite)
    peak_context_tokens: int = 0
    peak_output_tokens: int = 0  # largest single response
    capped_requests: int = 0  # responses that hit the output cap
    last_stop: str = ""  # stop reason of the last response
    tool_ms: int = 0
    tool_stats: dict[str, dict] = dataclasses.field(default_factory=dict)
    acted: int = 0  # times the variant logged that it kicked in


# HARNESS.md asks a variant to call pi.appendEntry("acted", ...) whenever it kicks in
ACTED_ENTRY = "acted"


# don't parse stream updates, since the final message repeats them
_DELTA_PREFIXES = ('{"type":"message_update"', '{"type":"tool_execution_update"')


def _records(path: Path) -> Iterator[tuple[str, dict | None]]:
    """Yield (type, record) for each line, with record None for stream updates"""
    # the agent could swap the file for a symlink or a pipe, so only read regular files
    if path.is_symlink() or not path.is_file():
        return
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(_DELTA_PREFIXES):
                yield line[9 : line.index('"', 9)], None
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield str(record.get("type", "")), record


def _count_call(ev: TaskEvents, name: str, ms: int, error: bool) -> None:
    stat = ev.tool_stats.setdefault(name, {"count": 0, "errors": 0, "ms": 0})
    stat["count"] += 1
    stat["errors"] += int(error)
    stat["ms"] += ms
    ev.tool_ms += ms


def parse_events(events: Path, seconds: float = 0.0) -> TaskEvents:
    """Sum up a run's events, counting a tool call that never finished as running until the run ended"""
    ev = TaskEvents()
    running: dict[str, str] = {}
    first_ms = answered_ms = 0
    for record_type, record in _records(path=events):
        if record is None:
            continue
        if record_type == "message_end":
            message = record.get("message", {})
            first_ms = first_ms or message.get("timestamp") or 0
            if message.get("role") != "assistant":
                continue
            # the timestamp is when the response was requested, so it ended durationMs later
            answered_ms = (message.get("timestamp") or 0) + (message.get("durationMs") or 0) or answered_ms
            usage = message.get("usage") or {}
            ev.output_tokens += usage.get("output", 0)
            prompt = usage.get("input", 0) + (usage.get("cacheRead") or 0) + (usage.get("cacheWrite") or 0)
            ev.peak_context_tokens = max(ev.peak_context_tokens, prompt)
            ev.peak_output_tokens = max(ev.peak_output_tokens, usage.get("output", 0))
            ev.capped_requests += message.get("stopReason") == "length"
            ev.last_stop = str(message.get("stopReason") or "")
        elif record_type == "tool_execution_start":
            running[str(record.get("toolCallId", ""))] = record.get("toolName", "unknown")
        elif record_type == "tool_execution_end":
            running.pop(str(record.get("toolCallId", "")), None)
            _count_call(
                ev=ev,
                name=record.get("toolName", "unknown"),
                ms=record.get("durationMs", 0) or 0,
                error=bool(record.get("isError")),
            )
        elif record_type == "entry_appended":
            entry = record.get("entry") or {}
            ev.acted += entry.get("type") == "custom" and entry.get("customType") == ACTED_ENTRY
    if seconds and first_ms and answered_ms:
        for name in running.values():
            _count_call(ev=ev, name=name, ms=max(0, round(seconds * 1000) - (answered_ms - first_ms)), error=False)
    return ev


def has_model_output(events: Path) -> bool:
    """Check if the model ever answered"""
    if not events.exists():
        return False
    with events.open(encoding="utf-8", errors="replace") as fh:
        return any('"message_end"' in line and '"assistant"' in line for line in fh)


def drop_updates(events: Path) -> None:
    """Rewrite a finished run's events without the stream updates, since the final events repeat them"""
    # each bash output update repeats all the output so far, so one task's updates can take hundreds of MB
    if events.is_symlink() or not events.is_file():
        return
    prefixes = tuple(prefix.encode() for prefix in _DELTA_PREFIXES)
    kept = events.with_name(f"{events.name}.tmp")
    with events.open("rb") as source, kept.open("wb") as dest:
        dest.writelines(line for line in source if not line.startswith(prefixes))
    kept.replace(events)


def model_error(events: Path) -> str | None:
    """Error message of the last model request, or None if it didn't fail"""
    failed: str | None = None
    for record_type, record in _records(path=events):
        if record is None or record_type != "message_end":
            continue
        msg = record.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        failed = (msg.get("errorMessage") or "model request failed") if msg.get("stopReason") == "error" else None
    return failed


def final_reply(events: Path) -> str:
    """The agent's final answer, or "" if the run didn't end on its own"""
    reply = ""
    for record_type, record in _records(path=events):
        message = (record or {}).get("message") or {}
        if record_type == "message_end" and message.get("role") == "assistant":
            # a run cut off mid-turn ends on narration, not an answer
            reply = _text_of(content=message.get("content")) if message.get("stopReason") == "stop" else ""
    return reply


def _json_lines(path: Path) -> list[dict]:
    return [record for _, record in _records(path=path) if record is not None]


# keys pi may send besides sampling and chat_template_kwargs (the forwarder log leaves out messages and tools)
_REQUEST_KEYS = ("model", "stream", "stream_options", "store", "max_tokens", "max_completion_tokens")
_CHAT_PATH = "/chat/completions"  # pi's openai-completions endpoint
_TOKENIZE_PATH = "/tokenize"  # metrics.ts counts tokens with this endpoint
_SIZES = ("system_tokens", "tool_def_tokens", "message_tokens")  # how metrics.ts splits up a request's tokens
# allow a few requests over the call budget, since compaction calls the model without going through budget.ts
BUDGET_SLACK = 5
CALL_BUDGET = "call budget"  # stop reason for a run that budget.ts stopped


def _off_spec(params: dict, model: core.ModelSpec) -> set[str]:
    """Ways one request differs from the pinned model settings"""
    found: set[str] = set()
    if params.get("model", model.model_id) != model.model_id:
        found.add(f"model {params.get('model')}")
    for key in ("max_tokens", "max_completion_tokens"):
        cap = params.get(key)
        if isinstance(cap, int) and cap > model.output_cap:
            found.add(f"output cap {cap} > {model.output_cap}")
    sent = {key: params[key] for key in model.sampling if key in params}
    if sent != model.sampling:
        found.add(f"sampling {sent}")
    kwargs = params.get("chat_template_kwargs")
    if (kwargs or {}) != model.chat_template_kwargs:
        found.add(f"chat_template_kwargs {kwargs}")
    expected = {*_REQUEST_KEYS, "chat_template_kwargs", *model.sampling}
    found |= {f"unexpected key {key}" for key in params.keys() - expected}
    return found


def contract_violations(requests: Path, model: core.ModelSpec) -> list[str]:
    """Requests in the forwarder log that break the pinned model settings"""
    found: set[str] = set()
    for line in _json_lines(path=requests):
        path = str(line.get("path", "")).split("?")[0]
        if "unread" in line:
            found.add(f"unread {path or 'traffic'}: {line['unread']}")
        elif "path" not in line or path == _TOKENIZE_PATH:
            continue
        elif not path.endswith(_CHAT_PATH):
            found.add(f"request to {path}")
        else:
            found |= _off_spec(params=line.get("params") or {}, model=model)
    return sorted(found)


@dataclasses.dataclass
class ModelCalls:
    requests: int = 0
    input_tokens: int = 0  # uncached prompt tokens, which is what pi calls input
    output_tokens: int = 0  # thinking included
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0


def model_calls(requests: Path) -> ModelCalls:
    """Model requests and token usage the forwarder logged for one run"""
    calls = ModelCalls()
    for line in _json_lines(path=requests):
        if "path" in line and str(line["path"]).split("?")[0] != _TOKENIZE_PATH:
            calls.requests += 1
        usage = line.get("usage")
        if isinstance(usage, dict):
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            calls.input_tokens += max(0, (usage.get("prompt_tokens") or 0) - cached)
            calls.output_tokens += usage.get("completion_tokens") or 0
            calls.cache_read_tokens += cached
            calls.reasoning_tokens += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return calls


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(c["text"]) for c in content if isinstance(c, dict) and c.get("type") == "text" and c.get("text")
    )


def task_metrics(
    instance_dir: Path, model: core.ModelSpec, seconds: float, patch: str, max_calls: int
) -> core.TaskMetrics:
    events = parse_events(events=instance_dir / "events.jsonl", seconds=seconds)
    requests = instance_dir / "requests.jsonl"
    # take requests and tokens from the forwarder log, so calls from an extension or a second pi process count too
    calls = model_calls(requests=requests)
    lines = _json_lines(path=instance_dir / "metrics.jsonl")
    largest = max(
        (m for m in lines if "system_tokens" in m), key=lambda m: sum(m.get(k, 0) for k in _SIZES), default={}
    )
    timing = [m for m in lines if m.get("type") == "timing"]
    ttfts = [m.get("ttft_ms", 0) or 0 for m in timing] or [0]
    gen_s = max(seconds - events.tool_ms / 1000.0, 1.0)
    violations = contract_violations(requests=requests, model=model)
    if max_calls and calls.requests > max_calls + BUDGET_SLACK:
        violations.append(f"{calls.requests} model requests, over the budget of {max_calls}")
    stopped = any(m.get("type") == "budget" for m in lines)
    return core.TaskMetrics(
        requests=calls.requests,
        input_tokens=calls.input_tokens,
        output_tokens=calls.output_tokens,
        cache_read_tokens=calls.cache_read_tokens,
        reasoning_tokens=calls.reasoning_tokens,
        tool_ms=events.tool_ms,
        tool_stats={name: core.ToolStat(**stat) for name, stat in events.tool_stats.items()},
        tok_s=round(events.output_tokens / gen_s, 2),
        context_pct=round(100.0 * events.peak_context_tokens / model.ctx, 1),
        system_tokens=largest.get("system_tokens", 0),
        tool_def_tokens=largest.get("tool_def_tokens", 0),
        message_tokens=largest.get("message_tokens", 0),
        ttft_first_ms=float(ttfts[0]),
        ttft_avg_ms=round(sum(ttfts) / len(ttfts), 1),
        ttft_max_ms=float(max(ttfts)),
        model_ms=int(sum(m.get("gen_ms", 0) or 0 for m in timing)),
        think_ms=int(sum(m.get("think_ms", 0) or 0 for m in timing)),
        # fall back to pi's events, since metrics.ts writes nothing without the model's root URL
        stop_reason=CALL_BUDGET if stopped else (timing[-1].get("stop_reason") if timing else "") or events.last_stop,
        patch_lines=len([line for line in patch.splitlines() if line.strip()]),
        peak_output_tokens=events.peak_output_tokens,
        capped_requests=events.capped_requests,
        acted=events.acted,
        contract_violations=violations,
    )


def system_context(metrics: Path) -> str:
    """System prompt and tool definitions of the first request, as metrics.ts saved them"""
    context = next((line for line in _json_lines(path=metrics) if line.get("type") == "context"), None)
    if context is None:
        return ""
    tools = json.dumps(context.get("tools"), indent=1, ensure_ascii=False)
    return f"## System prompt\n\n{context.get('system', '')}\n\n## Tool definitions\n\n```json\n{tools}\n```\n"


def trajectory(events: Path, metrics: Path) -> views.Trajectory:
    """Each response with its tool calls and timing, plus notes in between like retries and compactions"""
    view = views.Trajectory(timing=_timing(metrics=metrics))
    start: float | None = None
    calls: dict[str, views.Call] = {}
    for record_type, record in _records(path=events):
        if record is None:
            continue
        message = record.get("message") or {}
        role = message.get("role")
        if record_type == "message_start" and role == "assistant":
            view.inflight_at = message.get("timestamp") or time.time() * 1000
        elif record_type == "message_end" and role == "user" and not view.user:
            view.user = _text_of(content=message.get("content"))
            start = start or message.get("timestamp")
        elif record_type == "message_end" and role == "assistant":
            view.inflight_at = None
            stamp = message.get("timestamp") or 0
            start = start or stamp
            step = _step(message=message, t=stamp - start)
            calls.update({call.id: call for call in step.calls})
            view.items.append(step)
        elif record_type == "message_end" and role == "toolResult":
            call = calls.get(message.get("toolCallId", ""))
            if call is not None:
                out = _text_of(content=message.get("content"))
                call.ended(out=out, ok=not message.get("isError"), ms=message.get("durationMs"))
        elif record_type == "message_end" and role == "custom":
            view.items.append(_added(message=message))
        elif record_type == "auto_retry_start":
            error = str(record.get("errorMessage", ""))[:200]
            view.items.append(views.Note(text=f"request failed, retry {record.get('attempt')}: {error}"))
        elif record_type == "compaction_start":
            view.items.append(views.Note(text="context compacted"))
        elif record_type == "entry_appended":
            entry = record.get("entry") or {}
            if entry.get("type") == "custom_message":
                view.items.append(_added(message=entry))
            elif entry.get("type") == "context_edit":
                edit = "dropped from" if entry.get("replacement") is None else "changed in"
                view.items.append(views.Note(text=f"an earlier message was {edit} the context"))
    return view


def _added(message: dict) -> views.Note:
    """A message the package added for the model, via pi.sendMessage or an event handler's result"""
    text = f"harness message ({message.get('customType')}): {_text_of(content=message.get('content'))}"
    return views.Note(text=views.clip(text=text, limit=400))


def _step(message: dict, t: float) -> views.Step:
    parts = [p for p in message.get("content") or [] if isinstance(p, dict)]
    calls = [
        views.call(call_id=str(p.get("id") or ""), name=str(p.get("name") or "?"), args=_viewed_args(part=p))
        for p in parts
        if p.get("type") == "toolCall"
    ]
    stop = message.get("stopReason")
    return views.step(
        t=t,
        ms=message.get("durationMs"),
        thinking="\n".join(str(p.get("thinking") or "") for p in parts if p.get("type") == "thinking"),
        text=_text_of(content=parts),
        calls=calls,
        stop=stop,
        out_tokens=(message.get("usage") or {}).get("output"),
        error=message.get("errorMessage") if stop == "error" else None,
    )


def _viewed_args(part: dict) -> object:
    """Arguments of a tool call, with an edit call shown as its path plus the old and new text of each edit"""
    args = part.get("arguments")
    if part.get("name") != "edit" or not isinstance(args, dict):
        return args
    edits = args["edits"] if isinstance(args.get("edits"), list) else [args]
    return {
        "path": args.get("path"),
        "edits": [{"old": e.get("oldText", ""), "new": e.get("newText", "")} for e in edits if isinstance(e, dict)],
    }


def _timing(metrics: Path) -> list[views.Timing]:
    """Timing of each response and the size of the request it answered, from the lines metrics.ts wrote"""
    lines = _json_lines(path=metrics)
    sent = sorted((line for line in lines if "type" not in line), key=lambda line: line.get("ts", 0))
    starts = [line.get("ts", 0) for line in sent]
    rows: list[views.Timing] = []
    for timing in (line for line in lines if line.get("type") == "timing"):
        i = bisect.bisect_right(starts, timing.get("ts", 0)) - 1
        request = sent[i] if i >= 0 else {}
        sizes = [request.get(key) for key in _SIZES]
        rows.append(
            views.Timing(
                ttft=timing.get("ttft_ms"),
                total=timing.get("gen_ms"),
                think=timing.get("think_ms"),
                context=None if None in sizes else sum(sizes),
                stop=timing.get("stop_reason") or "",
            )
        )
    return rows
