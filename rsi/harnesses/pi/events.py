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
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    # input + cacheRead + cacheWrite of the largest prompt
    peak_context_tokens: int = 0
    peak_output_tokens: int = 0  # the largest single response
    capped_requests: int = 0  # responses cut at the output cap
    last_stop: str = ""  # the last response's stop reason
    tool_ms: int = 0
    tool_stats: dict[str, dict] = dataclasses.field(default_factory=dict)
    final_event: str = ""
    acted: int = 0  # times the variant said it kicked in


# what a variant appends with pi.appendEntry when it acts (named in HARNESS.md)
ACTED_ENTRY = "acted"


# stream updates, which the final message repeats
_DELTA_PREFIXES = ('{"type":"message_update"', '{"type":"tool_execution_update"')


def _records(path: Path) -> Iterator[tuple[str, dict | None]]:
    """Yield (type, record) per line, with record None for stream updates"""
    if not path.exists():
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
    """Sum up a run's events. A tool call still running at the end counts until the run ended"""
    ev = TaskEvents()
    running: dict[str, str] = {}
    first_ms = answered_ms = 0
    for record_type, record in _records(path=events):
        if record_type:
            ev.final_event = record_type
        if record is None:
            continue
        if record_type == "message_end":
            message = record.get("message", {})
            first_ms = first_ms or message.get("timestamp") or 0
            if message.get("role") != "assistant":
                continue
            # a response's timestamp is when it was asked for, so it ended durationMs later
            answered_ms = (message.get("timestamp") or 0) + (message.get("durationMs") or 0) or answered_ms
            usage = message.get("usage") or {}
            ev.output_tokens += usage.get("output", 0)
            ev.cache_read_tokens += usage.get("cacheRead", 0)
            ev.reasoning_tokens += usage.get("reasoning", 0) or 0
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


def model_error(events: Path) -> str | None:
    """The last model request's error, if it failed"""
    failed: str | None = None
    for record_type, record in _records(path=events):
        if record is None or record_type != "message_end":
            continue
        msg = record.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        failed = (msg.get("errorMessage") or "model request failed") if msg.get("stopReason") == "error" else None
    return failed


def _json_lines(path: Path) -> list[dict]:
    return [record for _, record in _records(path=path) if record is not None]


# other keys pi sends with each request
_REQUEST_KEYS = ("model", "stream", "stream_options", "store", "max_tokens", "max_completion_tokens")
_CHAT_PATH = "/chat/completions"  # pi's openai-completions endpoint
_TOKENIZE_PATH = "/tokenize"  # the metrics extension counts tokens with it
_SIZES = ("system_tokens", "tool_def_tokens", "message_tokens")  # of a request, as the metrics extension counts them


def _off_spec(params: dict, model: core.ModelSpec) -> set[str]:
    """How one request strays from the pinned model settings"""
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
    """Requests off the pinned model settings, from the forwarder log"""
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
    input_tokens: int = 0  # uncached prompt tokens, as pi counts them
    output_tokens: int = 0


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
    return calls


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(c["text"]) for c in content if isinstance(c, dict) and c.get("type") == "text" and c.get("text")
    )


def task_metrics(instance_dir: Path, model: core.ModelSpec, seconds: float, patch: str) -> core.TaskMetrics:
    events = parse_events(events=instance_dir / "events.jsonl", seconds=seconds)
    requests = instance_dir / "requests.jsonl"
    # every model call from the forwarder log, including those an extension or a second pi made
    calls = model_calls(requests=requests)
    lines = _json_lines(path=instance_dir / "metrics.jsonl")
    largest = max(
        (m for m in lines if "system_tokens" in m), key=lambda m: sum(m.get(k, 0) for k in _SIZES), default={}
    )
    timing = [m for m in lines if m.get("type") == "timing"]
    ttfts = [m.get("ttft_ms", 0) or 0 for m in timing] or [0]
    gen_s = max(seconds - events.tool_ms / 1000.0, 1.0)
    return core.TaskMetrics(
        requests=calls.requests,
        input_tokens=calls.input_tokens,
        output_tokens=calls.output_tokens,
        cache_read_tokens=events.cache_read_tokens,
        reasoning_tokens=events.reasoning_tokens,
        tool_ms=events.tool_ms,
        tool_stats={name: core.ToolStat(**stat) for name, stat in events.tool_stats.items()},
        final_event=events.final_event,
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
        # metrics.ts writes nothing without the model's root URL
        stop_reason=(timing[-1].get("stop_reason") if timing else "") or events.last_stop,
        patch_lines=len([line for line in patch.splitlines() if line.strip()]),
        peak_output_tokens=events.peak_output_tokens,
        capped_requests=events.capped_requests,
        acted=events.acted,
        contract_violations=contract_violations(requests=requests, model=model),
    )


def system_context(metrics: Path) -> str:
    """The system prompt and tool definitions from the first request (saved by metrics.ts)"""
    context = next((line for line in _json_lines(path=metrics) if line.get("type") == "context"), None)
    if context is None:
        return ""
    tools = json.dumps(context.get("tools"), indent=1, ensure_ascii=False)
    return f"## System prompt\n\n{context.get('system', '')}\n\n## Tool definitions\n\n```json\n{tools}\n```\n"


def trajectory(events: Path, metrics: Path) -> views.Trajectory:
    """Each response with its tool calls, the notes between them, and each response's timing"""
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
    """A message the package added for the model, from pi.sendMessage or an event handler's result"""
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
    """A tool call's arguments, with an edit's given as its path and each edit's old and new text"""
    args = part.get("arguments")
    if part.get("name") != "edit" or not isinstance(args, dict):
        return args
    edits = args["edits"] if isinstance(args.get("edits"), list) else [args]
    return {
        "path": args.get("path"),
        "edits": [{"old": e.get("oldText", ""), "new": e.get("newText", "")} for e in edits if isinstance(e, dict)],
    }


def _timing(metrics: Path) -> list[views.Timing]:
    """Each response's timing and the size of the request it answered, from the metrics extension's lines"""
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
