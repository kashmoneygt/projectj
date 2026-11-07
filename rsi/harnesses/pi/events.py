import bisect
import dataclasses
import json
import time
from collections import Counter
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
    first_ms = asked_ms = 0
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
            asked_ms = message.get("timestamp") or asked_ms
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
    if seconds and first_ms and asked_ms:
        for name in running.values():
            _count_call(ev=ev, name=name, ms=max(0, round(seconds * 1000) - (asked_ms - first_ms)), error=False)
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


def parse_breakdown(metrics: Path) -> tuple[int, int, int]:
    """System prompt, tool definition and message tokens of the largest request"""
    lines = [m for m in _json_lines(path=metrics) if "system_tokens" in m]
    largest = max(
        lines,
        key=lambda m: m.get("system_tokens", 0) + m.get("tool_def_tokens", 0) + m.get("message_tokens", 0),
        default={},
    )
    return largest.get("system_tokens", 0), largest.get("tool_def_tokens", 0), largest.get("message_tokens", 0)


def parse_timing(metrics: Path) -> tuple[float, float, float, int, int, str]:
    """ttft_first, ttft_avg, ttft_max, total generation ms, total thinking ms, final stop reason"""
    lines = [m for m in _json_lines(path=metrics) if m.get("type") == "timing"]
    if not lines:
        return 0.0, 0.0, 0.0, 0, 0, ""
    ttfts = [m.get("ttft_ms", 0) or 0 for m in lines]
    return (
        float(ttfts[0]),
        round(sum(ttfts) / len(ttfts), 1),
        float(max(ttfts)),
        int(sum(m.get("gen_ms", 0) or 0 for m in lines)),
        int(sum(m.get("think_ms", 0) or 0 for m in lines)),
        str(lines[-1].get("stop_reason", "")),
    )


# compact transcripts fit one read_file call; full ones keep everything
TRANSCRIPT_MAX_CHARS = 12_000
_KEEP_FIRST_ENTRIES = 6
FULL_OUTPUT_CHARS = 2_000  # per tool call input and output in the full transcript
TASK_GLIMPSE_CHARS = 800  # of the task message in the compact transcript
UNFINISHED = "still running when the run ended"


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _excerpt(text: str, head: int, tail: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= head + tail + 3:
        return flat
    return f"{flat[:head]} … {flat[-tail:]}"


def _tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _block(text: str, limit: int) -> str:
    """Keep multi-line text, or its start and end if it's too long"""
    text = text.strip("\n")
    if len(text) <= limit:
        return text
    head = limit // 4
    return f"{text[:head]}\n… ({len(text) - limit} characters omitted) …\n{text[-(limit - head) :]}"


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(c["text"]) for c in content if isinstance(c, dict) and c.get("type") == "text" and c.get("text")
    )


def _output_of(record: dict) -> str:
    result = record.get("result")
    return _text_of(content=result.get("content") if isinstance(result, dict) else result)


def _stop_note(message: dict, error_chars: int) -> str:
    stop = message.get("stopReason") or "-"
    if stop == "length":
        return "stop length (cut at the output cap)"
    if stop == "error":
        return f"stop error: {_clip(text=str(message.get('errorMessage') or ''), limit=error_chars)}"
    return "" if stop == "toolUse" else f"stop {stop}"


def _call_label(name: str, args: object) -> str:
    if not isinstance(args, dict):
        return f"{name} {_clip(text=str(args), limit=120)}"
    if name == "bash":
        lines = str(args.get("command", "")).strip().splitlines() or [""]
        more = f" (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
        return f"bash: {_clip(text=lines[0], limit=160)}{more}"
    if name == "read":
        offset = f" from line {args['offset']}" if args.get("offset") else ""
        return f"read {args.get('path', '?')}{offset}"
    if name == "edit":
        edits = args.get("edits")
        count = f" ({len(edits)} edits)" if isinstance(edits, list) and len(edits) != 1 else ""
        return f"edit {args.get('path', '?')}{count}"
    if name == "write":
        return f"write {args.get('path', '?')} ({len(str(args.get('content', '')).splitlines())} lines)"
    return f"{name} {_clip(text=json.dumps(args), limit=120)}"


def _call_outcome(name: str, record: dict | None) -> str:
    if record is None:
        return UNFINISHED
    lines = [line for line in _output_of(record=record).splitlines() if line.strip()]
    if record.get("isError"):
        return f"error: {_clip(text=' '.join(lines[-3:]), limit=240)}" if lines else "error"
    if name in ("edit", "write") or not lines:
        return "ok"
    if name == "read":
        return f"ok, {len(lines)} lines"
    return f"ok, {len(lines)} lines, last: {_clip(text=lines[-1], limit=160)}"


def _took(record: dict | None) -> str:
    """A tool call's duration, if it took at least a second"""
    ms = (record or {}).get("durationMs") or 0
    return f" ({ms / 1000:.0f} s)" if ms >= 1000 else ""


def _call_input(name: str, args: object) -> str:
    if not isinstance(args, dict):
        return _block(text=str(args), limit=FULL_OUTPUT_CHARS)
    if name == "bash":
        return "$ " + _block(text=str(args.get("command", "")), limit=FULL_OUTPUT_CHARS)
    if name == "read":
        window = {key: args[key] for key in ("offset", "limit") if args.get(key)}
        return f"{args.get('path', '?')} {json.dumps(window) if window else ''}".rstrip()
    if name == "write":
        return f"{args.get('path', '?')}\n" + _block(text=str(args.get("content", "")), limit=FULL_OUTPUT_CHARS)
    rest = {key: value for key, value in args.items() if key != "path"}
    head = f"{args['path']}\n" if "path" in args else ""
    return head + _block(text=json.dumps(rest, indent=1), limit=FULL_OUTPUT_CHARS)


@dataclasses.dataclass
class _Entry:
    lines: list[str]
    calls: Counter[str] = dataclasses.field(default_factory=Counter)  # calls and errors sum up what _fit drops
    errors: int = 0


def _turn_entry(index: int, message: dict, results: dict[str, dict], last: bool) -> _Entry:
    usage = message.get("usage") or {}
    parts = [p for p in message.get("content") or [] if isinstance(p, dict)]
    thinking = "\n".join(str(p.get("thinking", "")) for p in parts if p.get("type") == "thinking")
    text = _text_of(content=parts)
    head = f"[{index}] {_tokens(n=usage.get('output') or 0)} tok out"
    note = _stop_note(message=message, error_chars=160)
    entry = _Entry(lines=[f"{head}, {note}" if note else head])
    if thinking.strip():
        reasoning = usage.get("reasoning") or 0
        size = f" ({_tokens(n=reasoning)} tok)" if reasoning else ""
        excerpt = _excerpt(text=thinking, head=160, tail=1200) if last else _excerpt(text=thinking, head=120, tail=200)
        entry.lines.append(f"    thinking{size}: {excerpt}")
    if text.strip():
        entry.lines.append(f"    text: {_excerpt(text=text, head=240, tail=600 if last else 120)}")
    for call in (p for p in parts if p.get("type") == "toolCall"):
        name = str(call.get("name", "?"))
        record = results.get(str(call.get("id", "")))
        label = _call_label(name=name, args=call.get("arguments"))
        entry.lines.append(f"    {label} → {_call_outcome(name=name, record=record)}{_took(record=record)}")
        entry.calls[name] += 1
        entry.errors += int(bool(record and record.get("isError")))
    return entry


def _full_entry(index: int, message: dict, results: dict[str, dict]) -> _Entry:
    usage = message.get("usage") or {}
    parts = [p for p in message.get("content") or [] if isinstance(p, dict)]
    thinking = "\n".join(str(p.get("thinking", "")) for p in parts if p.get("type") == "thinking").strip()
    text = _text_of(content=parts).strip()
    head = f"## turn {index} · {_tokens(n=usage.get('output') or 0)} tok out"
    if usage.get("reasoning"):
        head += f" (thinking {_tokens(n=usage['reasoning'])})"
    note = _stop_note(message=message, error_chars=300)
    entry = _Entry(lines=["", f"{head} · {note}" if note else head])
    if thinking:
        entry.lines += ["### thinking", thinking]
    if text:
        entry.lines += ["### text", text]
    for call in (p for p in parts if p.get("type") == "toolCall"):
        name = str(call.get("name", "?"))
        record = results.get(str(call.get("id", "")))
        verdict = UNFINISHED if record is None else "error" if record.get("isError") else "ok"
        entry.lines += [
            f"### {name}",
            _call_input(name=name, args=call.get("arguments")),
            f"→ {verdict}{_took(record=record)}",
        ]
        output = _output_of(record=record or {})
        if output.strip():
            entry.lines.append(_block(text=output, limit=FULL_OUTPUT_CHARS))
    return entry


def transcript(events: Path, full: bool = False) -> str:
    # assistant messages (dict) and one-line notes (str), in order
    items: list[dict | str] = []
    results: dict[str, dict] = {}
    task: str | None = None
    for record_type, record in _records(path=events):
        if record is None:
            continue
        if record_type == "message_end":
            message = record.get("message") or {}
            role = message.get("role")
            body = _clip(text=_text_of(content=message.get("content")), limit=4_000 if full else 400)
            if role == "assistant":
                items.append(message)
            elif role == "user":
                if task is None:
                    task = _text_of(content=message.get("content")).strip()
                else:
                    items.append(f"user message: {body}")
            elif role not in ("toolResult", "system"):
                items.append(f"{message.get('customType') or role} message: {body}")
        elif record_type == "tool_execution_end":
            results[str(record.get("toolCallId", ""))] = record
        elif record_type == "compaction_end":
            items.append(f"context compacted ({record.get('reason', '?')})")
        elif record_type == "auto_retry_start":
            reason = _clip(text=str(record.get("errorMessage")), limit=160)
            items.append(f"request retry {record.get('attempt', '?')}: {reason}")
    last_turn = max((i for i, item in enumerate(items) if isinstance(item, dict)), default=-1)
    if last_turn < 0:
        return "(no model output)\n"
    entries: list[_Entry] = []
    turn = 0
    for i, item in enumerate(items):
        if isinstance(item, dict):
            turn += 1
            if full:
                entries.append(_full_entry(index=turn, message=item, results=results))
            else:
                entries.append(_turn_entry(index=turn, message=item, results=results, last=i == last_turn))
        else:
            entries.append(_Entry(lines=[f"    [{item}]"]))
    kept = entries if full else _fit(entries=entries, max_chars=TRANSCRIPT_MAX_CHARS)
    head = ""
    if task:
        head = f"## task message\n{task}\n" if full else f"task: {_clip(text=task, limit=TASK_GLIMPSE_CHARS)}\n\n"
    return head + "\n".join(line for entry in kept for line in entry.lines) + "\n"


def _fit(entries: list[_Entry], max_chars: int) -> list[_Entry]:
    """Keep the first entries and as many last ones as fit, summing up the middle"""
    size = [sum(len(line) + 1 for line in entry.lines) for entry in entries]
    if sum(size) <= max_chars:
        return entries
    budget = max_chars - sum(size[:_KEEP_FIRST_ENTRIES])
    start = len(entries)
    while start > _KEEP_FIRST_ENTRIES and budget >= size[start - 1]:
        start -= 1
        budget -= size[start]
    middle = entries[_KEEP_FIRST_ENTRIES:start]
    if not middle:
        return entries
    calls: Counter[str] = Counter()
    for entry in middle:
        calls.update(entry.calls)
    used = ", ".join(f"{name} {count}" for name, count in calls.most_common()) or "none"
    errors = sum(entry.errors for entry in middle)
    note = _Entry(lines=[f"    … {len(middle)} entries omitted (tool calls: {used}; {errors} errors) …"])
    return [*entries[:_KEEP_FIRST_ENTRIES], note, *entries[start:]]


def task_metrics(instance_dir: Path, model: core.ModelSpec, seconds: float, patch: str) -> core.TaskMetrics:
    events = parse_events(events=instance_dir / "events.jsonl", seconds=seconds)
    requests = instance_dir / "requests.jsonl"
    # every model call from the forwarder log, including those an extension or a second pi made
    calls = model_calls(requests=requests)
    system_tokens, tool_def_tokens, message_tokens = parse_breakdown(metrics=instance_dir / "metrics.jsonl")
    ttft_first, ttft_avg, ttft_max, model_ms, think_ms, stop_reason = parse_timing(
        metrics=instance_dir / "metrics.jsonl"
    )
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
        system_tokens=system_tokens,
        tool_def_tokens=tool_def_tokens,
        message_tokens=message_tokens,
        ttft_first_ms=ttft_first,
        ttft_avg_ms=ttft_avg,
        ttft_max_ms=ttft_max,
        model_ms=model_ms,
        think_ms=think_ms,
        stop_reason=stop_reason or events.last_stop,  # metrics.ts writes nothing without the model's root URL
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
        elif record_type == "auto_retry_start":
            error = str(record.get("errorMessage", ""))[:200]
            view.items.append(views.Note(text=f"request failed, retry {record.get('attempt')}: {error}"))
        elif record_type == "compaction_start":
            view.items.append(views.Note(text="context compacted"))
        elif record_type == "entry_appended" and (record.get("entry") or {}).get("type") == "context_edit":
            view.items.append(views.Note(text="last response dropped from the context"))
    return view


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
    """A tool call's arguments, with an edit's in the viewer's shape: a path and edits, each old and new"""
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
        sizes = [request.get(key) for key in ("system_tokens", "tool_def_tokens", "message_tokens")]
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
