import json
from pathlib import Path

from rsi import core, folders, paths

pi_events = folders.module(root=paths.harnesses_dir(), name="pi", file="events.py")


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_parse_events(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    usage = {"input": 100, "output": 40, "cacheRead": 10, "reasoning": 5, "totalTokens": 140}
    _write(
        path=events,
        records=[
            {"type": "session", "id": "x"},
            {"type": "message_start", "message": {"role": "user", "content": "hi"}},
            {"type": "message_end", "message": {"role": "assistant", "usage": usage}},
            {"type": "tool_execution_end", "toolName": "bash", "durationMs": 1234},
            {"type": "turn_end", "message": {"role": "assistant"}},
            {
                "type": "message_end",
                "message": {"role": "assistant", "usage": {"input": 220, "output": 60, "cacheRead": 100}},
            },
            {"type": "agent_end"},
        ],
    )
    with events.open("a") as fh:
        fh.write("not-json\n")
    ev = pi_events.parse_events(events=events)
    assert ev.output_tokens == 100
    assert ev.peak_context_tokens == 320
    assert ev.tool_ms == 1234
    assert ev.tool_stats == {"bash": {"count": 1, "errors": 0, "ms": 1234}}


def test_parse_events_missing(tmp_path: Path) -> None:
    ev = pi_events.parse_events(events=tmp_path / "nope.jsonl")
    assert ev.output_tokens == 0
    assert ev.tool_ms == 0


def test_a_finished_runs_events_drop_the_stream_updates_and_keep_the_rest(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    end = {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "bash", "durationMs": 900}
    # pi writes compact JSON, which is how the updates are told apart without parsing them
    lines = [
        '{"type":"message_start","message":{"role":"assistant"}}',
        '{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"Lo"}}',
        '{"type":"tool_execution_update","toolCallId":"c1","partialResult":"' + "x" * 1000 + '"}',
        json.dumps(end, separators=(",", ":")),
    ]
    events.write_text("".join(f"{line}\n" for line in lines))
    before = pi_events.parse_events(events=events)
    pi_events.drop_updates(events=events)
    assert events.read_text().splitlines() == [lines[0], lines[3]]
    assert pi_events.parse_events(events=events) == before
    pi_events.drop_updates(events=tmp_path / "missing.jsonl")


def test_a_tool_call_still_running_at_the_end_counts_until_the_run_ended(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    _write(
        path=events,
        records=[
            {"type": "message_end", "message": {"role": "user", "timestamp": 1_000}},
            {"type": "message_end", "message": {"role": "assistant", "timestamp": 61_000, "usage": {"output": 5}}},
            {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "read"},
            {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "read", "durationMs": 40},
            # this response was requested at 121 s and took 20 s, so the bash call after it started at 141 s
            {
                "type": "message_end",
                "message": {"role": "assistant", "timestamp": 121_000, "durationMs": 20_000, "usage": {"output": 5}},
            },
            {"type": "tool_execution_start", "toolCallId": "c2", "toolName": "bash"},
        ],
    )
    ev = pi_events.parse_events(events=events, seconds=3600.0)
    assert ev.tool_stats == {
        "read": {"count": 1, "errors": 0, "ms": 40},
        "bash": {"count": 1, "errors": 0, "ms": 3_460_000},
    }
    assert ev.tool_ms == 3_460_040
    assert pi_events.parse_events(events=events).tool_ms == 40


def test_parse_events_counts_when_the_variant_acted(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    acted = {"type": "custom", "customType": pi_events.ACTED_ENTRY, "data": {"what": "blocked a bare pytest"}}
    _write(
        path=events,
        records=[
            {"type": "entry_appended", "entry": acted},
            {"type": "entry_appended", "entry": {"type": "custom", "customType": "notes", "data": {}}},
            {"type": "entry_appended", "entry": {"type": "custom_message", "customType": pi_events.ACTED_ENTRY}},
            {"type": "entry_appended", "entry": acted},
        ],
    )
    assert pi_events.parse_events(events=events).acted == 2


def test_a_trajectory_shows_the_messages_a_package_added_and_its_edits(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    review = [{"type": "text", "text": "Check the diff."}]
    edit = {"type": "context_edit", "targetId": "b", "replacement": {"content": ""}}
    _write(
        path=events,
        records=[
            {"type": "message_end", "message": {"role": "user", "content": "Fix it", "timestamp": 1000}},
            # messages from pi.sendMessage arrive as message_end events, and ones returned by a turn_end or
            # agent_before_settle handler arrive as entry_appended events
            {"type": "message_end", "message": {"role": "custom", "customType": "nudge", "content": "Run the tests."}},
            {"type": "entry_appended", "entry": {"type": "custom_message", "customType": "review", "content": review}},
            {"type": "entry_appended", "entry": {"type": "context_edit", "targetId": "a", "replacement": None}},
            {"type": "entry_appended", "entry": edit},
            {"type": "entry_appended", "entry": {"type": "custom", "customType": pi_events.ACTED_ENTRY, "data": {}}},
        ],
    )
    view = pi_events.trajectory(events=events, metrics=tmp_path / "metrics.jsonl")
    assert view.user == "Fix it"
    assert [item.text for item in view.items] == [
        "harness message (nudge): Run the tests.",
        "harness message (review): Check the diff.",
        "an earlier message was dropped from the context",
        "an earlier message was changed in the context",
    ]


def test_task_metrics_take_the_largest_request_and_add_up_the_timing(tmp_path: Path) -> None:
    model = core.ModelSpec(name="q", source="s", model_id="qwen", ctx=131072)
    m = pi_events.task_metrics(instance_dir=tmp_path, model=model, seconds=10.0, patch="", max_calls=0)
    assert (m.system_tokens, m.ttft_first_ms, m.ttft_avg_ms, m.model_ms, m.stop_reason) == (0, 0.0, 0.0, 0, "")
    _write(
        path=tmp_path / "metrics.jsonl",
        records=[
            {"ts": 1, "system_tokens": 100, "tool_def_tokens": 300, "message_tokens": 50},
            {"type": "timing", "ts": 2, "ttft_ms": 120, "gen_ms": 900, "think_ms": 400, "stop_reason": "toolUse"},
            {"ts": 3, "system_tokens": 100, "tool_def_tokens": 300, "message_tokens": 400},
            {"type": "timing", "ts": 4, "ttft_ms": 300, "gen_ms": 1500, "think_ms": 100, "stop_reason": "stop"},
        ],
    )
    m = pi_events.task_metrics(instance_dir=tmp_path, model=model, seconds=10.0, patch="", max_calls=0)
    assert (m.system_tokens, m.tool_def_tokens, m.message_tokens) == (100, 300, 400)
    assert (m.ttft_first_ms, m.ttft_avg_ms, m.ttft_max_ms) == (120.0, 210.0, 300.0)
    assert (m.model_ms, m.think_ms, m.stop_reason) == (2400, 500, "stop")


def test_model_error(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    denied = "403: Host host.docker.internal is not allowed"
    _write(
        path=events,
        records=[
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": denied}},
            {"type": "agent_end", "aborted": False},
        ],
    )
    assert pi_events.model_error(events=events) == denied
    _write(
        path=events,
        records=[
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": "boom"}},
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop"}},
        ],
    )
    assert pi_events.model_error(events=events) is None
    assert pi_events.model_error(events=tmp_path / "missing.jsonl") is None


def test_has_model_output(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    assert pi_events.has_model_output(events=events) is False
    _write(path=events, records=[{"type": "session", "id": "x"}])
    assert pi_events.has_model_output(events=events) is False
    _write(
        path=events,
        records=[
            {"type": "session", "id": "x"},
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse"}},
        ],
    )
    assert pi_events.has_model_output(events=events) is True


def test_the_final_reply_is_the_text_of_the_last_answer(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    looked = [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "Let me look."}, {"type": "toolCall"}]
    answer = [{"type": "thinking", "thinking": "found it"}, {"type": "text", "text": "It's in a.py."}]
    looking = {"type": "message_end", "message": {"role": "assistant", "content": looked, "stopReason": "toolUse"}}
    done = {"type": "message_end", "message": {"role": "assistant", "content": answer, "stopReason": "stop"}}
    _write(path=events, records=[looking, done])
    assert pi_events.final_reply(events=events) == "It's in a.py."
    # a run stopped mid-turn ends on narration like "Let me look.", which isn't an answer
    result = {"type": "message_end", "message": {"role": "toolResult", "content": "x"}}
    _write(path=events, records=[done, looking, result])
    assert pi_events.final_reply(events=events) == ""
    # the agent could swap events.jsonl for a symlink to any file on the host, so final_reply ignores symlinks
    _write(path=events, records=[done])
    (tmp_path / "linked.jsonl").symlink_to(events)
    assert pi_events.final_reply(events=tmp_path / "linked.jsonl") == ""
    assert pi_events.final_reply(events=tmp_path / "missing.jsonl") == ""


def test_a_run_stopped_at_its_call_budget_says_so_and_one_past_it_breaks_the_contract(tmp_path: Path) -> None:
    model = core.ModelSpec(name="m", source="s", model_id="m")
    _write(path=tmp_path / "metrics.jsonl", records=[{"type": "timing", "stop_reason": "toolUse"}])
    chat = {"method": "POST", "path": "/v1/chat/completions", "params": {"model": "m"}}
    _write(path=tmp_path / "requests.jsonl", records=[chat] * 101)
    m = pi_events.task_metrics(instance_dir=tmp_path, model=model, seconds=10.0, patch="", max_calls=100)
    assert m.stop_reason == "toolUse" and m.contract_violations == []
    _write(
        path=tmp_path / "metrics.jsonl",
        records=[{"type": "timing", "stop_reason": "aborted"}, {"type": "budget", "calls": 100}],
    )
    # compaction calls the model without going through budget.ts, so allow a few requests over the budget
    _write(path=tmp_path / "requests.jsonl", records=[chat] * (100 + pi_events.BUDGET_SLACK + 1))
    m = pi_events.task_metrics(instance_dir=tmp_path, model=model, seconds=10.0, patch="", max_calls=100)
    assert m.stop_reason == pi_events.CALL_BUDGET
    assert m.contract_violations == ["106 model requests, over the budget of 100"]
