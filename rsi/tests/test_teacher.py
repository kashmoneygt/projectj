import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import IO

import pytest

from rsi import core, folders, models, paths, teacher

copilot = folders.module(root=paths.teachers_dir(), name="copilot", file="teacher.py", kind="teacher")

COPILOT_EVENTS = [
    {"type": "session.info", "timestamp": "2026-10-28T23:00:00.000Z", "data": {"message": "start"}},
    {"type": "user.message", "timestamp": "2026-10-28T23:00:00.000Z", "data": {"content": "Make one variant."}},
    {"type": "assistant.turn_start", "timestamp": "2026-10-28T23:00:01.000Z", "data": {}},
    {"type": "assistant.reasoning", "timestamp": "2026-10-28T23:00:02.000Z", "data": {"content": "Read the overview."}},
    {
        "type": "assistant.message",
        "timestamp": "2026-10-28T23:00:03.000Z",
        "data": {
            "content": "",
            "toolRequests": [
                {"toolCallId": "c1", "name": "container-bash", "arguments": {"command": "cat /work/overview.md"}},
                {
                    "toolCallId": "c2",
                    "name": "container-edit_file",
                    "arguments": {"path": "/out/a.md", "old_text": "one", "new_text": "two"},
                },
            ],
        },
    },
    {"type": "model.call_finished", "timestamp": "2026-10-28T23:00:03.000Z", "data": {}},
    {"type": "tool.execution_start", "timestamp": "2026-10-28T23:00:03.000Z", "data": {"toolCallId": "c1"}},
    {
        "type": "tool.execution_complete",
        "timestamp": "2026-10-28T23:00:05.000Z",
        "data": {"toolCallId": "c1", "success": True, "result": {"content": "exit code 0\n# The parent run"}},
    },
    {
        "type": "tool.execution_complete",
        "timestamp": "2026-10-28T23:00:05.000Z",
        "data": {"toolCallId": "c2", "success": False, "error": {"message": "old_text occurs 0 times"}},
    },
    {"type": "session.compaction_start", "timestamp": "2026-10-28T23:00:06.000Z", "data": {}},
    {"type": "session.compaction_complete", "timestamp": "2026-10-28T23:00:06.500Z", "data": {}},
    {
        "type": "assistant.message",
        "timestamp": "2026-10-28T23:00:07.000Z",
        "data": {"content": "Done.", "toolRequests": []},
    },
    {"type": "model.call_finished", "timestamp": "2026-10-28T23:00:07.000Z", "data": {}},
    {
        "type": "session.usage_checkpoint",
        "timestamp": "2026-10-28T23:00:07.000Z",
        "data": {"totalNanoAiu": 2_500_000_000, "totalPremiumRequests": 15},
    },
]


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def test_the_teacher_comes_from_models_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    (tmp_path / "models.toml").write_text(
        '[teachers.opus]\nagent = "copilot"\nmodel = "claude-opus-5.5"\neffort = "max"\n'
    )
    assert models.teacher(name="opus") == core.TeacherSpec(
        name="opus", agent="copilot", model="claude-opus-5.5", effort="max"
    )
    with pytest.raises(KeyError, match="unknown teacher"):
        models.teacher(name="sonnet")
    assert teacher.load(name="copilot").name == "copilot"


def test_copilot_runs_locked_to_its_container_with_the_users_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".copilot").mkdir()
    login = {"loggedInUsers": [{"login": "me"}], "lastLoggedInUser": {"login": "me"}, "trustedFolders": ["/"]}
    (tmp_path / ".copilot" / "config.json").write_text("// comment\n" + json.dumps(login))
    monkeypatch.setattr(copilot, "HOME", tmp_path / "teacher-home")
    seen: dict = {}

    class FakePopen:
        def __init__(
            self, command: list[str], stdout: IO[bytes], stderr: IO[bytes], env: dict[str, str], start_new_session: bool
        ) -> None:
            seen.update(command=command, env=env, own_group=start_new_session)
            stdout.write(b'{"type": "user.message", "data": {"content": "hi"}}\n')
            stdout.write(b'{"type": "tool.execution_start", "data": {"toolCallId": "c1"}}\n' * 2)
            out_dir = Path(command[command.index("--usage-output-file") + 1]).parent
            usage = {"totalNanoAiu": 12_300_000_000, "totalPremiumRequestCost": 15}
            usage["modelMetrics"] = {"claude-opus-5.5": {"usage": {"outputTokens": 900}}}
            (out_dir / "usage.json").write_text(json.dumps(usage))
            self.pid = 1

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr(copilot.subprocess, "Popen", FakePopen)
    out = tmp_path / "session"
    result = copilot.TEACHER.run(
        prompt="Make one variant.",
        model="claude-opus-5.5",
        effort="max",
        container="rsi-teacher-x",
        out_dir=out,
        timeout_s=60,
    )
    command = seen["command"]
    tools = command[command.index("--available-tools") + 1 : command.index("--available-tools") + 5]
    assert tools == ["container-bash", "container-read_file", "container-write_file", "container-edit_file"]
    for flag in ("--disable-builtin-mcps", "--no-custom-instructions", "--no-ask-user", "--sync-session=off"):
        assert flag in command
    assert command[command.index("--reasoning-effort") + 1] == "max"
    assert command[command.index("-C") + 1] == str(tmp_path / "teacher-home" / "cwd")
    assert seen["env"]["COPILOT_HOME"] == str(tmp_path / "teacher-home") and seen["own_group"]
    server = json.loads((out / "mcp.json").read_text())["mcpServers"]["container"]
    assert server["args"] == [copilot.container_mcp.__file__, "rsi-teacher-x"] and server["timeout"] > 3_600_000
    home = json.loads((tmp_path / "teacher-home" / "config.json").read_text())
    assert home == {**{k: login[k] for k in ("loggedInUsers", "lastLoggedInUser")}, "trustedFolders": [command[2]]}
    assert (result.status, result.output_tokens, result.cost) == ("ok", 900, "12.3 AI credits, 15 premium requests")
    assert result.tool_calls == 2


def test_copilot_needs_a_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(copilot, "HOME", tmp_path / "teacher-home")
    with pytest.raises(RuntimeError, match="/login"):
        copilot._home()


def test_a_session_keeps_copilots_own_record_of_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(copilot, "HOME", tmp_path / "teacher-home")
    older = tmp_path / "teacher-home" / "session-state" / "older"
    older.mkdir(parents=True)
    os.utime(older, (1_000, 1_000))
    (tmp_path / "teacher-home" / "session-state" / "this-one").mkdir()
    out = tmp_path / "session-1"
    out.mkdir()
    copilot._keep_session_state(out_dir=out, since=2_000)
    assert [p.name for p in (out / "copilot-session").iterdir()] == ["this-one"]
    assert older.is_dir()


def test_a_stopped_session_still_has_its_transcript_and_cost(tmp_path: Path) -> None:
    _write_events(path=tmp_path / "events.jsonl", events=COPILOT_EVENTS)
    with (tmp_path / "events.jsonl").open("a") as fh:
        fh.write('{"type": "assistant.mess')
    text = copilot.TEACHER.transcript(session_dir=tmp_path)
    assert text.startswith("## task\n\nMake one variant.\n")
    assert "### thinking\nRead the overview." in text and "### container-bash" in text and "→ error" in text
    assert "old_text occurs 0 times" in text and "(the conversation was compacted" in text
    calls, output, cost = copilot._usage(out_dir=tmp_path)
    assert (calls, output, cost) == (2, 0, "2.5 AI credits, 15 premium requests")


def test_a_teacher_session_reads_as_steps_for_the_viewer(tmp_path: Path) -> None:
    _write_events(path=tmp_path / "events.jsonl", events=COPILOT_EVENTS)
    view = copilot.TEACHER.trajectory(session_dir=tmp_path)
    assert view.user == "Make one variant."
    steps = [i for i in view.items if i.kind == "step"]
    bash, edit = steps[0].calls
    assert steps[0].thinking == "Read the overview." and steps[0].t == 3000
    assert (bash.name, bash.label, bash.ok, bash.ms) == ("bash", "cat /work/overview.md", True, 2000)
    assert edit.name == "edit" and edit.args == {"edits": [{"old": "one", "new": "two"}]} and not edit.ok
    assert any(i.kind == "note" for i in view.items) and steps[1].text == "Done."


def test_the_mcp_server_speaks_mcp_and_acts_only_in_the_container(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = folders.module(root=paths.teachers_dir(), name="copilot", file="container_mcp.py")
    files = {"/out/a.md": b"one two one"}
    commands: list[list[str]] = []

    def fake_in_container(container: str, args: list[str], stdin: bytes | None, timeout_s: int) -> tuple[int, bytes]:
        assert container == "rsi-teacher-x"
        commands.append(args)
        if args[0] == "cat":
            return (0, files[args[-1]]) if args[-1] in files else (1, b"cat: no such file")
        if args[0] == "sh":
            files[args[-1]] = stdin or b""
            return 0, b""
        return 0, b"hello\n"

    monkeypatch.setattr(server, "in_container", fake_in_container)

    def call(method: str, params: dict) -> dict:
        server.handle(
            container="rsi-teacher-x", message={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        )
        return json.loads(capsys.readouterr().out)

    def tool(name: str, arguments: dict) -> dict:
        return call(method="tools/call", params={"name": name, "arguments": arguments})

    hello = call(method="initialize", params={"protocolVersion": "2025-06-18"})
    assert hello["result"]["protocolVersion"] == "2025-06-18"
    server.handle(container="rsi-teacher-x", message={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert capsys.readouterr().out == ""
    listed = call(method="tools/list", params={})
    assert [t["name"] for t in listed["result"]["tools"]] == ["bash", "read_file", "write_file", "edit_file"]
    ran = tool(name="bash", arguments={"command": "echo hello"})
    assert ran["result"] == {"content": [{"type": "text", "text": "exit code 0\nhello\n"}], "isError": False}
    assert commands[-1] == ["bash", "-lc", "cd /work && echo hello"]
    refused = tool(name="edit_file", arguments={"path": "/out/a.md", "old_text": "one", "new_text": "1"})
    assert refused["result"]["isError"] and "occurs 2 times" in refused["result"]["content"][0]["text"]
    tool(name="edit_file", arguments={"path": "/out/a.md", "old_text": "two", "new_text": "2"})
    assert files["/out/a.md"] == b"one 2 one"
    read = tool(name="read_file", arguments={"path": "/out/a.md"})
    assert read["result"]["content"][0]["text"] == "1. one 2 one"
    files["/out/long.md"] = "\n".join("x" * 100 for _ in range(400)).encode()
    long = tool(name="read_file", arguments={"path": "/out/long.md"})["result"]["content"][0]["text"]
    assert len(long) < server.OUTPUT_CHARS + 100 and long.endswith("; the file has 400 lines)")
    after = int(long.rsplit("start_line=", 1)[1].split(";")[0])
    assert long.splitlines()[-2].startswith(f"{after - 1}. ")
    assert tool(name="web_fetch", arguments={})["error"]["message"] == "unknown tool"
    assert (
        len(server.clip(text="x" * (server.OUTPUT_CHARS + 500), limit=server.OUTPUT_CHARS)) < server.OUTPUT_CHARS + 100
    )


def test_the_mcp_server_answers_every_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    server = folders.module(root=paths.teachers_dir(), name="copilot", file="container_mcp.py")

    def no_docker(container: str, args: list[str], stdin: bytes | None, timeout_s: int) -> tuple[int, bytes]:
        raise FileNotFoundError("docker")

    monkeypatch.setattr(server, "in_container", no_docker)

    def call(method: str, params: object) -> dict:
        server.handle(
            container="rsi-teacher-x", message={"jsonrpc": "2.0", "id": 7, "method": method, "params": params}
        )
        return json.loads(capsys.readouterr().out)

    failed = call(method="tools/call", params={"name": "bash", "arguments": {"command": "ls"}})
    assert failed["result"]["isError"] and failed["result"]["content"][0]["text"] == "FileNotFoundError: docker"
    odd = call(method="tools/call", params={"name": "write_file", "arguments": {"path": "/out/a", "content": 5}})
    assert odd["result"]["isError"] and odd["result"]["content"][0]["text"].startswith("AttributeError")
    broken = call(method="initialize", params=["not", "an", "object"])
    assert broken["id"] == 7 and broken["error"]["code"] == -32603
    monkeypatch.setattr(server.sys, "argv", ["container_mcp.py", "rsi-teacher-x"])
    monkeypatch.setattr(server.sys, "stdin", io.StringIO('{"jsonrpc": "2.0", "id": 1, "meth\n[1, 2]\n\n'))
    server.main()
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["id"], r["error"]["code"]) for r in replies] == [(None, -32700), (None, -32600)]


def _try_variant(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """try_variant.py, which imports check_variant from its own folder"""
    monkeypatch.syspath_prepend(paths.harnesses_dir() / "pi")  # python3 does this for a script it runs
    return folders.module(root=paths.harnesses_dir(), name="pi", file="try_variant.py")


def test_try_variant_sums_up_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    try_variant = _try_variant(monkeypatch=monkeypatch)
    records = [
        {"type": "message_update", "assistantMessageEvent": {}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "toolUse",
                "usage": {"output": 120},
                "content": [
                    {"type": "thinking", "thinking": "Run the tests first."},
                    {"type": "toolCall", "name": "bash", "arguments": {"command": "pytest -q"}},
                ],
            },
        },
        {"type": "entry_appended", "entry": {"type": "custom_message", "customType": "guard", "content": "Check it."}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "usage": {"output": 9},
                "content": [{"type": "text", "text": "Fixed."}],
            },
        },
    ]
    events = tmp_path / "events.jsonl"
    events.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert try_variant.summary(events=events) == [
        "[1] 120 tokens, stop toolUse",
        '    bash: {"command": "pytest -q"}',
        "    harness message (guard): Check it.",
        "[2] 9 tokens, stop stop: Fixed.",
    ]


def _response(stop: str, text: str = "", call: str = "") -> dict:
    content = [{"type": "text", "text": text}] if text else []
    content += [{"type": "toolCall", "name": call, "arguments": {}}] if call else []
    return {"type": "message_end", "message": {"role": "assistant", "stopReason": stop, "content": content}}


def _events(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))
    return path


def test_the_question_check_says_how_the_package_got_in_the_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try_variant = _try_variant(monkeypatch=monkeypatch)

    def problems(records: list[dict], diff: str = "", finished: bool = True, failed: str = "") -> list[str]:
        events = _events(path=tmp_path / "events.jsonl", records=records)
        tried = try_variant.Tried(events=events, repo=tmp_path, diff=diff, finished=finished)
        return try_variant.question_problems(tried=tried, timeout=540, failed=failed)

    read = _response(stop="toolUse", call="read")
    answer = _response(stop="stop", text="It still works on its expiry date.")
    assert problems(records=[read, answer]) == []
    assert problems(records=[read, answer, read, answer]) == ["the agent answered, and the package made it go on"]
    assert problems(records=[read, answer], finished=False) == ["the agent answered, but the run hadn't ended 540 s in"]
    assert problems(records=[read, _response(stop="stop")]) == ["the agent never answered"]
    assert problems(records=[read], finished=False) == ["the agent never answered in 540 s"]
    assert problems(records=[], failed="pi got no response from the model") == [
        "pi got no response from the model, though the model answers a plain request"
    ]
    # an answer the package made the agent go on from still counts, even if the next request failed
    failed = {"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": "x"}}
    assert problems(records=[read, answer, failed], failed="pi's last request failed (x)") == [
        "the agent answered, and the package made it go on"
    ]
    diff = "diff --git a/src/cart.js b/src/cart.js\n--- a/src/cart.js\n+++ b/src/cart.js\n"
    assert problems(records=[read, answer], diff=diff) == [
        "the run changed src/cart.js, though the question needed no change"
    ]
    # a stray edit a hook then rightly acts on shows both, and the second try tells them apart
    assert problems(records=[read, answer, read, answer], diff=diff) == [
        "the agent answered, and the package made it go on",
        "the run changed src/cart.js, though the question needed no change",
    ]
    # a response that quotes an update event is still read
    quoted = _response(stop="stop", text='It prints {"type":"message_update"} lines.')
    assert problems(records=[read, quoted]) == []


def test_a_run_without_an_answer_from_the_model_says_nothing_about_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try_variant = _try_variant(monkeypatch=monkeypatch)
    events = tmp_path / "events.jsonl"
    read = _response(stop="toolUse", call="read")
    failed = {"type": "message_end", "message": {"role": "assistant", "stopReason": "error", "errorMessage": "refused"}}
    assert try_variant.unavailable(events=_events(path=events, records=[])) == "pi got no response from the model"
    assert try_variant.unavailable(events=_events(path=events, records=[read, failed])) == (
        "pi's last request failed (refused)"
    )
    answer = _response(stop="stop", text="Yes.")
    assert try_variant.unavailable(events=_events(path=events, records=[read, answer])) == ""


@pytest.mark.parametrize(
    ("outcomes", "code", "asked"),
    [
        (["clean"], 0, 1),
        (["pushy", "clean"], 0, 2),
        (["pushy", "pushy"], 1, 2),
        (["pushy", "edited"], 1, 2),
        (["edited", "clean"], 0, 2),
        (["edited", "edited"], 1, 2),
        (["down", "clean"], 0, 2),
        (["down", "down"], 2, 2),
        (["down", "edited"], 2, 2),
        (["edited", "down"], 2, 2),
        (["down", "pushy"], 2, 2),
        # pi got no answer while the model answered a plain request, so the package may have broken pi's requests
        (["broke", "broke"], 1, 2),
        (["broke", "clean"], 0, 2),
        (["broke", "down"], 2, 2),
        (["broke", "pushy"], 1, 2),
    ],
)
def test_the_question_is_asked_again_only_while_the_outcome_could_be_chance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcomes: list[str],
    code: int,
    asked: int,
) -> None:
    try_variant = _try_variant(monkeypatch=monkeypatch)
    pi = folders.module(root=paths.harnesses_dir(), name="pi", file="harness.py", kind="harness")
    read = _response(stop="toolUse", call="read")
    answer = _response(stop="stop", text="Yes, through the end of that day.")
    records = {
        "clean": [read, answer],
        "pushy": [read, answer, answer],
        "edited": [read, answer],
        "down": [],
        "broke": [],
    }
    ran: list[str] = []

    def run(package: Path, repo: Path, message: str, timeout: int) -> object:
        outcome = outcomes[len(ran)]
        ran.append(outcome)
        events = _events(path=tmp_path / f"events-{len(ran)}.jsonl", records=records[outcome])
        diff = "diff --git a/notes.md b/notes.md\n" if outcome == "edited" else ""
        return try_variant.Tried(events=events, repo=tmp_path, diff=diff, finished=True)

    monkeypatch.setattr(try_variant, "run", run)
    monkeypatch.setattr(try_variant, "model_answers", lambda: ran[-1] == "broke")
    monkeypatch.setattr(try_variant, "QUESTION", pi.PRACTICE_QUESTION)
    assert try_variant.question(package=tmp_path, timeout=540, tries=2) == code
    assert len(ran) == asked
    printed = capsys.readouterr().out
    assert ("\nunavailable: " in printed) == (code == 2)
    if outcomes == ["broke", "broke"]:
        assert "problem: pi got no response from the model, though the model answers a plain request" in printed


def test_try_variant_stops_pi_when_it_is_stopped(tmp_path: Path) -> None:
    # the bash tool's timeout sends SIGTERM to try-variant's process group, which pi isn't in
    pi = folders.module(root=paths.harnesses_dir(), name="pi", file="harness.py", kind="harness")
    fake = tmp_path / "bin" / "pi"
    fake.parent.mkdir()
    fake.write_text(f"#!/bin/sh\necho $$ > '{tmp_path}/pi.pid'\nsleep 60 &\necho $! > '{tmp_path}/child.pid'\nwait\n")
    fake.chmod(0o755)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "models.json").write_text("{}")
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "README.md").write_text("A small repo.\n")
    (tmp_path / "task.md").write_text("What is this repo?\n")
    argv = ["try-variant", str(pi.HARNESS.package_dir), str(tmp_path / "repo"), str(tmp_path / "task.md")]
    script = (
        "import sys\nfrom pathlib import Path\nimport try_variant\n"
        f"try_variant.MODELS = Path({str(tmp_path / 'models')!r})\n"
        f"try_variant.TRIES = Path({str(tmp_path / 'tries')!r})\n"
        f"sys.exit(try_variant.main(argv={argv!r}))\n"
    )
    # PYTHONPATH stands in for the script's own folder, which python3 puts on the path in the container
    env = {**os.environ, "PATH": f"{fake.parent}{os.pathsep}{os.environ['PATH']}", "PYTHONPATH": str(pi.PI_DIR)}
    proc = subprocess.Popen([sys.executable, "-c", script], env=env, stderr=subprocess.DEVNULL)
    started = time.monotonic()
    while not (tmp_path / "child.pid").is_file() or not (tmp_path / "child.pid").read_text().strip():
        assert proc.poll() is None and time.monotonic() - started < 30
        time.sleep(0.1)
    pids = [int((tmp_path / name).read_text()) for name in ("pi.pid", "child.pid")]
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=15)

    def gone(pid: int) -> bool:
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.1)
        return False

    assert all(gone(pid=pid) for pid in pids)


def test_every_question_is_about_a_small_repo_that_needs_no_change() -> None:
    pi = folders.module(root=paths.harnesses_dir(), name="pi", file="harness.py", kind="harness")
    questions = sorted(d.name for d in pi.QUESTIONS_DIR.iterdir() if d.is_dir())
    assert questions == ["javascript", "python"] and pi.PRACTICE_QUESTION.name == "javascript"
    for name in questions:
        question = pi.QUESTIONS_DIR / name
        assert (question / "question.md").read_text().strip().endswith("?")
        assert (question / "repo" / "README.md").is_file()
    # npm install leaves a committed lockfile as it is, so a reflexive install changes nothing
    assert (pi.PRACTICE_QUESTION / "repo" / "package-lock.json").is_file()


def test_the_teacher_container_installs_both_helpers() -> None:
    pi = folders.module(root=paths.harnesses_dir(), name="pi", file="harness.py", kind="harness")
    assert "check_variant.py --terms /opt/rsi/terms.txt --manifest /opt/harness/package.json" in pi.CHECK_COMMAND
    assert "try_variant.py" in pi.TRY_COMMAND
    assert subprocess.run(["sh", "-n", "-c", pi.TRY_COMMAND], check=False).returncode == 0


def test_mcp_server_runs_with_the_standard_library_alone() -> None:
    # Copilot runs it with the host's python3, without rsi installed
    command = [sys.executable, "-S", copilot.container_mcp.__file__, "rsi-teacher-x"]
    done = subprocess.run(command, input="", capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
