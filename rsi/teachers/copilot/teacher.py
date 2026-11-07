import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rsi import folders, paths, teacher, views

COPILOT = os.environ.get("RSI_COPILOT_BIN", "copilot")
container_mcp = folders.module(root=paths.teachers_dir(), name="copilot", file="container_mcp.py", kind="MCP server")
# only the container's tools, since Copilot's own reach the host and the internet
TOOLS = ("container-bash", "container-read_file", "container-write_file", "container-edit_file")
MCP_TIMEOUT_MS = 3_700_000  # a container command may take an hour (Copilot's default is 5 minutes)
# a separate Copilot home, so the user's history, skills, plugins and MCP servers stay out (only the login is borrowed)
HOME = Path(os.environ.get("RSI_TEACHER_HOME", str(Path.home() / ".cache" / "rsi" / "copilot")))
CLIP_CHARS = 6_000  # per tool call and result in the transcript
# what kind of tool each container tool is, for the viewer (Copilot calls them <server>-<tool>)
_KINDS = {"bash": "bash", "read_file": "read", "write_file": "write", "edit_file": "edit"}


class CopilotTeacher:
    name = "copilot"

    def describe(self, *, model: str, effort: str) -> str:
        version = subprocess.run([COPILOT, "--version"], capture_output=True, text=True, check=True).stdout
        return f"{version.splitlines()[0].strip()}, {model} at {effort} reasoning effort"

    def run(
        self, *, prompt: str, model: str, effort: str, container: str, out_dir: Path, timeout_s: int
    ) -> teacher.TeacherRun:
        out_dir.mkdir(parents=True, exist_ok=True)
        cwd = _home()
        server = {
            "type": "local",
            "command": sys.executable,
            "args": [container_mcp.__file__, container],
            "tools": ["*"],
        }
        config = out_dir / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {"container": {**server, "timeout": MCP_TIMEOUT_MS}}}, indent=2))
        command = [
            COPILOT,
            "-C",
            str(cwd),
            "-p",
            prompt,
            "--model",
            model,
            "--reasoning-effort",
            effort,
            "--allow-all-tools",
            "--available-tools",
            *TOOLS,
            "--additional-mcp-config",
            f"@{config}",
            "--disable-builtin-mcps",
            "--no-custom-instructions",
            "--no-ask-user",
            "--no-auto-update",
            "--disable-client-telemetry",
            "--sync-session=off",
            "--output-format",
            "json",
            "--usage-output-file",
            str(out_dir / "usage.json"),
            "--log-dir",
            str(out_dir / "copilot-logs"),
        ]
        env = {**os.environ, "COPILOT_HOME": str(HOME), "NO_COLOR": "1"}
        start = time.time()
        status, error = "ok", None
        with open(out_dir / "events.jsonl", "wb") as events, open(out_dir / "copilot.err", "wb") as err:
            process = subprocess.Popen(command, stdout=events, stderr=err, env=env, start_new_session=True)
            try:
                code = process.wait(timeout=timeout_s)
                if code != 0:
                    status, error = "error", f"copilot exited with {code}"
            except subprocess.TimeoutExpired:
                status, error = "timeout", f"stopped after {timeout_s} s"
                _stop(process=process)
            except BaseException:
                _stop(process=process)  # it's in its own process group, so our interrupt doesn't reach it
                raise
        if status == "error":
            tail = (out_dir / "copilot.err").read_text(errors="replace").strip()[-1_500:]
            error = f"{error}: {tail}" if tail else error
        _keep_session_state(out_dir=out_dir, since=start)
        calls, output, cost = _usage(out_dir=out_dir)
        tools = [e for e in _events(path=out_dir / "events.jsonl") if e.get("type") == "tool.execution_start"]
        return teacher.TeacherRun(
            status=status,
            seconds=round(time.time() - start, 1),
            error=error,
            model_calls=calls,
            tool_calls=len(tools),
            output_tokens=output,
            cost=cost,
        )

    def transcript(self, *, session_dir: Path) -> str:
        lines: list[str] = []
        for event in _events(path=session_dir / "events.jsonl"):
            kind, data = event.get("type"), event.get("data") or {}
            if kind == "user.message":
                lines += ["## task", "", str(data.get("content", "")).strip(), ""]
            elif kind == "assistant.reasoning" and data.get("content"):
                lines += ["### thinking", str(data["content"]).strip(), ""]
            elif kind == "assistant.message":
                if str(data.get("content") or "").strip():
                    lines += ["### says", str(data["content"]).strip(), ""]
                for call in data.get("toolRequests") or []:
                    arguments = json.dumps(call.get("arguments"), indent=1, ensure_ascii=False)
                    lines += [
                        f"### {call.get('name', '?')}",
                        "```",
                        container_mcp.clip(text=arguments, limit=CLIP_CHARS),
                        "```",
                    ]
            elif kind == "tool.execution_complete":
                result = data.get("result") or {}
                text = str(result.get("content") or (data.get("error") or {}).get("message") or "")
                outcome = "ok" if data.get("success") else "error"
                lines += [f"→ {outcome}", "```", container_mcp.clip(text=text, limit=CLIP_CHARS), "```", ""]
            elif kind == "session.compaction_complete":
                lines += ["(the conversation was compacted to fit the context window)", ""]
        return "\n".join(lines) + "\n"

    def trajectory(self, *, session_dir: Path) -> views.Trajectory:
        view = views.Trajectory()
        start: float | None = None
        calls: dict[str, views.Call] = {}
        started: dict[str, float] = {}
        thinking: list[str] = []
        for event in _events(path=session_dir / "events.jsonl"):
            kind, data = event.get("type"), event.get("data") or {}
            t = _epoch_ms(stamp=event.get("timestamp"))
            start = start or t
            if kind == "user.message" and not view.user:
                view.user = str(data.get("content") or "")
            elif kind == "assistant.turn_start":
                view.inflight_at = t
            elif kind == "assistant.reasoning":
                thinking.append(str(data.get("content") or ""))
            elif kind == "assistant.message":
                view.inflight_at = None
                requests = data.get("toolRequests") or []
                step = views.step(
                    t=t - start,
                    ms=None,
                    thinking="\n".join(thinking),
                    text=str(data.get("content") or ""),
                    calls=[_call(request=request) for request in requests],
                    stop="toolUse" if requests else "stop",
                    out_tokens=None,
                    error=None,
                )
                thinking = []
                calls.update({call.id: call for call in step.calls})
                view.items.append(step)
            elif kind == "tool.execution_start":
                started[str(data.get("toolCallId", ""))] = t
            elif kind == "tool.execution_complete":
                call = calls.get(str(data.get("toolCallId", "")))
                if call is not None:
                    error = (data.get("error") or {}).get("message")
                    out = str((data.get("result") or {}).get("content") or error or "")
                    since = started.get(call.id)
                    call.ended(
                        out=out, ok=bool(data.get("success")), ms=round(t - since) if since is not None else None
                    )
            elif kind == "session.compaction_start":
                view.items.append(views.Note(text="context compacted"))
        return view


def _call(request: dict) -> views.Call:
    name = str(request.get("name") or "?")
    kind = _KINDS.get(name.partition("-")[2], name)
    args = request.get("arguments")
    if kind == "edit" and isinstance(args, dict):
        args = {"path": args.get("path"), "edits": [{"old": args.get("old_text", ""), "new": args.get("new_text", "")}]}
    return views.call(call_id=str(request.get("toolCallId") or ""), name=kind, args=args)


def _epoch_ms(stamp: object) -> float:
    try:
        return datetime.fromisoformat(str(stamp)).timestamp() * 1000
    except ValueError:
        return time.time() * 1000


def _keep_session_state(out_dir: Path, since: float) -> None:
    """Move Copilot's own session record (which has compactions) into out_dir"""
    for folder in (HOME / "session-state").glob("*"):
        if folder.is_dir() and folder.stat().st_mtime >= since:
            (out_dir / "copilot-session").mkdir(exist_ok=True)
            shutil.move(str(folder), str(out_dir / "copilot-session" / folder.name))


def _home() -> Path:
    """Copy the user's Copilot login into the teacher's home, so a new /login carries over"""
    cwd = HOME / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    user_config = Path.home() / ".copilot" / "config.json"
    text = user_config.read_text() if user_config.is_file() else "{}"
    settings = json.loads("\n".join(line for line in text.splitlines() if not line.strip().startswith("//")))
    login = {key: settings[key] for key in ("loggedInUsers", "lastLoggedInUser") if key in settings}
    if not login:
        raise RuntimeError("Copilot CLI isn't logged in: run `copilot`, then /login")
    (HOME / "config.json").write_text(json.dumps({**login, "trustedFolders": [str(cwd)]}, indent=2))
    return cwd


def _stop(process: subprocess.Popen) -> None:
    """Interrupt copilot so it writes its usage file, then kill what's left"""
    for sig, wait_s in ((signal.SIGINT, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def _events(path: Path) -> list[dict]:
    """The session's events, without the streamed pieces each whole message repeats"""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if '_delta"' in line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a stopped session can end on a half-written line
    return out


def _usage(out_dir: Path) -> tuple[int, int, str]:
    """Model calls, output tokens and cost, from the usage file or the last checkpoint"""
    events = _events(path=out_dir / "events.jsonl")
    calls = sum(1 for e in events if e.get("type") == "model.call_finished")
    usage_file = out_dir / "usage.json"
    if usage_file.is_file():
        usage = json.loads(usage_file.read_text())
        output = sum(m.get("usage", {}).get("outputTokens", 0) for m in usage.get("modelMetrics", {}).values())
        return calls, output, _cost(nano_aiu=usage.get("totalNanoAiu", 0), premium=usage.get("totalPremiumRequestCost"))
    checkpoints = [e.get("data") or {} for e in events if e.get("type") == "session.usage_checkpoint"]
    last = checkpoints[-1] if checkpoints else {}
    return calls, 0, _cost(nano_aiu=last.get("totalNanoAiu", 0), premium=last.get("totalPremiumRequests"))


def _cost(nano_aiu: int, premium: float | None) -> str:
    credits = f"{nano_aiu / 1e9:.1f} AI credits"
    return f"{credits}, {premium:g} premium requests" if premium is not None else credits


TEACHER = CopilotTeacher()
