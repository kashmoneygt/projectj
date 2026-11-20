import argparse
import dataclasses
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import check_variant

MODELS = Path("/opt/rsi/models")
TRIES = Path("/out/tries")
QUESTION = Path("/opt/rsi/question")  # a small repo and a question about it that needs no change
MODEL_REF = (MODELS / "ref.txt").read_text().strip() if (MODELS / "ref.txt").is_file() else ""
SHOWN = 160
# skip pi's streaming updates without parsing them, since a run writes thousands
_UPDATES = ('{"type":"message_update"', '{"type":"tool_execution_update"')
# git with the repo's hooks and fsmonitor off, since the agent could set them (a task run does the same)
GIT = ("git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null")


@dataclasses.dataclass(frozen=True)
class Tried:
    events: Path
    repo: Path  # copy of the repo the agent worked in
    diff: str
    finished: bool  # pi ended on its own, before the timeout


def stage(package: Path, repo: Path, copy: Path, work: Path) -> tuple[str, Path, Path]:
    """Copy and commit the repo, then stage the package"""
    shutil.copytree(repo, copy, symlinks=True)
    if not (copy / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=copy, check=True)
    # keep Python caches out of the diff, since real repos usually gitignore them
    (copy / ".git" / "info").mkdir(parents=True, exist_ok=True)
    with open(copy / ".git" / "info" / "exclude", "a") as exclude:
        exclude.write("\n__pycache__/\n*.pyc\n")
    git = ["git", "-c", "user.email=try@variant", "-c", "user.name=try"]
    subprocess.run([*git, "add", "-A"], cwd=copy, check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "start"], cwd=copy, check=True)
    start = subprocess.run(["git", "rev-parse", "HEAD"], cwd=copy, capture_output=True, text=True, check=True)
    staged = check_variant.stage_runtime_package(package=package, dest=work / "package")
    agent = work / "agent"
    check_variant.copy_agent_resources(package=package, agent_dir=agent)
    shutil.copy2(MODELS / "models.json", agent / "models.json")
    return start.stdout.strip(), staged, agent


def collected_diff(repo: Path, start: str) -> str:
    """Diff of every change since start, new files included, taken the same way as in a task run"""
    staged = subprocess.run([*GIT, "add", "-A"], cwd=repo, check=False).returncode == 0
    diff = [*GIT, "diff", *(["--cached"] if staged else []), "--binary", "--no-ext-diff", "--no-textconv", start]
    return subprocess.run(diff, cwd=repo, capture_output=True, text=True, check=False).stdout


def _records(events: Path) -> Iterator[dict]:
    for raw in events.read_text(errors="replace").splitlines():
        if raw.startswith(_UPDATES):
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if isinstance(record, dict):
            yield record


def _parts(message: dict) -> list[dict]:
    return [p for p in message.get("content") or [] if isinstance(p, dict)]


def _responses(events: Path) -> list[dict]:
    """The model's responses, in order"""
    found: list[dict] = []
    for record in _records(events=events):
        message = record.get("message") or {}
        if record.get("type") == "message_end" and message.get("role") == "assistant":
            found.append(message)
    return found


def summary(events: Path) -> list[str]:
    lines: list[str] = []
    turn = 0
    for record in _records(events=events):
        kind = record.get("type")
        message = record.get("message") or {}
        entry = record.get("entry") or {}
        if kind == "message_end" and message.get("role") == "assistant":
            turn += 1
            parts = _parts(message=message)
            calls = [p for p in parts if p.get("type") == "toolCall"]
            words = " ".join(str(p.get("text") or p.get("thinking") or "") for p in parts).split()
            head = f"[{turn}] {message.get('usage', {}).get('output', 0)} tokens, stop {message.get('stopReason')}"
            lines.append(head + (f": {' '.join(words)[:SHOWN]}" if not calls and words else ""))
            for call in calls:
                lines.append(f"    {call.get('name')}: {json.dumps(call.get('arguments'))[:SHOWN]}")
        elif kind == "message_end" and message.get("role") == "custom":
            lines.append(_added(message=message))
        elif kind == "entry_appended" and entry.get("type") == "custom_message":
            lines.append(_added(message=entry))
        elif kind == "entry_appended" and entry.get("type") == "custom" and entry.get("customType") == "acted":
            lines.append(f"    acted: {json.dumps(entry.get('data'))[:SHOWN]}")
        elif kind == "auto_retry_start":
            lines.append(f"    request retried: {str(record.get('errorMessage'))[:200]}")
    return lines


def _added(message: dict) -> str:
    """A message the package added for the model, via pi.sendMessage or a turn_end or agent_before_settle result"""
    content = message.get("content")
    text = content if isinstance(content, str) else " ".join(str(p.get("text") or "") for p in _parts(message=message))
    return f"    harness message ({message.get('customType')}): {text[:400]}"


def answers(events: Path) -> list[bool]:
    """For each model response, whether it was an answer, i.e. it stopped on its own with text and no tool call"""
    found: list[bool] = []
    for message in _responses(events=events):
        parts = _parts(message=message)
        text = any(p.get("type") == "text" and str(p.get("text") or "").strip() for p in parts)
        calls = any(p.get("type") == "toolCall" for p in parts)
        found.append(message.get("stopReason") == "stop" and text and not calls)
    return found


def unavailable(events: Path) -> str:
    """Why the run can't judge the package (no response, or the last request failed), or "" if it can"""
    responses = _responses(events=events)
    if not responses:
        return "pi got no response from the model"
    if responses[-1].get("stopReason") == "error":
        return f"pi's last request failed ({str(responses[-1].get('errorMessage') or 'no message')[:200]})"
    return ""


def model_answers() -> bool:
    """Check if the model answers a plain 1-token request, sent to the same endpoint pi uses"""
    try:
        provider = json.loads((MODELS / "models.json").read_text())["providers"]["local"]
        body = {"model": provider["models"][0]["id"], "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
        request = urllib.request.Request(
            f"{provider['baseUrl']}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {provider['apiKey']}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return bool(json.loads(response.read()).get("choices"))
    except (OSError, ValueError, KeyError, IndexError, http.client.HTTPException):
        return False


def question_problems(tried: Tried, timeout: int, failed: str = "") -> list[str]:
    """Problems the package caused on a question that needed no change"""
    found = answers(events=tried.events)
    problems: list[str] = []
    if True not in found:
        never = "the agent never answered" + ("" if tried.finished else f" in {timeout} s")
        # failed is why pi's requests went unanswered even though a plain request worked
        problems.append(f"{failed}, though the model answers a plain request" if failed else never)
    elif found.index(True) < len(found) - 1:
        # in a one-shot pi run, only an extension can make the agent keep going after it answers
        problems.append("the agent answered, and the package made it go on")
    elif not tried.finished:
        problems.append(f"the agent answered, but the run hadn't ended {timeout} s in")
    changed = [line.split(" b/", 1)[-1] for line in tried.diff.splitlines() if line.startswith("diff --git ")]
    if changed:
        problems.append(f"the run changed {', '.join(changed)}, though the question needed no change")
    return problems


def run(package: Path, repo: Path, message: str, timeout: int) -> Tried:
    """Run pi with the package on a copy of repo, like a task run"""
    TRIES.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="try-", dir=TRIES))
    copy = Path(tempfile.mkdtemp(prefix="try-repo-", dir="/tmp")) / "repo"
    start, staged, agent = stage(package=package, repo=repo, copy=copy, work=work)
    events = work / "events.jsonl"
    command = ["pi", "--print", "-e", str(staged), "--model", MODEL_REF, "--session-dir", str(work / "session")]
    command += ["--mode", "json"]
    env = {**os.environ, "PI_CODING_AGENT_DIR": str(agent)}
    finished = True
    with open(events, "w") as out, open(work / "pi.err", "w") as err:
        # start pi in its own process group, so stopping it also stops anything it started, e.g. a second pi
        proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=out, stderr=err, cwd=copy, env=env, text=True, start_new_session=True
        )
        try:
            proc.communicate(input=message, timeout=timeout)
        except subprocess.TimeoutExpired:
            finished = False
        finally:
            stop(proc=proc)
    diff = collected_diff(repo=copy, start=start)
    (work / "patch.diff").write_text(diff)
    return Tried(events=events, repo=copy, diff=diff, finished=finished)


def stop(proc: subprocess.Popen) -> None:
    """Stop pi and everything it started"""
    if proc.poll() is None:
        # SIGTERM first so pi stops its bash commands itself, since each one runs in its own process group
        _signal_group(pid=proc.pid, sig=signal.SIGTERM)
        try:
            # keep this short, since the bash tool's timeout kills try-variant 5 s after sending SIGTERM
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    _signal_group(pid=proc.pid, sig=signal.SIGKILL)
    proc.wait()


def _signal_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def show(tried: Tried, timeout: int) -> None:
    print("\n".join(summary(events=tried.events)))
    ended = "the agent finished" if tried.finished else f"stopped after {timeout} s"
    print(f"\n{ended}. Events: {tried.events}, errors: {tried.events.parent / 'pi.err'}, repo: {tried.repo}")
    diff = tried.diff
    print(f"diff ({len(diff.splitlines())} lines):\n{diff[:6000]}" if diff else "the run changed no file")


def question(package: Path, timeout: int, tries: int) -> int:
    """Ask until the outcome is clear, and exit 1 on a problem or 2 if the model couldn't be asked enough times"""
    # the model can cause any of these problems by chance, e.g. it leaves a stray file that a package's hook then
    # rightly acts on. So only report a problem seen on 2 tries the model answered, or on the only try if tries is 1
    message = (QUESTION / "question.md").read_text().strip()
    needed = min(2, tries)
    with_problems = 0
    down = ""
    for _ in range(tries):
        tried = run(package=package, repo=QUESTION / "repo", message=message, timeout=timeout)
        show(tried=tried, timeout=timeout)
        failed = unavailable(events=tried.events)
        # the package can make pi's requests fail too, so only blame the model if a plain request fails as well
        if failed and not model_answers():
            down = failed
            print(f"\nthe model couldn't be asked: {failed}\n")
            continue
        problems = question_problems(tried=tried, timeout=timeout, failed=failed)
        if not problems:
            return verdict(problems=[])
        with_problems += 1
        if with_problems == needed:
            return verdict(problems=problems)
        print("\nasking again, since that could be chance\n")
    print(f"unavailable: {down or 'the model answered too few tries to tell'}")
    return 2


def verdict(problems: list[str]) -> int:
    for problem in problems:
        print(f"problem: {problem}")
    if not problems:
        print("ok: the package stayed out of the way of a question that needed no change")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="try-variant", description="Run a pi package the way a task run does, against the live model"
    )
    parser.add_argument("package", type=Path)
    parser.add_argument("repo", type=Path, nargs="?")
    parser.add_argument(
        "task", type=Path, nargs="?", help="a file with the message the agent gets, as a user would write it"
    )
    parser.add_argument(
        "--question",
        action="store_true",
        help="ask a question about a small repo that needs no change instead, and say how the package got in the way",
    )
    parser.add_argument(
        "--tries", type=int, default=1, help="with --question, ask again up to this many times while it could be chance"
    )
    parser.add_argument(
        "--timeout", type=int, default=540, help="seconds before the run is stopped (the bash tool stops at 600)"
    )
    args = parser.parse_args(argv[1:])
    # handle SIGTERM (sent by the bash tool's timeout) like ctrl-c, so pi gets stopped too
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal.default_int_handler)
    if args.question:
        return question(package=args.package.resolve(), timeout=args.timeout, tries=args.tries)
    if not (args.repo and args.task):
        parser.error("give a repo and a task file, or --question")
    tried = run(
        package=args.package.resolve(),
        repo=args.repo.resolve(),
        message=args.task.read_text().strip(),
        timeout=args.timeout,
    )
    show(tried=tried, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(argv=sys.argv))
    except KeyboardInterrupt:
        sys.exit("try-variant was stopped, and so was pi")
