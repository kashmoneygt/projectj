import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import check_variant

MODELS = Path("/opt/rsi/models")
TASK_RULES = Path("/opt/rsi/task-rules.md")
TRIES = Path("/out/tries")
MODEL_REF = (MODELS / "ref.txt").read_text().strip() if (MODELS / "ref.txt").is_file() else ""
SHOWN = 160
# git with the repo's hooks, fsmonitor, external diff and textconv off, as in a task run
GIT = ("git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null")


def stage(package: Path, repo: Path, copy: Path, work: Path) -> tuple[str, Path, Path]:
    """Copy and commit the repo, then stage the package"""
    shutil.copytree(repo, copy, symlinks=True)
    if not (copy / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=copy, check=True)
    # keep Python caches out of the diff, as real repos do
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
    budget = check_variant.sets_thinking_budgets(package=package)
    shutil.copy2(MODELS / ("budget.json" if budget else "plain.json"), agent / "models.json")
    return start.stdout.strip(), staged, agent


def collected_diff(repo: Path, start: str) -> str:
    """Every change since start, collected like a task run does"""
    staged = subprocess.run([*GIT, "add", "-A"], cwd=repo, check=False).returncode == 0
    diff = [*GIT, "diff", *(["--cached"] if staged else []), "--binary", "--no-ext-diff", "--no-textconv", start]
    return subprocess.run(diff, cwd=repo, capture_output=True, text=True, check=False).stdout


def summary(events: Path) -> list[str]:
    lines: list[str] = []
    turn = 0
    for raw in events.read_text(errors="replace").splitlines():
        if '"message_update"' in raw or '"tool_execution_update"' in raw:
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        kind = record.get("type")
        message = record.get("message") or {}
        if kind == "message_end" and message.get("role") == "assistant":
            turn += 1
            parts = [p for p in message.get("content") or [] if isinstance(p, dict)]
            calls = [p for p in parts if p.get("type") == "toolCall"]
            words = " ".join(str(p.get("text") or p.get("thinking") or "") for p in parts).split()
            head = f"[{turn}] {message.get('usage', {}).get('output', 0)} tokens, stop {message.get('stopReason')}"
            lines.append(head + (f": {' '.join(words)[:SHOWN]}" if not calls and words else ""))
            for call in calls:
                lines.append(f"    {call.get('name')}: {json.dumps(call.get('arguments'))[:SHOWN]}")
        elif kind == "entry_appended" and (record.get("entry") or {}).get("type") == "custom_message":
            entry = record["entry"]
            text = entry.get("content") if isinstance(entry.get("content"), str) else json.dumps(entry.get("content"))
            lines.append(f"    harness message ({entry.get('customType')}): {text[:400]}")
        elif kind == "auto_retry_start":
            lines.append(f"    request retried: {str(record.get('errorMessage'))[:200]}")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="try-variant", description="Run a pi package the way a task run does, against the live model"
    )
    parser.add_argument("package", type=Path)
    parser.add_argument("repo", type=Path)
    parser.add_argument("issue", type=Path, help="a file with the issue text the agent gets")
    parser.add_argument(
        "--timeout", type=int, default=540, help="seconds before the run is stopped (the bash tool stops at 600)"
    )
    args = parser.parse_args(argv[1:])
    TRIES.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="try-", dir=TRIES))
    repo = Path(tempfile.mkdtemp(prefix="try-repo-", dir="/tmp")) / "repo"
    start, staged, agent = stage(package=args.package.resolve(), repo=args.repo.resolve(), copy=repo, work=work)
    message = f"{args.issue.read_text().strip()}\n\n{TASK_RULES.read_text()}"
    events = work / "events.jsonl"
    command = ["pi", "--print", "-e", str(staged), "--model", MODEL_REF, "--session-dir", str(work / "session")]
    command += ["--mode", "json"]
    env = {**os.environ, "PI_CODING_AGENT_DIR": str(agent)}
    ended = "the agent finished"
    with open(events, "w") as out, open(work / "pi.err", "w") as err:
        try:
            subprocess.run(
                command,
                input=message,
                text=True,
                stdout=out,
                stderr=err,
                cwd=repo,
                env=env,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            ended = f"stopped after {args.timeout} s"
    diff = collected_diff(repo=repo, start=start)
    (work / "patch.diff").write_text(diff)
    print("\n".join(summary(events=events)))
    print(f"\n{ended}. Events: {events}, errors: {work / 'pi.err'}, repo: {repo}")
    print(f"diff ({len(diff.splitlines())} lines):\n{diff[:6000]}" if diff else "the run changed no file")
    return 0


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
