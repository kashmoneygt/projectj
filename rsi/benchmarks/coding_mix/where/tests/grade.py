from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPLY = Path("/logs/agent/response.txt")  # the agent's final answer
FIX = Path("/tests/reference/gold.patch")  # the project's real fix
LOGS = Path("/logs/verifier")
OTHERS = 4  # how many extra non-test files an answer can name and still pass
# regex for files in a fix that aren't code (docs, changelogs, CI config, lock files), same as in ../../benchmark.py
NOT_CODE = re.compile(
    r"(^|/)(docs?|changelog|changes|news|\.github)(/|\.|$)|\.(md|rst|txt|adoc)$"
    r"|(^|/)(package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|pnpm-lock\.yaml|Pipfile\.lock|poetry\.lock|uv\.lock)$"
    r"|(^|/)(go\.(mod|sum)|Cargo\.lock)$",
    re.IGNORECASE,
)
TESTS = re.compile(
    r"(^|/)(tests?|__tests__|specs?|testdata|fixtures)/|(^|/)test_|_test\.|\.(test|spec)\.|tests?\.\w+$", re.IGNORECASE
)


def git(args: list[str]) -> list[str]:
    # turn off fsmonitor, since the agent could have set it up to run its own code when we call git
    command = ["git", "-c", "safe.directory=*", "-c", "core.fsmonitor=false", *args]
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    return done.stdout.splitlines()


def named(reply: str, files: list[str]) -> set[str]:
    """Repo files the reply mentions, by full path or by a path suffix that matches only one file"""
    owners: dict[str, set[str]] = {}
    for file in files:
        parts = file.split("/")
        for i in range(len(parts)):
            owners.setdefault("/".join(parts[i:]), set()).add(file)
    found: set[str] = set()
    for token in re.findall(r"[\w.@+-]+(?:/[\w.@+-]+)*", reply):
        token = token.rstrip(".")
        # skip plain words like "test", even if a file has that name
        if "/" not in token and "." not in token:
            continue
        parts = token.split("/")
        for i in range(len(parts)):
            match = owners.get("/".join(parts[i:]), set())
            if len(match) == 1:
                found |= match
                break
    return found


def main() -> None:
    fixed = sorted(
        {f for f in re.findall(r"^diff --git a/(\S+) b/", FIX.read_text(), re.MULTILINE) if not NOT_CODE.search(f)}
    )
    reply = REPLY.read_text(errors="replace") if REPLY.is_file() else ""
    mentioned = named(reply=reply, files=git(args=["ls-files"]))
    missing = [f for f in fixed if f not in mentioned]
    others = sorted(f for f in mentioned - set(fixed) if not TESTS.search(f))
    edited = git(args=["status", "--porcelain", "--untracked-files=no"])
    passed = bool(fixed) and not missing and len(others) <= OTHERS and not edited
    (LOGS / "reward.txt").write_text("1\n" if passed else "0\n")
    print(json.dumps({"fix": fixed, "missing": missing, "others": others, "edited": edited}, indent=2))


if __name__ == "__main__":
    main()
