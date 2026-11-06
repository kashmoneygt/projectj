#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CARD = "VARIANT.md"  # never reaches the agent

# what the agent gets (SYSTEM.md replaces pi's system prompt, APPEND_SYSTEM.md adds to it)
RUNTIME_FILES = ("package.json", "SYSTEM.md", "APPEND_SYSTEM.md", "settings.json", "mcp.json")
RUNTIME_DIRS = ("extensions", "skills", "agents", "scripts")
# pi reads these from its agent dir, not from a package
AGENT_DIR_FILES = ("SYSTEM.md", "APPEND_SYSTEM.md", "settings.json", "mcp.json")

_CODE_SUFFIXES = (".ts", ".mts", ".cts", ".js", ".mjs", ".cjs", ".py", ".sh")
# code that would change the model, its requests or our measurements
_FORBIDDEN_CODE = (
    (re.compile(r"before_provider_(?:request|headers)"), "rewrites the model request"),
    (re.compile(r"\bregister(?:Provider|VirtualModel)\b"), "adds a model provider or model route"),
    (re.compile(r"\bset(?:Model|ThinkingLevel)\b"), "changes the model or its thinking level"),
    (re.compile(r"\bRSI_[A-Z_]+"), "touches the evaluation instrumentation"),
)
# settings.json keys a variant can set, from pi 1.4.0's settings-schema.ts
_SETTINGS = frozenset(
    (
        # how the agent works
        "defaultTools",
        "codemode",
        "compaction",
        "branchSummary",
        "retry",
        "httpIdleTimeoutMs",
        "websocketConnectTimeoutMs",
        "enableSkillCommands",
        "steeringMode",
        "followUpMode",
        "queueMode",
        "shellCommandPrefix",
        "images",
        # display only
        "$schema",
        "lastChangelogVersion",
        "theme",
        "hideThinkingBlock",
        "showCacheMissNotices",
        "quietStartup",
        "collapseChangelog",
        "terminal",
        "doubleEscapeAction",
        "treeFilterMode",
        "editorPaddingX",
        "outputPad",
        "autocompleteMaxVisible",
        "showHardwareCursor",
        "markdown",
        "warnings",
        "tuiMode",
        "fullscreenExitOutput",
        "fullscreenScrollbar",
        "fullscreenCopyOnSelect",
        "fullscreenWheelScrollLines",
    )
)
# the rest of pi 1.4.0's settings, and why a variant can't set them
_FIXED_SETTINGS = {
    **dict.fromkeys(
        ("defaultProvider", "defaultModel", "defaultThinkingLevel", "modelThinkingLevels", "enabledModels"),
        "picks the model, its provider or its thinking level",
    ),
    # pi only sends a budget when the provider config names the server's budget field, and that config is the model's
    "thinkingBudgets": "caps how much the model thinks, which is set with the model",
    **dict.fromkeys(("httpProxy", "transport", "websockets"), "changes how requests reach the model"),
    "cacheWarming": "sends the model requests of its own",
    **dict.fromkeys(
        ("extensions", "packages", "npmCommand", "skills", "prompts", "themes", "defaultProjectTrust"),
        "loads code or files from outside the package",
    ),
    **dict.fromkeys(("shellPath", "externalEditor"), "runs a program from outside the package"),
    "sessionDir": "moves the session files the runner keeps",
    **dict.fromkeys(
        ("sessionShareDestination", "enableInstallTelemetry", "enableAnalytics", "trackingId", "deviceId"),
        "sends data off the machine",
    ),
}
_LOAD_ERRORS = (
    "Failed to load extension",
    "Extension path does not exist",
    "Unknown built-in extension",
    "extension name conflict",
)
_DEAD_MODELS = {
    "providers": {
        "local": {
            "baseUrl": "http://127.0.0.1:9/v1",
            "api": "openai-completions",
            "apiKey": "local",
            "models": [{"id": "loadcheck", "contextWindow": 8192, "maxTokens": 4096}],
        }
    }
}


def stage_runtime_package(package: Path, dest: Path) -> Path:
    """Copy the files the agent gets (a run mounts them at /opt/harness)"""
    dest.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_FILES:
        if (package / name).is_file():
            shutil.copy2(package / name, dest / name)
    for name in RUNTIME_DIRS:
        if (package / name).is_dir():
            shutil.copytree(package / name, dest / name, dirs_exist_ok=True)
    return dest


def copy_agent_resources(package: Path, agent_dir: Path) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    for name in AGENT_DIR_FILES:
        if (package / name).is_file():
            (agent_dir / name).write_bytes((package / name).read_bytes())
    if (package / "agents").is_dir():
        (agent_dir / "agents").mkdir(parents=True, exist_ok=True)
        for f in sorted((package / "agents").glob("*.md")):
            (agent_dir / "agents" / f.name).write_bytes(f.read_bytes())


def _visible(package: Path) -> list[str]:
    """Every path in the package except dotfiles"""
    return sorted(
        p.relative_to(package).as_posix()
        for p in package.rglob("*")
        if not any(part.startswith(".") for part in p.relative_to(package).parts)
    )


def _files(package: Path) -> list[str]:
    return [rel for rel in _visible(package=package) if (package / rel).is_file()]


def _is_runtime(rel: str) -> bool:
    return rel in RUNTIME_FILES or rel.split("/", 1)[0] in RUNTIME_DIRS


def _pi_manifest(folder: Path) -> dict[str, list[str]] | None:
    """The "pi" section of package.json as pi reads it, or None"""
    try:
        data = json.loads((folder / "package.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pi"), dict):
        return None
    lists = {key: value for key, value in data["pi"].items() if isinstance(value, list)}
    return {key: value for key, value in lists.items() if all(isinstance(entry, str) for entry in value)}


def _is_pattern(entry: str) -> bool:
    """Same as pi's isPattern"""
    return entry.startswith(("!", "+", "-")) or "*" in entry or "?" in entry


def _glob(pattern: str) -> re.Pattern[str]:
    """A glob as a regex, like minimatch ('*' stays in one folder, '**' crosses folders)"""
    out = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] in "*?":
            out, i = out + ("[^/]*" if pattern[i] == "*" else "[^/]"), i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z")


def _entry_points(package: Path, folder: Path) -> list[str] | None:
    """The extensions a folder names itself: package.json entries, or index.ts / index.js"""
    manifest = _pi_manifest(folder=folder) or {}
    listed = [(folder / e).resolve() for e in manifest.get("extensions", [])]
    found = [p.relative_to(package.resolve()).as_posix() for p in listed if p.exists()]
    if found:
        return found
    for index in ("index.ts", "index.js"):
        if (folder / index).is_file():
            return [(folder / index).relative_to(package).as_posix()]
    return None


def _discover(package: Path, folder: Path) -> list[str]:
    """Find the extensions in a folder the way pi does"""
    named = _entry_points(package=package, folder=folder)
    if named is not None:
        return named
    found: list[str] = []
    for child in sorted(folder.iterdir()):
        if child.name.startswith(".") or child.name == "node_modules":
            continue
        if child.is_file() and child.suffix in (".ts", ".js"):
            found.append(child.relative_to(package).as_posix())
        elif child.is_dir():
            found += _entry_points(package=package, folder=child) or []
    return found


def loaded_extensions(package: Path) -> set[str]:
    """The extension files pi loads from the package"""
    manifest = _pi_manifest(folder=package)
    entries = manifest.get("extensions") if manifest is not None else None
    if entries is None:
        folder = package / "extensions"
        return set(_discover(package=package, folder=folder)) if folder.is_dir() else set()
    visible = _visible(package=package)
    loaded: list[str] = []
    for entry in (e for e in entries if not e.startswith(("!", "+", "-"))):
        if _is_pattern(entry=entry):
            targets = [rel for rel in visible if _glob(pattern=entry.removeprefix("./")).match(rel)]
        else:
            targets = [Path(os.path.normpath(entry)).as_posix()]
        for rel in targets:
            if (package / rel).is_file():
                loaded.append(rel)
            elif (package / rel).is_dir():
                loaded += _discover(package=package, folder=package / rel)
    picked = set(loaded)
    overrides = [e for e in entries if e.startswith(("!", "+", "-"))]
    for pattern in (e[1:].removeprefix("./") for e in overrides if e.startswith("!")):
        regex = _glob(pattern=pattern)
        picked = {rel for rel in picked if not (regex.match(rel) or regex.match(Path(rel).name))}
    picked |= {e[1:].removeprefix("./") for e in overrides if e.startswith("+")} & set(loaded)
    picked -= {e[1:].removeprefix("./") for e in overrides if e.startswith("-")}
    return picked


def term_problems(package: Path, terms: tuple[str, ...]) -> list[str]:
    """Files the agent reads that name the benchmark or its rules (the card may)"""
    problems: list[str] = []
    for rel in _files(package=package):
        if not _is_runtime(rel=rel):
            continue
        raw = (package / rel).read_bytes()
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8", "replace").lower()
        problems += [f"{rel} names the workload or its protocol ({term!r})" for term in terms if term.lower() in text]
    return problems


def static_problems(package: Path, terms: tuple[str, ...] = (), manifest: Path | None = None) -> list[str]:
    """Problems found without starting pi (manifest is the package.json every variant keeps)"""
    links = sorted(p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_symlink())
    if links:
        return [f"{rel} is a symlink; a package holds regular files only" for rel in links]
    if not (package / "package.json").is_file():
        return ["package.json is missing"]
    try:
        json.loads((package / "package.json").read_text())
    except ValueError as exc:
        return [f"package.json is not valid JSON: {exc}"]
    problems: list[str] = term_problems(package=package, terms=terms)
    if manifest is not None and (package / "package.json").read_bytes() != manifest.read_bytes():
        problems.append(
            "package.json differs from the baseline's; leave it unchanged and say what changed in VARIANT.md"
        )
    pi_section = _pi_manifest(folder=package) or {}
    root = package.resolve()
    for entry in pi_section.get("extensions", []) + pi_section.get("skills", []):
        target = (package / entry.lstrip("+-!")).resolve()
        if target != root and root not in target.parents:
            problems.append(f"package.json lists {entry!r}, which points outside the package")
        elif not _is_pattern(entry=entry) and not target.exists():
            problems.append(f"package.json lists {entry!r}, which does not exist")
    loaded = loaded_extensions(package=package)
    for rel in _files(package=package):
        if rel == CARD:
            continue
        if not _is_runtime(rel=rel):
            problems.append(f"{rel} never reaches the agent (runtime files: {', '.join(RUNTIME_FILES + RUNTIME_DIRS)})")
        elif (
            rel.startswith("extensions/")
            and rel.endswith((".ts", ".js"))
            and rel not in loaded
            and "export default" in (package / rel).read_text(errors="replace")
        ):
            problems.append(f"{rel} is an extension that package.json does not load, so pi never runs it")
    for rel in _files(package=package):
        if not _is_runtime(rel=rel) or not rel.endswith(_CODE_SUFFIXES):
            continue
        text = (package / rel).read_text(errors="replace")
        for pattern, why in _FORBIDDEN_CODE:
            match = pattern.search(text)
            if match:
                problems.append(f"{rel}: {why} ({match.group(0)}); the model and its instrumentation are fixed")
    for name in ("settings.json", "mcp.json"):
        if not (package / name).is_file():
            continue
        try:
            config = json.loads((package / name).read_text())
        except ValueError as exc:
            problems.append(f"{name} is not valid JSON: {exc}")
            continue
        if name == "settings.json" and isinstance(config, dict):
            for key in config:
                if key in _FIXED_SETTINGS:
                    problems.append(f"settings.json sets {key}, which {_FIXED_SETTINGS[key]}")
                elif key not in _SETTINGS:
                    problems.append(f"settings.json sets {key}, which pi 1.4.0 doesn't have")
    return problems


def load_problems(package: Path, workdir: Path, timeout_s: float = 90.0) -> list[str]:
    """Start pi with the package against a dead model, to check that it loads"""
    staged = stage_runtime_package(package=package, dest=workdir / "package")
    agent = workdir / "agent"
    copy_agent_resources(package=package, agent_dir=agent)
    (agent / "models.json").write_text(json.dumps(_DEAD_MODELS))
    command = [
        "pi",
        "--print",
        "-e",
        str(staged),
        "--model",
        "local/loadcheck:off",
        "--session-dir",
        str(workdir / "session"),
        "--mode",
        "json",
    ]
    env = {**os.environ, "PI_CODING_AGENT_DIR": str(agent)}
    try:
        done = subprocess.run(
            command,
            input="LOAD-CHECK\n",
            capture_output=True,
            text=True,
            cwd=staged,
            env=env,
            timeout=timeout_s,
            check=False,
        )
        out, err = done.stdout, done.stderr
    except subprocess.TimeoutExpired as exc:
        out, err = _text(raw=exc.stdout), _text(raw=exc.stderr)
    except FileNotFoundError:
        return ["pi is not installed here"]
    problems = [marker for marker in _LOAD_ERRORS if marker in err]
    problems += [line.strip()[:200] for line in err.splitlines() if line.startswith("Error:")]
    if not out.strip():
        problems.append("pi produced no events (it did not start)")
    return problems


def _text(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    return raw.decode(errors="replace") if isinstance(raw, bytes) else raw


def check(package: Path, terms: tuple[str, ...] = (), manifest: Path | None = None) -> list[str]:
    problems = static_problems(package=package, terms=terms, manifest=manifest)
    if any("is a symlink" in problem for problem in problems):
        return problems  # never stage a package with a symlink
    with tempfile.TemporaryDirectory(prefix="check-variant-") as tmp:
        problems += load_problems(package=package, workdir=Path(tmp))
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="check-variant", description="Check a pi variant package.")
    parser.add_argument("package", type=Path)
    parser.add_argument("--terms", type=Path, help="file of words the agent-visible files may not contain")
    parser.add_argument("--manifest", type=Path, help="the package.json every variant keeps")
    args = parser.parse_args(argv[1:])
    if not args.package.is_dir():
        print(f"usage: check-variant <package dir> ({args.package} is not a directory)")
        return 2
    terms = tuple(line.strip() for line in args.terms.read_text().splitlines() if line.strip()) if args.terms else ()
    try:
        problems = check(package=args.package.resolve(), terms=terms, manifest=args.manifest)
    except Exception as exc:  # noqa: BLE001
        problems = [f"check-variant crashed: {type(exc).__name__}: {exc}"]
    for problem in problems:
        print(f"problem: {problem}")
    if not problems:
        print("ok: pi loads the package and every file reaches the agent")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
