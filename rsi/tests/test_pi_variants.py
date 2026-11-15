import ast
import io
import json
import subprocess
import sys
import tokenize
from pathlib import Path

import requests

from rsi import core, folders, paths

pi = folders.module(root=paths.harnesses_dir(), name="pi", file="harness.py", kind="harness")


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "v"
    (package / "extensions").mkdir(parents=True)
    (package / "package.json").write_text((pi.HARNESS.package_dir / "package.json").read_text())
    (package / "extensions" / "notes.ts").write_text("export default function notes(pi) {}\n")
    (package / "variant.json").write_text("{}")
    return package


def test_check_accepts_the_baseline_shape(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    (package / "SYSTEM.md").write_text("You fix bugs in large codebases.\n")
    (package / "APPEND_SYSTEM.md").write_text("Verify before you finish.\n")
    (package / "extensions" / "guard.ts").write_text(
        "export default function guard(pi) { pi.on('turn_end', () => {}); }\n"
    )
    (package / "extensions" / "review.ts").write_text(
        "export default function review(pi) {\n"
        "  pi.on('agent_end', async (_e, ctx) => { await ctx.modelRegistry.completeSimple(ctx.model, {}); });\n"
        "}\n"
    )
    (package / "skills" / "debug").mkdir(parents=True)
    (package / "skills" / "debug" / "SKILL.md").write_text("---\nname: debug\ndescription: d\n---\nsteps\n")
    (package / "settings.json").write_text(
        json.dumps(
            {
                "defaultTools": ["+grep"],
                "compaction": {"enabled": True, "reserveTokens": 20000},
                "branchSummary": {"skipPrompt": True},
                "retry": {"maxRetries": 5, "provider": {"timeoutMs": 600000}},
                "steeringMode": "all",
                "shellCommandPrefix": "set -o pipefail;",
                "hideThinkingBlock": True,
            }
        )
    )
    assert pi.check.static_problems(package=package) == []


def test_check_reports_a_changed_manifest(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    baseline = pi.HARNESS.package_dir / "package.json"
    assert pi.check.static_problems(package=package, manifest=baseline) == []
    manifest = json.loads(baseline.read_text())
    (package / "package.json").write_text(json.dumps({**manifest, "name": "@projectj/pi-my-variant"}, indent=2) + "\n")
    assert pi.check.static_problems(package=package, manifest=baseline) == [
        "package.json differs from the baseline's; leave it unchanged and say what changed in variant.json"
    ]


def test_check_reports_files_that_never_reach_the_agent(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    (package / "prompts").mkdir()
    (package / "prompts" / "system.md").write_text("never staged\n")
    (package / "NOTES.md").write_text("scratch\n")
    # pi only warns about a tools/ folder
    (package / "tools").mkdir()
    (package / "tools" / "search.ts").write_text("export default {}\n")
    problems = pi.check.static_problems(package=package)
    assert any(p.startswith("prompts/system.md never reaches the agent") for p in problems)
    assert any(p.startswith("NOTES.md never reaches the agent") for p in problems)
    assert any(p.startswith("tools/search.ts never reaches the agent") for p in problems)


def test_check_reports_manifest_mistakes(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    (package / "extensions" / "guard.ts").write_text("export default function guard(pi) {}\n")
    (package / "package.json").write_text(
        json.dumps({"pi": {"extensions": ["extensions/guard.ts", "extensions/x.ts"]}})
    )
    problems = pi.check.static_problems(package=package)
    assert "package.json lists 'extensions/x.ts', which does not exist" in problems
    assert any(p.startswith("extensions/notes.ts is an extension that package.json does not load") for p in problems)
    (package / "package.json").write_text(json.dumps({"pi": {"extensions": ["../outside.ts"]}}))
    assert "package.json lists '../outside.ts', which points outside the package" in pi.check.static_problems(
        package=package
    )
    (package / "package.json").write_text("{not json")
    assert pi.check.static_problems(package=package)[0].startswith("package.json is not valid JSON")


def test_loaded_extensions_follow_pi_manifest_rules(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    (package / "extensions" / "my-guard.ts").write_text("export default function g(pi) {}\n")
    (package / "extensions" / "lib").mkdir()
    (package / "extensions" / "lib" / "util.ts").write_text("export const x = 1;\n")
    both = {"extensions/notes.ts", "extensions/my-guard.ts"}

    def loaded(entries: object) -> set[str]:
        (package / "package.json").write_text(json.dumps({"pi": {"extensions": entries}}))
        return pi.check.loaded_extensions(package=package)

    assert loaded(entries=["extensions/*.ts"]) == both
    assert loaded(entries=["extensions"]) == both and loaded(entries=["./extensions"]) == both
    assert loaded(entries=["extensions/**/*.ts"]) == both | {"extensions/lib/util.ts"}
    assert loaded(entries=["extensions/my-guard.ts"]) == {"extensions/my-guard.ts"}
    assert loaded(entries=["extensions/*.ts", "!notes.ts"]) == {"extensions/my-guard.ts"}
    assert loaded(entries=["extensions/*.ts", "-extensions/notes.ts"]) == {"extensions/my-guard.ts"}
    assert loaded(entries=[]) == set()
    assert loaded(entries=None) == both
    (package / "package.json").write_text(json.dumps({"pi": None}))
    assert pi.check.loaded_extensions(package=package) == both


def test_check_survives_odd_manifests_and_rejects_symlinks(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    (package / "extensions" / "cap.ts").write_text("export default (pi) => pi.setModel(null);\n")
    (package / "package.json").write_text(json.dumps({"pi": None}))
    problems = pi.check.static_problems(package=package)
    assert any("changes the model or its thinking level" in p for p in problems)
    (package / "extensions" / "link.ts").symlink_to("/work/parent/extensions/notes.ts")
    assert pi.check.static_problems(package=package) == [
        "extensions/link.ts is a symlink; a package holds regular files only"
    ]
    assert pi.check.check(package=package) == pi.check.static_problems(package=package)


def test_check_rejects_changes_to_the_model_contract(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    (package / "extensions" / "cap.ts").write_text(
        "export default function cap(pi) {\n"
        "  pi.on('before_provider_request', (e) => ({ ...e.payload, max_tokens: 32768 }));\n"
        "  pi.setThinkingLevel('off');\n"
        "  pi.unregisterProvider('local');\n"
        "  process.env.RSI_MAX_TURNS = '1000';\n"
        "}\n"
    )
    (package / "settings.json").write_text(
        json.dumps(
            {"defaultThinkingLevel": "high", "compaction": {"enabled": True}, "thinkingBudgets": {"medium": 6000}}
        )
    )
    problems = pi.check.static_problems(package=package)
    assert any("rewrites the model request" in p for p in problems)
    assert any("changes the model or its thinking level" in p for p in problems)
    assert any("changes the model providers or model routes (unregisterProvider)" in p for p in problems)
    assert any("touches the evaluation instrumentation" in p for p in problems)
    assert any("settings.json sets defaultThinkingLevel" in p for p in problems)
    assert "settings.json sets thinkingBudgets, which caps how much the model thinks, which is set with the model" in (
        problems
    )
    assert not any("compaction" in p for p in problems)


def test_check_takes_only_settings_about_how_the_agent_works(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    settings = {
        "defaultModel": "local/other",
        "enabledModels": ["local/*"],
        "httpProxy": "http://127.0.0.1:8080",
        "cacheWarming": "idle",
        "extensions": ["/usr/local/lib/node_modules/@earendil-works/pi-coding-agent/examples/extensions"],
        "packages": ["npm:some-pi-package"],
        "skills": ["/testbed/.skills"],
        "defaultProjectTrust": "always",
        "shellPath": "/bin/zsh",
        "sessionDir": "/tmp/sessions",
        "enableAnalytics": True,
        "maxTurns": 50,
        "retry": {"maxRetries": 2},
    }
    (package / "settings.json").write_text(json.dumps(settings))
    assert pi.check.static_problems(package=package) == [
        "settings.json sets defaultModel, which picks the model, its provider or its thinking level",
        "settings.json sets enabledModels, which picks the model, its provider or its thinking level",
        "settings.json sets httpProxy, which changes how requests reach the model",
        "settings.json sets cacheWarming, which sends the model requests of its own",
        "settings.json sets extensions, which loads code or files from outside the package",
        "settings.json sets packages, which loads code or files from outside the package",
        "settings.json sets skills, which loads code or files from outside the package",
        "settings.json sets defaultProjectTrust, which loads code or files from outside the package",
        "settings.json sets shellPath, which runs a program from outside the package",
        "settings.json sets sessionDir, which moves the session files the runner keeps",
        "settings.json sets enableAnalytics, which sends data off the machine",
        "settings.json sets maxTurns, which pi doesn't have",
    ]
    assert not pi.check._SETTINGS & pi.check._FIXED_SETTINGS.keys()


def _write_events(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records))


def test_contract_violations(tmp_path: Path) -> None:
    model = core.ModelSpec(
        name="q",
        source="s",
        model_id="qwen",
        ctx=1000,
        max_output_tokens=600,
        sampling={"temperature": 1.0, "top_p": 0.95},
        chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "xhigh"},
    )
    kwargs = {"enable_thinking": True, "reasoning_effort": "xhigh"}
    requests = tmp_path / "requests.jsonl"
    chat = {
        "model": "qwen",
        "stream": True,
        "stream_options": {"include_usage": True},
        "store": False,
        "max_completion_tokens": 600,
        "temperature": 1,
        "top_p": 0.95,
        "chat_template_kwargs": kwargs,
    }
    params = [
        chat,
        {"model": "qwen", "max_tokens": 90, "temperature": 1, "top_p": 0.95, "chat_template_kwargs": kwargs},
    ]

    def logged() -> Path:
        lines = [{"ip": "a", "method": "POST", "path": "/v1/chat/completions", "params": p} for p in params]
        requests.write_text("".join(json.dumps(line) + "\n" for line in lines))
        return requests

    assert pi.events.contract_violations(requests=logged(), model=model) == []
    low = {**kwargs, "reasoning_effort": "low"}
    params.append({"model": "other", "max_tokens": 900, "temperature": 0.7, "chat_template_kwargs": low})
    params.append({**chat, "seed": 7, "max_tokens": 10, "max_completion_tokens": 5000})
    assert pi.events.contract_violations(requests=logged(), model=model) == [
        "chat_template_kwargs {'enable_thinking': True, 'reasoning_effort': 'low'}",
        "model other",
        "output cap 5000 > 600",
        "output cap 900 > 600",
        "sampling {'temperature': 0.7}",
        "unexpected key seed",
    ]
    assert pi.events.contract_violations(requests=tmp_path / "missing.jsonl", model=model) == []


def test_contract_violations_check_every_request_the_forwarder_logged(tmp_path: Path) -> None:
    kwargs = {"enable_thinking": True, "reasoning_effort": "medium"}
    model = core.ModelSpec(
        name="q",
        source="s",
        model_id="qwen",
        ctx=1000,
        sampling={"temperature": 1.0, "top_p": 0.95},
        chat_template_kwargs=kwargs,
    )
    chat = {
        "model": "qwen",
        "stream": True,
        "stream_options": {"include_usage": True},
        "store": False,
        "max_completion_tokens": 900,
        "temperature": 1,
        "top_p": 0.95,
        "chat_template_kwargs": kwargs,
    }
    lines = [
        {"ip": "a", "method": "POST", "path": "/v1/chat/completions", "params": chat},
        {"ip": "a", "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        {"ip": "a", "method": "POST", "path": "/tokenize", "params": {"content": "(5000 bytes)"}},
    ]
    requests = tmp_path / "requests.jsonl"
    requests.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert pi.events.contract_violations(requests=requests, model=model) == []
    no_top_p = {key: value for key, value in chat.items() if key != "top_p"}
    lines += [
        {
            "ip": "a",
            "method": "POST",
            "path": "/v1/chat/completions",
            "params": {**chat, "temperature": 0.2, "seed": 7},
        },
        {"ip": "a", "method": "POST", "path": "/v1/chat/completions", "params": no_top_p},
        {"ip": "a", "method": "POST", "path": "/v1/chat/completions", "params": {**chat, "thinking_token_budget": 60}},
        {"ip": "a", "method": "POST", "path": "/v1/completions?x=1", "params": {"prompt": "(9000 bytes)"}},
        {"ip": "a", "method": "POST", "path": "/v1/chat/completions", "unread": "chunked body"},
        {"ip": "a", "unread": "read_requests: ValueError: not enough values to unpack"},
    ]
    requests.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert pi.events.contract_violations(requests=requests, model=model) == [
        "request to /v1/completions",
        "sampling {'temperature': 0.2, 'top_p': 0.95}",
        "sampling {'temperature': 1}",
        "unexpected key seed",
        "unexpected key thinking_token_budget",
        "unread /v1/chat/completions: chunked body",
        "unread traffic: read_requests: ValueError: not enough values to unpack",
    ]


def test_task_metrics_count_every_call_the_forwarder_saw(tmp_path: Path) -> None:
    usage = {"input": 500, "output": 40, "cacheRead": 1000}
    _write_events(
        path=tmp_path / "events.jsonl",
        records=[{"type": "message_end", "message": {"role": "assistant", "usage": usage}}],
    )
    model = core.ModelSpec(name="q", source="s", model_id="qwen", ctx=131072)
    # the agent's own events don't count, since the forwarder sees every call
    m = pi.events.task_metrics(instance_dir=tmp_path, model=model, seconds=10.0, patch="")
    assert (m.requests, m.input_tokens, m.output_tokens) == (0, 0, 0)
    chat = {"method": "POST", "path": "/v1/chat/completions", "params": {}}
    lines = [
        {"ip": "a", **chat},
        {"ip": "a", "method": "POST", "path": "/tokenize", "params": {}},
        {
            "ip": "a",
            "usage": {"prompt_tokens": 1500, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 1000}},
        },
        {"ip": "a", **chat},
        {"ip": "a", "usage": {"prompt_tokens": 300, "completion_tokens": 25}},
    ]
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    m = pi.events.task_metrics(instance_dir=tmp_path, model=model, seconds=10.0, patch="")
    assert (m.requests, m.input_tokens, m.output_tokens) == (2, 800, 65)
    assert (m.cache_read_tokens, m.context_pct, m.tok_s) == (1000, 1.1, 4.0)


def test_task_metrics_measure_the_whole_prompt(tmp_path: Path) -> None:
    _write_events(
        path=tmp_path / "events.jsonl",
        records=[
            {"type": "message_end", "message": {"role": "assistant", "usage": {"input": 500, "cacheRead": 64_500}}},
        ],
    )
    model = core.ModelSpec(name="q", source="s", model_id="qwen", ctx=131072)
    m = pi.events.task_metrics(instance_dir=tmp_path, model=model, seconds=10.0, patch="+a\n\n")
    assert m.context_pct == 49.6
    assert m.patch_lines == 1


def test_the_teacher_gets_pis_session_record_and_the_metrics_but_nothing_that_says_if_a_run_passed(
    tmp_path: Path,
) -> None:
    (tmp_path / "session").mkdir()
    for name in ("2026-11-06T21-19-02Z_a.jsonl", "2026-11-06T21-30-00Z_b.jsonl", "events.jsonl", "metrics.jsonl"):
        (tmp_path / ("session" if name.startswith("2026") else "") / name).write_text("{}\n")
    (tmp_path / "result.json").write_text('{"resolved": true}')
    assert pi.HARNESS.run_files(instance_dir=tmp_path) == {
        "session.jsonl": tmp_path / "session" / "2026-11-06T21-19-02Z_a.jsonl",
        "session-2.jsonl": tmp_path / "session" / "2026-11-06T21-30-00Z_b.jsonl",
        "metrics.jsonl": tmp_path / "metrics.jsonl",
    }


def test_task_metrics_count_the_largest_and_capped_responses(tmp_path: Path) -> None:
    _write_events(
        path=tmp_path / "events.jsonl",
        records=[
            {
                "type": "message_end",
                "message": {"role": "assistant", "stopReason": "toolUse", "usage": {"output": 900}},
            },
            {
                "type": "message_end",
                "message": {"role": "assistant", "stopReason": "length", "usage": {"output": 16384}},
            },
            {
                "type": "message_end",
                "message": {"role": "assistant", "stopReason": "length", "usage": {"output": 16384}},
            },
            {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop", "usage": {"output": 40}}},
        ],
    )
    model = core.ModelSpec(name="q", source="s", model_id="qwen", ctx=131072)
    m = pi.events.task_metrics(instance_dir=tmp_path, model=model, seconds=10.0, patch="")
    # with no metrics.jsonl, the stop reason comes from the agent's own events
    assert (m.peak_output_tokens, m.capped_requests, m.stop_reason) == (16384, 2, "stop")


def test_a_call_still_running_when_the_run_ended_counts_until_then(tmp_path: Path) -> None:
    call = {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "grep -r x /"}}
    _write_events(
        path=tmp_path / "events.jsonl",
        records=[
            {
                "type": "message_end",
                "message": {"role": "user", "timestamp": 1_000, "content": [{"type": "text", "text": "Fix it."}]},
            },
            {
                "type": "message_end",
                "message": {"role": "assistant", "timestamp": 5_000, "stopReason": "toolUse", "content": [call]},
            },
            {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "bash"},
        ],
    )
    model = core.ModelSpec(name="q", source="s", model_id="qwen", ctx=131072)
    m = pi.events.task_metrics(instance_dir=tmp_path, model=model, seconds=600.0, patch="")
    assert (m.tool_ms, m.tool_stats["bash"].count) == (596_000, 1)


def test_wait_timeouts_are_recognized_in_both_shapes() -> None:
    assert pi._timed_out(exc=requests.exceptions.ReadTimeout("read timeout=60"))
    pool = "UnixHTTPConnectionPool(host='localhost', port=None): Read timed out."
    assert pi._timed_out(exc=requests.exceptions.ConnectionError(pool))
    assert not pi._timed_out(exc=requests.exceptions.ConnectionError("Connection refused"))
    assert not pi._timed_out(exc=RuntimeError("no such container"))


def test_model_address() -> None:
    local = core.ModelEndpoint(model_id="q", base_url="http://host.docker.internal:11501/v1")
    assert pi.model_address(endpoint=local) == ("host.docker.internal", 11501)
    remote = core.ModelEndpoint(model_id="q", base_url="https://api.example.com/v1")
    assert pi.model_address(endpoint=remote) == ("api.example.com", 443)


def test_check_rejects_workload_terms_in_agent_visible_files(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    terms = ("swebench", "benchmark", "hidden tests")
    assert pi.check.static_problems(package=package, terms=terms) == []
    (package / "APPEND_SYSTEM.md").write_text("Do not try to identify the task as a known Benchmark instance.\n")
    (package / "extensions" / "done.ts").write_text("// the hidden tests check it\nexport default (pi) => {};\n")
    (package / "scripts").mkdir()
    (package / "scripts" / "bin.dat").write_bytes(b"\x00swebench\x00")
    problems = pi.check.static_problems(package=package, terms=terms)
    assert problems == [
        "APPEND_SYSTEM.md names the workload or its protocol ('benchmark')",
        "extensions/done.ts names the workload or its protocol ('hidden tests')",
    ]
    (package / "variant.json").write_text('{"why": "tuned on swebench benchmark failures"}')
    assert pi.check.term_problems(package=package, terms=terms) == problems


def test_check_leaves_the_card_to_the_loop(tmp_path: Path) -> None:
    package = _package(tmp_path=tmp_path)
    assert pi.check.static_problems(package=package) == []
    (package / "variant.json").unlink()
    assert pi.check.static_problems(package=package) == []


def test_scripts_run_with_the_standard_library_alone() -> None:
    # the teacher's container runs them without rsi installed
    for script in (pi.CHECK_SCRIPT, pi.TRY_SCRIPT):
        done = subprocess.run(
            [sys.executable, "-S", str(script), "--help"], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr
    loaded = subprocess.run(
        [sys.executable, "-S", "-c", f"import runpy; runpy.run_path({str(pi.FORWARDER_SCRIPT)!r})"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert loaded.returncode == 0, loaded.stderr


def _newer_f_strings(source: str) -> list[int]:
    """Lines with an f-string python 3.11 can't parse: one inside another with the same quotes, or a backslash in {}"""
    outer: list[str] = []  # the quotes of the f-strings we're inside
    found: list[int] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.FSTRING_MIDDLE:
            if len(outer) > 1 and "\\" in token.string:
                found.append(token.start[0])
            continue
        if token.type in (tokenize.STRING, tokenize.FSTRING_START):
            body = token.string.lstrip("rRbBuUfF")
            quote = body[:3] if body[:3] in ('"""', "'''") else body[:1]
            if outer and (
                "\\" in token.string or (quote[0] == outer[-1][0] and (len(outer[-1]) == 1 or len(quote) == 3))
            ):
                found.append(token.start[0])
            if token.type == tokenize.FSTRING_START:
                outer.append(quote)
        elif token.type == tokenize.FSTRING_END:
            outer.pop()
    return found


def test_scripts_parse_on_the_containers_python() -> None:
    # pi's image, which runs them in the containers, has python 3.11. Its parser rejects these, as tried in the image
    rejected = ['f"{x["a"]}"', 'f"""{f"""{1}"""}"""', r'''f"{'\n'.join(x)}"''', r'''f"{f'a\n'}"''']
    accepted = ["f\"{x['a']}\"", 'f"""{x["a"]}"""', r'''f"a\n{x}"''', "f\"{f'{x}'}\""]
    assert [bool(_newer_f_strings(source=s)) for s in rejected + accepted] == [True] * 4 + [False] * 4
    for script in (pi.CHECK_SCRIPT, pi.TRY_SCRIPT, pi.CARDS_SCRIPT, pi.FORWARDER_SCRIPT):
        source = script.read_text()
        ast.parse(source, filename=str(script), feature_version=(3, 11))
        assert _newer_f_strings(source=source) == [], script.name
