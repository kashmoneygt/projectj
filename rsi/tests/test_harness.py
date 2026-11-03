import json
import shutil
import subprocess

from rsi import core, folders, harness, paths

pi = folders.module(root=paths.harnesses_dir(), name="pi", file="harness.py", kind="harness")


def test_load_pi_harness() -> None:
    h = harness.load(name="pi")
    assert h.name == "pi"
    assert h.version == "1.4.0"
    assert h.package_dir.name == "package"


def test_pi_package_manifest() -> None:
    h = harness.load(name="pi")
    manifest = json.loads((h.package_dir / "package.json").read_text())
    assert "pi-package" in manifest["keywords"]
    assert manifest["pi"]["extensions"] == ["extensions/*.ts"]
    assert sorted(p.name for p in h.package_dir.iterdir()) == ["package.json"]
    assert h.doc.is_file()
    assert sorted(p.name for p in h.eval_dir.iterdir()) == ["metrics.ts"]
    assert h.task_rules.exists()
    assert not (h.package_dir / "prompts").exists()


def test_pi_install_extensions(tmp_path) -> None:
    source = tmp_path / "eval"
    source.mkdir()
    (source / "metrics.ts").write_text("metrics")
    agent_dir = tmp_path / "agent"
    pi._install_extensions(source_dir=source, agent_dir=agent_dir)
    assert (agent_dir / "extensions" / "metrics.ts").read_text() == "metrics"


def test_pi_timeout_saves_the_diff_a_finished_run_gets() -> None:
    calls: list[list[str]] = []

    class _Container:
        def exec_run(self, cmd: list[str]) -> None:
            calls.append(cmd)

    pi._flush_partial_patch(container=_Container())
    assert calls == [["bash", "-lc", pi.CAPTURE_PATCH]]


def test_pi_patch_keeps_new_and_binary_files(tmp_path) -> None:
    repo, out = tmp_path / "testbed", tmp_path / "out"
    out.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "start"], cwd=repo, check=True)
    fresh = tmp_path / "fresh"
    shutil.copytree(repo, fresh)
    script = pi.CAPTURE_PATCH.replace("/testbed", str(repo)).replace("/out", str(out))
    subprocess.run(["bash", "-c", f"git -C {repo} rev-parse HEAD > {out}/orig-head\n{script}"], check=True)
    assert (out / "patch.diff").read_text() == ""
    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.py").write_text("y = 1\n")
    (repo / "blob.bin").write_bytes(bytes(range(256)))
    subprocess.run([*git, "commit", "-qam", "the agent committed"], cwd=repo, check=True)
    subprocess.run(["bash", "-c", script], check=True)
    subprocess.run(["git", "apply", str(out / "patch.diff")], cwd=fresh, check=True)
    assert (fresh / "a.py").read_text() == "x = 2\n" and (fresh / "new.py").read_text() == "y = 1\n"
    assert (fresh / "blob.bin").read_bytes() == bytes(range(256))
    assert not (out / pi.STAGING_FAILED).exists()
    (repo / "a.py").write_text("x = 3\n")
    (repo / ".git" / "index.lock").write_text("")
    subprocess.run(["bash", "-c", script], check=False)
    assert "+x = 3" in (out / "patch.diff").read_text()
    assert "index.lock" in (out / pi.STAGING_FAILED).read_text()


def test_pi_patch_ignores_the_git_config_an_agent_sets(tmp_path) -> None:
    repo, out = tmp_path / "testbed", tmp_path / "out"
    out.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "start"], cwd=repo, check=True)
    ran = tmp_path / "ran"
    hook = repo / ".git" / "hooks" / "post-index-change"
    hook.write_text(f"#!/bin/sh\necho hook >> {ran}\n")
    hook.chmod(0o755)
    monitor = tmp_path / "fsmonitor.sh"
    monitor.write_text(f"#!/bin/sh\necho fsmonitor >> {ran}\n")
    monitor.chmod(0o755)
    external = tmp_path / "external.sh"
    external.write_text("#!/bin/sh\necho 'all clean'\n")
    external.chmod(0o755)
    for key, value in (
        ("core.fsmonitor", str(monitor)),
        ("diff.external", str(external)),
        ("diff.clean.textconv", "sed s/2/1/"),
    ):
        subprocess.run(["git", "config", key, value], cwd=repo, check=True)
    (repo / ".gitattributes").write_text("*.py diff=clean\n")
    (repo / "a.py").write_text("x = 2\n")
    script = pi.CAPTURE_PATCH.replace("/testbed", str(repo)).replace("/out", str(out))
    subprocess.run(["bash", "-c", f"git -C {repo} rev-parse HEAD > {out}/orig-head\n{script}"], check=True)
    patch = (out / "patch.diff").read_text()
    assert "-x = 1\n+x = 2\n" in patch and "all clean" not in patch
    assert not ran.exists()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert "hook" in ran.read_text()
    plain = subprocess.run(["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True, check=True)
    assert "all clean" in plain.stdout


def test_pi_image_names() -> None:
    assert pi.instance_image(instance_id="Django__django-11001") == "sweb.eval.x86_64.django__django-11001:latest"
    assert pi.overlay_image(instance_id="django__django-11001") == f"rsi/pi-swe/django__django-11001:{pi.PI_VERSION}"


def _endpoint() -> core.ModelEndpoint:
    return core.ModelEndpoint(
        model_id="qwen-3.8-27b",
        base_url="http://host.docker.internal:11501/v1",
        root_url="http://host.docker.internal:11501",
    )


def test_pi_models_json() -> None:
    qwen = core.ModelSpec(
        name="q",
        source="path:/x",
        model_id="qwen-3.8-27b",
        ctx=262144,
        thinking=True,
        sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "xhigh"},
        thinking_budget_field="thinking_token_budget",
    )
    cfg = pi.pi_models_json(model=qwen, endpoint=_endpoint(), thinking_budgets=False)
    entry = cfg["providers"]["local"]["models"][0]
    assert cfg["providers"]["local"]["baseUrl"] == "http://host.docker.internal:11501/v1"
    assert cfg["providers"]["local"]["api"] == "openai-completions"
    assert entry["contextWindow"] == entry["maxTokens"] == 262144
    assert entry["samplingParams"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    assert entry["reasoning"] is True
    assert entry["compat"] == {
        "thinkingFormat": "chat-template",
        "chatTemplateKwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
    }
    budgeted = pi.pi_models_json(model=qwen, endpoint=_endpoint(), thinking_budgets=True)
    assert budgeted["providers"]["local"]["models"][0]["compat"]["thinkingTokenBudgetField"] == "thinking_token_budget"
    assert pi.pi_model_ref(model=qwen) == "local/qwen-3.8-27b"

    plain = core.ModelSpec(name="p", source="path:/x", model_id="m", max_output_tokens=16384)
    cfg = pi.pi_models_json(model=plain, endpoint=_endpoint(), thinking_budgets=True)
    entry = cfg["providers"]["local"]["models"][0]
    assert entry["maxTokens"] == 16384
    assert "samplingParams" not in entry and "reasoning" not in entry and "compat" not in entry
    assert pi.pi_model_ref(model=plain) == "local/m:off"


def test_pi_check_model() -> None:
    ok = core.ModelSpec(name="q", source="path:/x", model_id="q", thinking=True, chat_template_kwargs={"a": 1})
    assert pi.HARNESS.check_model(model=ok) is None
    off = core.ModelSpec(name="q", source="path:/x", model_id="q", chat_template_kwargs={"enable_thinking": False})
    assert "thinking off" in pi.HARNESS.check_model(model=off)


def test_pi_task_prompt_ignores_child(tmp_path) -> None:
    h = harness.load(name="pi")
    pkg = tmp_path / "child-pkg"
    pkg.mkdir()
    (pkg / "package.json").write_text("{}")
    (pkg / "prompts").mkdir()
    (pkg / "prompts" / "task-rules.md").write_text("CHILD RULE TEXT")
    instance = core.Instance(instance_id="a__b-1", repo="a/b", base_commit="c", problem_statement="FIX IT")
    prompt = h.task_prompt(instance=instance)
    assert "CHILD RULE TEXT" not in prompt
    assert prompt.startswith("FIX IT")
    assert "Task rules" in prompt
    assert "hidden tests" in prompt and "diff" in prompt
    assert "step by step" not in prompt and "minimal" not in prompt


def test_pi_copy_variant_resources(tmp_path) -> None:
    package = tmp_path / "pkg"
    (package / "agents").mkdir(parents=True)
    (package / "settings.json").write_text('{"defaultTools": ["+grep"], "thinkingBudgets": {"medium": 6000}}')
    (package / "mcp.json").write_text('{"mcpServers": {"srv": {"command": "echo"}}}')
    (package / "SYSTEM.md").write_text("You are a careful engineer.")
    (package / "APPEND_SYSTEM.md").write_text("Run the tests before you finish.")
    (package / "agents" / "scout.md").write_text("---\nname: scout\n---\nrecon")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    pi.check.copy_agent_resources(package=package, agent_dir=agent_dir)
    assert (agent_dir / "settings.json").read_text() == (package / "settings.json").read_text()
    assert pi.check.sets_thinking_budgets(package=package)
    assert (agent_dir / "mcp.json").read_text() == '{"mcpServers": {"srv": {"command": "echo"}}}'
    assert (agent_dir / "agents" / "scout.md").read_text().startswith("---")
    assert (agent_dir / "SYSTEM.md").read_text() == "You are a careful engineer."
    assert (agent_dir / "APPEND_SYSTEM.md").read_text() == "Run the tests before you finish."

    bare = tmp_path / "bare"
    bare.mkdir()
    bare_agent = tmp_path / "bare-agent"
    pi.check.copy_agent_resources(package=bare, agent_dir=bare_agent)
    assert list(bare_agent.iterdir()) == []
    assert not pi.check.sets_thinking_budgets(package=bare)
    (bare / "settings.json").write_text('{"defaultTools": ["+grep"]}')
    assert not pi.check.sets_thinking_budgets(package=bare)
    (bare / "settings.json").write_text('{"thinkingBudgets": {"high": 30000}}')
    assert not pi.check.sets_thinking_budgets(package=bare)


def test_pi_stage_runtime_package(tmp_path) -> None:
    package = tmp_path / "child"
    (package / "extensions").mkdir(parents=True)
    (package / "skills" / "verify").mkdir(parents=True)
    (package / "examples").mkdir()
    (package / "package.json").write_text("{}")
    (package / "SYSTEM.md").write_text("sp")
    (package / "APPEND_SYSTEM.md").write_text("more")
    (package / "settings.json").write_text("{}")
    (package / "mcp.json").write_text("{}")
    (package / "extensions" / "gate.ts").write_text("gate")
    (package / "skills" / "verify" / "SKILL.md").write_text("skill")
    (package / "examples" / "new-tool.ts").write_text("example")
    (package / "VARIANT.md").write_text("# card")
    (package / "NOTES.md").write_text("notes")

    dest = tmp_path / "staged"
    pi.check.stage_runtime_package(package=package, dest=dest)

    staged = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert staged == {
        "package.json",
        "SYSTEM.md",
        "APPEND_SYSTEM.md",
        "settings.json",
        "mcp.json",
        "extensions/gate.ts",
        "skills/verify/SKILL.md",
    }
    assert (dest / "extensions" / "gate.ts").read_text() == "gate"
    assert not (dest / "VARIANT.md").exists()
    assert not (dest / "NOTES.md").exists()
    assert not (dest / "examples").exists()


def test_pi_cleanup_kills_live_containers() -> None:
    calls: list[str] = []

    class Fake:
        short_id = "fake"
        status = "exited"

        def reload(self) -> None:
            calls.append("reload")

        def kill(self) -> None:
            calls.append("kill")

        def remove(self, force: bool) -> None:
            calls.append("remove")

    pi._live.append(Fake())
    pi.HARNESS.cleanup()
    assert calls == ["kill", "reload", "remove"]
    assert pi._live == []
    pi.HARNESS.cleanup()


def test_pi_contract_digest(tmp_path) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "package").mkdir()
    (tmp_path / "task-rules.md").write_text("rules")
    (tmp_path / "eval" / "metrics.ts").write_text("metrics")
    (tmp_path / "package" / "package.json").write_text("{}")
    h = pi.PiHarness(
        package_dir=tmp_path / "package", eval_dir=tmp_path / "eval", task_rules=tmp_path / "task-rules.md"
    )
    digest = h.contract()
    assert len(digest) == 12 and h.contract() == digest
    (tmp_path / "package" / ".DS_Store").write_bytes(b"\0")
    assert h.contract() == digest
    assert (
        pi.PiHarness(
            package_dir=h.package_dir, eval_dir=h.eval_dir, task_rules=h.task_rules, version="1.5.0"
        ).contract()
        != digest
    )
    seen = {digest}
    for change in (
        lambda: (tmp_path / "task-rules.md").write_text("rules, reworded"),
        lambda: (tmp_path / "eval" / "metrics.ts").write_text("metrics, changed"),
        lambda: (tmp_path / "eval" / "metrics.ts").rename(tmp_path / "eval" / "timing.ts"),
        lambda: (tmp_path / "package" / "SYSTEM.md").write_text("a system prompt"),
    ):
        change()
        seen.add(h.contract())
    assert len(seen) == 5
    assert len(pi.HARNESS.contract()) == 12
