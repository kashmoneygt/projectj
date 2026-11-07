import json
import shutil
import subprocess
from pathlib import Path

import docker
import pytest
import requests

from rsi import cards, core, folders, harness, paths

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
    assert not (h.package_dir / "prompts").exists()


def test_pi_install_extensions(tmp_path: Path) -> None:
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

    pi._flush_partial_patch(container=_Container(), workdir="/testbed")
    assert calls == [["bash", "-lc", pi.capture_patch(workdir="/testbed")]]


def test_pi_patch_keeps_new_and_binary_files(tmp_path: Path) -> None:
    repo, out = tmp_path / "testbed", tmp_path / "out"
    out.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "start"], cwd=repo, check=True)
    fresh = tmp_path / "fresh"
    shutil.copytree(repo, fresh)
    script = pi.capture_patch(workdir=str(repo)).replace("/out", str(out))
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


def test_pi_patch_ignores_the_git_config_an_agent_sets(tmp_path: Path) -> None:
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
    script = pi.capture_patch(workdir=str(repo)).replace("/out", str(out))
    subprocess.run(["bash", "-c", f"git -C {repo} rev-parse HEAD > {out}/orig-head\n{script}"], check=True)
    patch = (out / "patch.diff").read_text()
    assert "-x = 1\n+x = 2\n" in patch and "all clean" not in patch
    assert not ran.exists()
    # change the file again so plain git add has to update the index and run the hook
    (repo / "a.py").write_text("x = 3\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert "hook" in ran.read_text()
    plain = subprocess.run(["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True, check=True)
    assert "all clean" in plain.stdout


def test_pi_overlay_image_name() -> None:
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
    )
    cfg = pi.pi_models_json(model=qwen, endpoint=_endpoint())
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
    assert pi.pi_model_ref(model=qwen) == "local/qwen-3.8-27b"

    plain = core.ModelSpec(name="p", source="path:/x", model_id="m", max_output_tokens=16384)
    cfg = pi.pi_models_json(model=plain, endpoint=_endpoint())
    entry = cfg["providers"]["local"]["models"][0]
    assert entry["maxTokens"] == 16384
    assert "samplingParams" not in entry and "reasoning" not in entry and "compat" not in entry
    assert pi.pi_model_ref(model=plain) == "local/m:off"


def test_pi_check_model() -> None:
    ok = core.ModelSpec(name="q", source="path:/x", model_id="q", thinking=True, chat_template_kwargs={"a": 1})
    assert pi.HARNESS.check_model(model=ok) is None
    off = core.ModelSpec(name="q", source="path:/x", model_id="q", chat_template_kwargs={"enable_thinking": False})
    assert "thinking off" in pi.HARNESS.check_model(model=off)


def test_pi_task_script_works_in_the_tasks_folder_and_never_names_the_task(tmp_path: Path) -> None:
    instance = core.Instance(
        instance_id="django__django-11001", repo="django/django", image="img", workdir="/work/repo", task="Fix it."
    )
    model = core.ModelSpec(name="q", source="path:/x", model_id="q")
    ctx = harness.RunContext(
        instance=instance, out_dir=tmp_path, model=model, endpoint=_endpoint(), budget=core.Budget(timeout_s=60)
    )
    script = pi._task_script(ctx=ctx)
    assert script.startswith("set -u\ncd /work/repo\n") and "--session-id task " in script
    assert "cd /work/repo && start=" in script
    assert "django" not in script


def test_pi_copy_variant_resources(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    (package / "agents").mkdir(parents=True)
    (package / "settings.json").write_text('{"defaultTools": ["+grep"]}')
    (package / "mcp.json").write_text('{"mcpServers": {"srv": {"command": "echo"}}}')
    (package / "SYSTEM.md").write_text("You are a careful engineer.")
    (package / "APPEND_SYSTEM.md").write_text("Run the tests before you finish.")
    (package / "agents" / "scout.md").write_text("---\nname: scout\n---\nrecon")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    pi.check.copy_agent_resources(package=package, agent_dir=agent_dir)
    assert (agent_dir / "settings.json").read_text() == (package / "settings.json").read_text()
    assert (agent_dir / "mcp.json").read_text() == '{"mcpServers": {"srv": {"command": "echo"}}}'
    assert (agent_dir / "agents" / "scout.md").read_text().startswith("---")
    assert (agent_dir / "SYSTEM.md").read_text() == "You are a careful engineer."
    assert (agent_dir / "APPEND_SYSTEM.md").read_text() == "Run the tests before you finish."

    bare = tmp_path / "bare"
    bare.mkdir()
    bare_agent = tmp_path / "bare-agent"
    pi.check.copy_agent_resources(package=bare, agent_dir=bare_agent)
    assert list(bare_agent.iterdir()) == []


def test_pi_stage_runtime_package(tmp_path: Path) -> None:
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


class _CheckContainer:
    """A check's container that ends the way a test says"""

    def __init__(self, ends: str, removed: list[bool]) -> None:
        self.ends = ends
        self.removed = removed
        self.status = "running"
        self.short_id = "c1"

    def wait(self, timeout: int) -> dict:
        if self.ends == "interrupt":
            raise KeyboardInterrupt
        if self.ends == "timeout":
            raise requests.exceptions.ReadTimeout("Read timed out")
        return {"StatusCode": {"ok": 0, "problem": 1, "unavailable": 2}[self.ends]}

    def logs(self, tail: int | None = None) -> bytes:
        printed = {
            "ok": "ok: the package stayed out of the way",
            "problem": "problem: the agent answered, and the package made it go on",
            "unavailable": "unavailable: pi got no response from the model",
        }
        return f"{printed.get(self.ends, 'pi is still going')}\n".encode()

    def kill(self) -> None:
        self.status = "exited"

    def reload(self) -> None:
        pass

    def remove(self, force: bool) -> None:
        self.removed.append(force)


class _CheckContainers:
    def __init__(self, container: _CheckContainer | None) -> None:
        self.container = container

    def run(self, image: str, **kwargs: object) -> _CheckContainer:
        if self.container is None:
            raise docker.errors.DockerException("the docker daemon isn't running")
        return self.container


@pytest.mark.parametrize(
    ("ends", "found"),
    [
        ("ok", (0, "ok: the package stayed out of the way\n")),
        ("problem", (1, "problem: the agent answered, and the package made it go on\n")),
        ("timeout", (-1, "problem: the question did not finish within 60 s")),
        ("unavailable", "the question couldn't run: pi got no response from the model"),
        ("docker", "the question couldn't run: docker: DockerException: the docker daemon isn't running"),
        # a question check calls the model, so it mustn't keep going after ctrl-c
        ("interrupt", "KeyboardInterrupt"),
    ],
)
def test_pi_check_reads_how_its_container_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ends: str, found: tuple[int, str] | str
) -> None:
    removed: list[bool] = []
    container = None if ends == "docker" else _CheckContainer(ends=ends, removed=removed)

    class Client:
        containers = _CheckContainers(container=container)

    monkeypatch.setattr(pi.docker, "from_env", lambda: Client())
    monkeypatch.setattr(pi, "_ensure_image", lambda client, tag, dockerfile: None)
    monkeypatch.setattr(pi, "_remove_stale_container", lambda client, name: None)
    command = {"command": ["true"], "volumes": {}, "network": "rsi-sandbox-11501", "timeout_s": 60}
    log_path = tmp_path / "question.log"
    if isinstance(found, tuple):
        assert pi._run_check(what="the question", name="rsi-question", log_path=log_path, **command) == found
    elif found == "KeyboardInterrupt":
        with pytest.raises(KeyboardInterrupt):
            pi._run_check(what="the question", name="rsi-question", log_path=log_path, **command)
    else:
        with pytest.raises(harness.CheckUnavailable) as raised:
            pi._run_check(what="the question", name="rsi-question", log_path=log_path, **command)
        assert str(raised.value) == found
    # a container that started is always removed, and its log kept
    assert removed == ([] if container is None else [True]) and pi._live == []
    assert log_path.is_file() == (container is not None)


def test_pi_question_check_asks_each_question_until_one_has_problems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[tuple[str, list[str], int]] = []
    printed = {"javascript": "ok: fine\n", "python": "problem: the run changed shipping.py\n"}

    def run_check(
        what: str,
        name: str,
        command: list[str],
        volumes: dict[str, dict[str, str]],
        network: str,
        timeout_s: int,
        log_path: Path,
    ) -> tuple[int, str]:
        question = Path(next(source for source, mount in volumes.items() if mount["bind"] == "/opt/rsi/question"))
        asked.append((question.name, command, timeout_s))
        output = printed[question.name]
        return (1 if output.startswith("problem") else 0), output

    monkeypatch.setattr(pi, "_run_check", run_check)
    monkeypatch.setattr(pi, "sandbox_network", lambda client, endpoint: "rsi-sandbox-11501")
    monkeypatch.setattr(pi.docker, "from_env", lambda: None)
    model = core.ModelSpec(name="q", source="s", model_id="qwen", thinking=True)
    endpoint = core.ModelEndpoint(model_id="qwen", base_url="http://host.docker.internal:11501/v1")

    def ask() -> list[str]:
        return pi.HARNESS.question_problems(
            variant=tmp_path / "v", out_dir=tmp_path / "out", model=model, endpoint=endpoint
        )

    assert ask() == ["python question: the run changed shipping.py"]
    assert [question for question, _, _ in asked] == ["javascript", "python"]
    _, command, timeout_s = asked[0]
    assert "--tries=2" in command and "--timeout=540" in command and timeout_s == 2 * (540 + 120) + 120
    # it stops at the first question with problems
    printed["javascript"] = "problem: the agent never answered\n"
    asked.clear()
    assert ask() == ["javascript question: the agent never answered"]
    assert [question for question, _, _ in asked] == ["javascript"]


def test_pi_contract_digest(tmp_path: Path) -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "package").mkdir()
    (tmp_path / "eval" / "metrics.ts").write_text("metrics")
    (tmp_path / "package" / "package.json").write_text("{}")
    h = pi.PiHarness(package_dir=tmp_path / "package", eval_dir=tmp_path / "eval")
    digest = h.contract()
    assert len(digest) == 12 and h.contract() == digest
    (tmp_path / "package" / ".DS_Store").write_bytes(b"\0")
    assert h.contract() == digest
    assert pi.PiHarness(package_dir=h.package_dir, eval_dir=h.eval_dir, version="1.5.0").contract() != digest
    seen = {digest}
    for change in (
        lambda: (tmp_path / "eval" / "metrics.ts").write_text("metrics, changed"),
        lambda: (tmp_path / "eval" / "metrics.ts").rename(tmp_path / "eval" / "timing.ts"),
        lambda: (tmp_path / "package" / "SYSTEM.md").write_text("a system prompt"),
    ):
        change()
        seen.add(h.contract())
    assert len(seen) == 4
    assert len(pi.HARNESS.contract()) == 12


def test_every_question_is_about_a_small_repo_that_needs_no_change() -> None:
    questions = sorted(d.name for d in pi.QUESTIONS_DIR.iterdir() if d.is_dir())
    assert questions == ["javascript", "python"] and pi.PRACTICE_QUESTION.name == "javascript"
    for name in questions:
        question = pi.QUESTIONS_DIR / name
        assert (question / "question.md").read_text().strip().endswith("?")
        assert (question / "repo" / "README.md").is_file()
    # npm install leaves a committed lockfile as it is, so a reflexive install changes nothing
    assert (pi.PRACTICE_QUESTION / "repo" / "package-lock.json").is_file()


def test_the_teacher_container_installs_both_helpers() -> None:
    assert "check_variant.py --terms /opt/rsi/terms.txt --manifest /opt/harness/package.json" in pi.CHECK_COMMAND
    assert "try_variant.py" in pi.TRY_COMMAND
    assert subprocess.run(["sh", "-n", "-c", pi.TRY_COMMAND], check=False).returncode == 0


def test_records_activations_looks_for_the_acted_entry_in_code(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "extensions").mkdir(parents=True)
    (package / "APPEND_SYSTEM.md").write_text('Never call pi.appendEntry("acted").\n')
    (package / cards.FILE).write_text('gate\nWhat: it calls pi.appendEntry("acted", ...)\n')
    (package / "extensions" / "gate.ts").write_text('pi.appendEntry("noted", { what: "x" });\n')
    assert not pi.HARNESS.records_activations(package=package)
    (package / "extensions" / "gate.ts").write_text("pi.appendEntry(\n  'acted', { what: 'blocked a call' });\n")
    assert pi.HARNESS.records_activations(package=package)
