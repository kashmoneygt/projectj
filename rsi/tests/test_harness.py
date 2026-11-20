import json
import os
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
    assert h.version == "1.5.2"
    assert h.package_dir.name == "package"


def test_pi_package_manifest() -> None:
    h = harness.load(name="pi")
    manifest = json.loads((h.package_dir / "package.json").read_text())
    assert "pi-package" in manifest["keywords"]
    assert manifest["pi"]["extensions"] == ["extensions/*.ts"]
    assert sorted(p.name for p in h.package_dir.iterdir()) == ["package.json"]
    assert h.doc.is_file()
    assert sorted(p.name for p in h.eval_dir.iterdir()) == ["budget.ts", "metrics.ts"]
    assert not (h.package_dir / "prompts").exists()


def test_pi_install_extensions(tmp_path: Path) -> None:
    source = tmp_path / "eval"
    source.mkdir()
    (source / "metrics.ts").write_text("metrics")
    agent_dir = tmp_path / "agent"
    pi._install_extensions(source_dir=source, agent_dir=agent_dir)
    assert (agent_dir / "extensions" / "metrics.ts").read_text() == "metrics"


def _patch_scripts(out: Path, snapshot: Path) -> tuple[str, str]:
    """Scripts that record the workdir at the start and diff it at the end, writing to out instead of /out"""
    start = pi.patch_start().replace("/out", str(out)).replace(pi.SNAPSHOT, str(snapshot))
    return start, pi.capture_patch().replace("/out", str(out)).replace(pi.SNAPSHOT, str(snapshot))


def _repo(path: Path) -> list[str]:
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "a.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "-A"], cwd=path, check=True)
    subprocess.run([*git, "commit", "-qm", "start"], cwd=path, check=True)
    return git


def test_pi_patch_keeps_new_and_binary_files(tmp_path: Path) -> None:
    repo, out = tmp_path / "testbed", tmp_path / "out"
    out.mkdir()
    git = _repo(path=repo)
    # the image build left an untracked file, which isn't the agent's change
    (repo / "=0.9.0").write_text("")
    fresh = tmp_path / "fresh"
    shutil.copytree(repo, fresh)
    start, capture = _patch_scripts(out=out, snapshot=tmp_path / "snapshot.git")
    subprocess.run(["bash", "-c", start], cwd=repo, check=True)
    assert (out / "patch-start").read_text().startswith("repo ")
    subprocess.run(["bash", "-c", capture], cwd=repo, check=True)
    assert (out / "patch.diff").read_text() == ""
    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.py").write_text("y = 1\n")
    (repo / "blob.bin").write_bytes(bytes(range(256)))
    subprocess.run([*git, "commit", "-qam", "the agent committed"], cwd=repo, check=True)
    subprocess.run(["bash", "-c", capture], cwd=repo, check=True)
    subprocess.run(["git", "apply", str(out / "patch.diff")], cwd=fresh, check=True)
    assert (fresh / "a.py").read_text() == "x = 2\n" and (fresh / "new.py").read_text() == "y = 1\n"
    assert (fresh / "blob.bin").read_bytes() == bytes(range(256))
    assert not (out / pi.STAGING_FAILED).exists()
    # capture leaves the repo's own git index as the agent left it, since grading may use it
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=repo, capture_output=True, text=True, check=True
    )
    assert staged.stdout == ""
    (repo / "a.py").write_text("x = 3\n")
    # an index.lock the agent left doesn't block the diff, since capture stages into its own index file
    (repo / ".git" / "index.lock").write_text("")
    subprocess.run(["bash", "-c", capture], cwd=repo, check=True)
    assert "+x = 3" in (out / "patch.diff").read_text()
    assert not (out / pi.STAGING_FAILED).exists()


def test_pi_patch_of_a_folder_that_isnt_a_repo_leaves_it_without_one(tmp_path: Path) -> None:
    folder, out, snapshot = tmp_path / "app", tmp_path / "out", tmp_path / "snapshot.git"
    folder.mkdir()
    out.mkdir()
    (folder / "data.csv").write_text("a,b\n1,2\n")
    start, capture = _patch_scripts(out=out, snapshot=snapshot)
    subprocess.run(["bash", "-c", start], cwd=folder, check=True)
    assert (out / "patch-start").read_text() == "snapshot HEAD\n"
    (folder / "data.csv").write_text("a,b\n1,3\n")
    (folder / "tool.py").write_text("print(1)\n")
    subprocess.run(["bash", "-c", capture], cwd=folder, check=True)
    patch = (out / "patch.diff").read_text()
    assert "+1,3" in patch and "b/tool.py" in patch
    assert sorted(p.name for p in folder.iterdir()) == ["data.csv", "tool.py"]


def test_pi_patch_ignores_the_git_config_an_agent_sets(tmp_path: Path) -> None:
    repo, out = tmp_path / "testbed", tmp_path / "out"
    out.mkdir()
    _repo(path=repo)
    start, capture = _patch_scripts(out=out, snapshot=tmp_path / "snapshot.git")
    subprocess.run(["bash", "-c", start], cwd=repo, check=True)
    # then the agent sets up git hooks and config to run its own code or hide its changes
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
    subprocess.run(["bash", "-c", capture], cwd=repo, check=True)
    patch = (out / "patch.diff").read_text()
    assert "-x = 1\n+x = 2\n" in patch and "all clean" not in patch
    assert not ran.exists()
    # change the file again so plain git add has to update the index and run the hook
    (repo / "a.py").write_text("x = 3\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert "hook" in ran.read_text()
    plain = subprocess.run(["git", "diff", "--cached"], cwd=repo, capture_output=True, text=True, check=True)
    assert "all clean" in plain.stdout


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


def test_pi_agent_script_never_names_the_task(tmp_path: Path) -> None:
    instance = core.Instance(
        instance_id="django__django-11001", repo="django/django", image="img", workdir="/work/repo", task="Fix it."
    )
    model = core.ModelSpec(name="q", source="path:/x", model_id="q")
    ctx = harness.RunContext(
        instance=instance, out_dir=tmp_path, model=model, endpoint=_endpoint(), budget=core.Budget(timeout_s=60)
    )
    script = pi._agent_script(ctx=ctx)
    # pi runs on its own node binary, and its bin folder goes last on PATH so the task's own tools come first
    assert script.startswith(f'export PATH="$PATH:{pi.PI_HOME}/bin"\n{pi.PI_HOME}/bin/node {pi.PI_CLI} --print ')
    assert f"< /out/{harness.TASK_FILE} > /out/events.jsonl" in script and "--session-id task " in script
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
    (package / "variant.json").write_text("{}")
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
    assert not (dest / "variant.json").exists()
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
    """Fake check container that ends the way the test asks"""

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
        # a question check calls the model, so ctrl-c must stop it
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
    # asking stops at the first question with problems
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
    # npm install keeps a committed lockfile as is, so an agent that runs it out of habit changes nothing
    assert (pi.PRACTICE_QUESTION / "repo" / "package-lock.json").is_file()


def test_the_teacher_container_installs_both_helpers() -> None:
    assert "check_variant.py --terms /opt/rsi/terms.txt --manifest /opt/harness/package.json" in pi.CHECK_COMMAND
    assert "try_variant.py" in pi.TRY_COMMAND
    assert subprocess.run(["sh", "-n", "-c", pi.TRY_COMMAND], check=False).returncode == 0


def test_check_variant_gets_the_banned_terms_hashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mounted: list[str] = []

    def run_check(
        what: str,
        name: str,
        command: list[str],
        volumes: dict[str, dict[str, str]],
        network: str,
        timeout_s: int,
        log_path: Path,
    ) -> tuple[int, str]:
        mounted.append(
            next(Path(source).read_text() for source, mount in volumes.items() if mount["bind"].endswith("terms.txt"))
        )
        return 0, "ok: fine\n"

    monkeypatch.setattr(pi, "_run_check", run_check)
    package = tmp_path / "package"
    package.mkdir()
    (package / "APPEND_SYSTEM.md").write_text("Work like on a Termigen task.\n")
    assert pi.HARNESS.validate_variant(variant=package, out_dir=tmp_path / "check", terms=("termigen",)) == []
    # the teacher's container holds the same file, so it mustn't name the benchmark, yet the check still finds it
    (terms,) = mounted
    assert "termigen" not in terms.lower()
    found = pi.check.term_problems(package=package, terms=tuple(terms.splitlines()))
    assert found == ["APPEND_SYSTEM.md names the workload or its protocol ('termigen')"]


def test_records_activations_looks_for_the_acted_entry_in_code(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "extensions").mkdir(parents=True)
    (package / "APPEND_SYSTEM.md").write_text('Never call pi.appendEntry("acted").\n')
    (package / cards.FILE).write_text('gate\nWhat: it calls pi.appendEntry("acted", ...)\n')
    (package / "extensions" / "gate.ts").write_text('pi.appendEntry("noted", { what: "x" });\n')
    assert not pi.HARNESS.records_activations(package=package)
    (package / "extensions" / "gate.ts").write_text("pi.appendEntry(\n  'acted', { what: 'blocked a call' });\n")
    assert pi.HARNESS.records_activations(package=package)


class _TaskContainer:
    name = "rsi-task"

    def __init__(self) -> None:
        self.attrs = {"Config": {"WorkingDir": "/app"}, "NetworkSettings": {"Networks": {"net": {"IPAddress": "ip"}}}}

    def reload(self) -> None:
        pass

    def exec_run(self, cmd: list[str], user: str = "") -> None:
        pass


def _record_task(monkeypatch: pytest.MonkeyPatch, setup_code: int = 0) -> list[str]:
    """Record each part of a task run, in order, instead of running it in Docker"""
    done: list[str] = []
    monkeypatch.setattr(pi, "_start", lambda client, instance, network, volumes, name: _TaskContainer())
    monkeypatch.setattr(pi, "_set_internet", lambda client, container, on: done.append(f"internet {on}"))
    monkeypatch.setattr(pi, "_finish_container", lambda container: done.append("removed"))

    def run(container: _TaskContainer, script: str, workdir: str, timeout_s: float) -> int:
        part = "setup" if "setup.log" in script else "start" if "g init" in script else "diff"
        done.append(f"{part} in {workdir}")
        return setup_code if part == "setup" else 0

    monkeypatch.setattr(pi, "_exec", run)

    def verify(client: object, container: _TaskContainer, verifier: core.Verifier, workdir: str, out_dir: Path) -> None:
        done.append(f"grade online={verifier.internet}")

    monkeypatch.setattr(pi, "_verify", verify)
    return done


def _graded_instance() -> core.Instance:
    return core.Instance(
        instance_id="src.task",
        repo="task",
        image="img",
        workdir="",
        task="Build it.",
        setup="bash setup.sh",
        internet=False,
        verifier=core.Verifier(tests="/host/tests", command="/tests/test.sh", internet=True),
    )


def test_a_task_is_worked_with_its_own_network_and_graded_where_it_was_worked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    done = _record_task(monkeypatch=monkeypatch)

    def act(*, container: _TaskContainer, workdir: str) -> tuple[str, str | None]:
        done.append(f"agent in {workdir}")
        # the agent swaps result.json for a symlink out of the run folder, which the host would write through
        (tmp_path / "result.json").symlink_to(tmp_path.parent / "elsewhere.json")
        return "ok", None

    found = pi.run_task(
        client=None, instance=_graded_instance(), out_dir=tmp_path, network="net", volumes={}, act=act, name="n"
    )
    assert found[0] == "ok" and found[1] is None and found[3] == "ip"
    assert done == [
        "internet False",
        "setup in /app",
        "start in /app",
        "agent in /app",
        "diff in /app",
        "grade online=True",
        "removed",
    ]
    assert not (tmp_path / "result.json").is_symlink()


def test_a_task_whose_setup_fails_never_reaches_the_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    done = _record_task(monkeypatch=monkeypatch, setup_code=2)
    (tmp_path / "setup.log").write_text("pip: no such package\n")
    status, error, _, _ = pi.run_task(
        client=None,
        instance=_graded_instance(),
        out_dir=tmp_path,
        network="net",
        volumes={},
        act=lambda *, container, workdir: ("ok", None),
        name="n",
    )
    assert status == "error" and "setup exited 2: pip: no such package" in error
    assert done[-1] == "removed" and not any(step.startswith(("agent", "grade")) for step in done)


@pytest.mark.parametrize(
    ("left", "code", "failed"),
    [(["reward.txt"], 0, False), ([], 0, True), ([], None, False)],
    ids=["graded", "left-no-reward", "ran-out-of-time"],
)
def test_grading_that_leaves_no_reward_fails_unless_it_ran_out_of_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, left: list[str], code: int | None, failed: bool
) -> None:
    monkeypatch.setattr(pi, "_set_internet", lambda client, container, on: None)
    monkeypatch.setattr(pi, "_put", lambda container, source, dest: None)
    monkeypatch.setattr(pi, "_exec", lambda container, script, workdir, timeout_s, environment: code)

    def fetch(container: _TaskContainer, source: str, dest: Path) -> None:
        dest.mkdir()
        (dest / "test-stdout.txt").write_text("collected 3 items\n")
        for name in left:
            (dest / name).write_text("1\n")

    monkeypatch.setattr(pi, "_fetch", fetch)
    verifier = core.Verifier(tests="/host/tests", command="/tests/test.sh")
    why = pi._verify(client=None, container=_TaskContainer(), verifier=verifier, workdir="/app", out_dir=tmp_path)
    assert (why is not None) == failed
    if failed:
        # the teacher sees the run's error, so the error must not say what the grader printed or checks for
        assert why == "grading exited 0 without a result"
    grading = json.loads((tmp_path / harness.GRADING_FILE).read_text())
    assert grading["timed_out"] == (code is None) and grading["exit_code"] == code


def test_only_files_stay_in_a_run_folder_the_host_reads_and_writes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "verifier").mkdir(parents=True)
    (run / "patch.diff").write_text("+x\n")
    (run / "verifier" / "reward.txt").write_text("0\n")
    (tmp_path / "planted").mkdir()
    # symlinks the agent could leave to forge a result or reach host files, and a pipe that would block a read
    (run / "pi.err").symlink_to(Path.home() / ".ssh" / "id_rsa")
    (run / "verifier-2").symlink_to(tmp_path / "planted", target_is_directory=True)
    os.mkfifo(run / "events.jsonl")
    pi._only_files(folder=run)
    left = sorted(p.relative_to(run).as_posix() for p in run.rglob("*"))
    assert left == ["patch.diff", "verifier", "verifier/reward.txt"] and (tmp_path / "planted").is_dir()
