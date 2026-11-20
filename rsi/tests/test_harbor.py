import json
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from rsi import benchmarks, folders, harness, paths

harbor = folders.module(root=paths.benchmarks_dir(), name="harbor", file="harbor.py", kind="Harbor tasks")


def _source(
    name: str = "src",
    internet: bool | None = None,
    leave_out: str = "",
    pick: Callable[[], Iterable[str]] | None = None,
    request: str = "{task}",
    grader: str = "",
) -> harbor.Source:
    return harbor.Source(
        name=name,
        kind="tools",
        host=harbor.HF,
        repo="o/r",
        revision="a" * 40,
        folder="tasks",
        internet=internet,
        leave_out=leave_out,
        pick=pick,
        request=request,
        grader=grader,
    )


def _task(source: harbor.Source, name: str, config: str, environment: dict[str, str]) -> Path:
    folder = source.task_dir(name=name)
    (folder / "environment").mkdir(parents=True)
    (folder / "tests").mkdir()
    (folder / "task.toml").write_text(config)
    (folder / "instruction.md").write_text("Build the tool.\n")
    for file, text in environment.items():
        (folder / "environment" / file).write_text(text)
    return folder


def test_a_task_is_run_the_way_harbor_runs_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    monkeypatch.setenv("TOKEN", "t0k")
    source = _source()
    config = (
        '[environment]\ndocker_image = "img:1"\nworkdir = "/app"\nmemory_mb = 2048\ncpus = 2\n'
        '[environment.env]\nKEY = "${TOKEN}"\nOTHER = "${MISSING:-fallback}"\n'
        '[agent]\nnetwork_mode = "no-network"\n'
        "[verifier]\ntimeout_sec = 120.0\n"
    )
    folder = _task(source=source, name="t1", config=config, environment={})
    (folder / "solution").mkdir()
    (folder / "solution" / "solve.sh").write_text("true\n")
    instance = source.instance(name="t1")
    assert (instance.instance_id, instance.image, instance.workdir) == ("src.t1", "img:1", "/app")
    assert instance.task == "Build the tool." and instance.setup == ""
    assert (instance.cpus, instance.memory_mb) == (2.0, 2048)
    assert instance.env == {"KEY": "t0k", "OTHER": "fallback"}
    # the agent runs offline, but grading keeps the task's network setting
    assert not instance.internet and instance.verifier.internet
    assert instance.verifier.timeout_s == 120 and instance.verifier.tests == str(folder / "tests")
    assert instance.solution == str(folder / "solution") and instance.kind == "tools"


def test_a_task_without_an_image_is_built_and_a_source_can_keep_it_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    source = _source(internet=False)
    _task(source=source, name="t2", config="[environment]\nallow_internet = true\n", environment={"Dockerfile": "x"})
    instance = source.instance(name="t2")
    assert instance.image == "rsi/src-aaaaaaaa:t2" and instance.workdir == ""
    assert not instance.internet and not instance.verifier.internet and instance.solution == ""
    online = _source().instance(name="t2")
    assert online.internet and online.verifier.internet


def _graded(folder: Path, grading: dict | None, files: dict[str, str]) -> Path:
    (folder / harness.VERIFIER_DIR).mkdir(parents=True)
    if grading is not None:
        (folder / harness.GRADING_FILE).write_text(json.dumps(grading))
    for name, text in files.items():
        (folder / harness.VERIFIER_DIR / name).write_text(text)
    return folder


@pytest.mark.parametrize(
    ("grading", "files", "resolved"),
    [
        ({"timed_out": False}, {"reward.txt": "1\n"}, True),
        ({"timed_out": False}, {"reward.txt": "0\n"}, False),
        ({"timed_out": False}, {"reward.json": '{"reward": 0.5}', "reward.txt": "1"}, False),
        ({"timed_out": True, "seconds": 600}, {}, False),
        ({"timed_out": False}, {}, None),
        (None, {"reward.txt": "1\n"}, None),
    ],
    ids=["passed", "failed", "json-first", "ran-out-of-time", "no-reward", "never-graded"],
)
def test_a_verdict_comes_from_what_grading_left(
    tmp_path: Path, grading: dict | None, files: dict[str, str], resolved: bool | None
) -> None:
    bench = harbor.HarborBenchmark(name="b", sources=(_source(),), work="w", terms=())
    verdict = bench.verdict(task_dir=_graded(folder=tmp_path / "task", grading=grading, files=files))
    assert (verdict.resolved if verdict else None) == resolved
    if grading and grading.get("timed_out"):
        assert verdict.log.startswith("grading ran out of time after 600 s")


def test_a_sample_takes_an_equal_share_of_each_kind_split_between_its_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    first = _source(name="a", leave_out=r"skip")
    second = _source(name="b", pick=lambda: ["b-3", "b-1", "b-2"])
    third = harbor.Source(name="c", kind="data", host=harbor.HF, repo="o/d", revision="b" * 40, folder="tasks")
    names = {"a": ["a-1", "a-2", "a-skip", "a-3"], "b": ["b-1", "b-2", "b-3", "b-4"], "c": ["c-1", "c-2", "c-3"]}
    for source in (first, second, third):
        source.cache.mkdir(parents=True)
        (source.cache / "names.json").write_text(json.dumps(names[source.name]))
        for name in names[source.name]:
            _task(source=source, name=name, config='[environment]\ndocker_image = "i"\n', environment={})
    bench = harbor.HarborBenchmark(name="mix", sources=(first, second, third), work="w", terms=())
    assert first.names() == ["a-1", "a-2", "a-3"] and second.names() == ["b-1", "b-2", "b-3"]
    picked = [i.instance_id for i in bench.select(tasks=4, seed=7)]
    assert sorted(i.split(".")[0] for i in picked) == ["a", "b", "c", "c"]
    assert picked == [i.instance_id for i in bench.select(tasks=4, seed=7)]
    assert [i.instance_id for i in bench.by_ids(ids=["b.b-2"])] == ["b.b-2"]
    with pytest.raises(ValueError, match="unknown task id"):
        bench.by_ids(ids=["x.y"])


def test_a_source_can_ask_for_its_tasks_its_own_way_and_grade_them_with_its_own_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    grader = tmp_path / "grader"
    for part, file in (("tests", "test.sh"), ("solution", "solve.sh")):
        (grader / part).mkdir(parents=True)
        (grader / part / file).write_text(f"# our {file}\n")
    asked = "Someone asked:\n\n{task}\n\nDon't change anything."
    plain = _source(request=asked, grader=str(grader)).changes
    # caches and dotfiles show up on their own, so they don't change the digest or get copied into the task
    (grader / "tests" / "__pycache__").mkdir()
    (grader / "tests" / "__pycache__" / "grade.cpython-315.pyc").write_bytes(b"\x00")
    (grader / ".DS_Store").write_bytes(b"\x00")
    source = _source(request=asked, grader=str(grader))
    assert source.changes == plain
    task = _task(source=source, name="t1", config='[environment]\ndocker_image = "i"\n', environment={})
    (task / "tests" / "test.sh").write_text("# its own grader\n")
    (task / "solution").mkdir()
    (task / "solution" / "fix.patch").write_text("the fix\n")
    instance = source.instance(name="t1")
    assert instance.task == "Someone asked:\n\nBuild the tool.\n\nDon't change anything."
    tests, solution = Path(instance.verifier.tests), Path(instance.solution)
    assert (tests / "test.sh").read_text() == "# our test.sh\n" and (solution / "solve.sh").is_file()
    assert not (tests / "__pycache__").exists() and not (tests.parent / ".DS_Store").exists()
    assert (
        (tests / "reference" / "fix.patch").read_text()
        == "the fix\n"
        == (solution / "reference" / "fix.patch").read_text()
    )
    bench = harbor.HarborBenchmark(name="b", sources=(source,), work="w", terms=("bench",))
    assert bench.protocol_terms == ("bench", "Someone asked:", "Don't change anything.")
    # the dataset id includes a hash of our grader and request, so runs made with another grader don't compare
    assert bench.dataset.dataset_id.endswith(f" as {source.changes}")
    (grader / "tests" / "test.sh").write_text("# our grader, changed\n")
    changed = _source(request=asked, grader=str(grader))
    assert changed.changes != source.changes and Path(changed.instance(name="t1").verifier.tests) != tests


def test_the_training_mix_bans_its_sources_by_name_and_runs_them_offline() -> None:
    bench = benchmarks.get(name="coding-mix")
    assert [s.name for s in bench.sources] == [
        "rebench-fix",
        "rebench-build",
        "rebench-where",
        "termigen",
        "dacode",
        "bird",
    ]
    assert len({s.kind for s in bench.sources}) == 5
    assert {"rebench", "termigen", "dacode"} <= set(benchmarks.banned_terms(benchmark=bench))
    assert "Please resolve this issue:" in bench.protocol_terms
    assert all(s.internet is False for s in bench.sources)
    assert bench.dataset.dataset_id.count(" offline") == len(bench.sources)


grade = folders.module(root=paths.benchmarks_dir() / "coding_mix" / "where", name="tests", file="grade.py")


def test_an_answer_names_a_file_by_its_path_or_an_end_of_it_no_other_file_has() -> None:
    files = ["src/app/parser.py", "src/app/util.py", "lib/util.py", "tests/test_parser.py", "scripts/test"]
    reply = (
        "It's in `app/parser.py` (see parser.py:42), not util.py. A test would go in tests/test_parser.py, and the "
        "test script needs no change."
    )
    assert grade.named(reply=reply, files=files) == {"src/app/parser.py", "tests/test_parser.py"}


@pytest.mark.parametrize(
    ("reply", "edited", "reward"),
    [
        ("A fix goes in src/a.py, which b.py calls.", "", "1"),
        ("A fix goes in src/a.py.", "b.py", "0"),
        # the image's own setup can leave files like these changed, which isn't the agent editing code
        ("A fix goes in src/a.py.", "yarn.lock", "1"),
        ("A fix goes in src/a.py.", ".gitmodules", "1"),
        ("It's somewhere in b.py or c.py.", "", "0"),
        ("It's in src/a.py, b.py, c.py, d.py, e.py or f.py.", "", "0"),
    ],
    ids=["right", "edited", "lock-file-changed", "dotfile-changed", "wrong-file", "too-many-files"],
)
def test_an_answer_passes_when_it_names_the_fixs_files_and_few_others_and_no_code_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: str, edited: str, reward: str
) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    for name in ("src/a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "README.md", "yarn.lock", ".gitmodules"):
        (repo / name).write_text("x = 1\n")
    for command in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
        subprocess.run(["git", *command], cwd=repo, check=True)
    if edited:
        (repo / edited).write_text("x = 2\n")
    (tmp_path / "reply.txt").write_text(reply)
    # README.md and package-lock.json in the fix don't count, so the answer only has to name src/a.py
    (tmp_path / "fix.patch").write_text(
        "diff --git a/src/a.py b/src/a.py\n+x\ndiff --git a/README.md b/README.md\n+y\n"
        "diff --git a/web/package-lock.json b/web/package-lock.json\n+z\n"
    )
    monkeypatch.setattr(grade, "REPLY", tmp_path / "reply.txt")
    monkeypatch.setattr(grade, "FIX", tmp_path / "fix.patch")
    monkeypatch.setattr(grade, "LOGS", tmp_path)
    monkeypatch.chdir(repo)
    grade.main()
    assert (tmp_path / "reward.txt").read_text().strip() == reward


def test_the_grader_leaves_out_the_files_the_mix_left_out_when_it_picked_the_task() -> None:
    mix = folders.module(root=paths.benchmarks_dir(), name="coding-mix", file="benchmark.py")
    assert (grade.NOT_CODE.pattern, grade.NOT_CODE.flags) == (mix.NOT_CODE.pattern, mix.NOT_CODE.flags)
