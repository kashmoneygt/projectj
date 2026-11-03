import json
import runpy
import subprocess
from pathlib import Path

import pytest
from swebench.image_builder import docker_build

from rsi import benchmarks, core, folders, paths

mod = folders.module(root=paths.benchmarks_dir(), name="swebench-verified", file="benchmark.py")


def _tasks_repo(root: Path) -> Path:
    repo = root / "swe-bench-tasks"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text("tasks")
    return repo


def test_report_path_and_read(tmp_path: Path) -> None:
    path = mod.report_path(cwd=tmp_path, model_name="gemma-4-12b", run_id="node-a")
    assert path.name == "gemma-4-12b.node-a.json"
    report = {"resolved_ids": ["a__b-1"], "submitted_ids": ["a__b-1", "c__d-2", "e__f-3"], "error_ids": ["e__f-3"]}
    path.write_text(json.dumps(report))
    assert mod.read_report(cwd=tmp_path, model_name="gemma-4-12b", run_id="node-a") == {
        "a__b-1": True,
        "c__d-2": False,
    }
    slash = mod.report_path(cwd=tmp_path, model_name="org/model", run_id="node-a")
    assert slash.name == "org__model.node-a.json"


def test_by_ids(monkeypatch) -> None:
    rows = [
        {"instance_id": "a__b-1", "repo": "r1", "base_commit": "c1", "problem_statement": "p1"},
        {"instance_id": "c__d-2", "repo": "r2", "base_commit": "c2", "problem_statement": "p2"},
    ]
    monkeypatch.setattr(benchmarks, "rows", lambda spec: rows)
    picked = mod.BENCHMARK.by_ids(ids=["c__d-2", "a__b-1"])
    assert [i.instance_id for i in picked] == ["c__d-2", "a__b-1"]
    assert picked[0].base_commit == "c2"
    with pytest.raises(ValueError, match="x__y-9"):
        mod.BENCHMARK.by_ids(ids=["x__y-9"])


def test_command_args(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    repo = _tasks_repo(root=tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(mod, "run_sb", lambda args, cwd: calls.append(args))
    instance = core.Instance(instance_id="django__django-1", repo="r", base_commit="c", problem_statement="p")

    mod.BENCHMARK.build_images(instances=[instance], workers=2, cwd=tmp_path)
    assert calls[-1] == ["images", "build", str(repo), "-d", "verified", "-j", "2", "-i", "django__django-1"]

    mod.BENCHMARK.gold(instance_id="sympy__sympy-20590", run_id="gold-check", cwd=tmp_path)
    assert calls[-1][:3] == ["eval", "verified", "--gold"]
    assert calls[-1][calls[-1].index("-i") + 1] == "sympy__sympy-20590"
    assert calls[-1][calls[-1].index("--task-repo") + 1] == str(repo)


def test_tasks_repo_clone_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="--clone-tasks"):
        mod.BENCHMARK.ensure_assets(clone=False)

    def fake_run(cmd: list[str], check: bool) -> None:
        _tasks_repo(root=tmp_path)

    monkeypatch.setattr(subprocess, "run", fake_run)
    repo = mod.tasks_repo(clone=True)
    assert repo == tmp_path / "swe-bench-tasks"
    assert mod.tasks_repo(clone=False) == repo


def test_score_persists_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    repo = _tasks_repo(root=tmp_path)
    calls: list[list[str]] = []

    def fake_run_sb(args: list[str], cwd: Path) -> None:
        calls.append(args)
        mod.report_path(cwd=cwd, model_name="gemma-4-12b", run_id="node-a").write_text(
            json.dumps({"resolved_ids": ["a__b-1"], "submitted_ids": ["a__b-1", "c__d-2"], "error_ids": []})
        )

    monkeypatch.setattr(mod, "run_sb", fake_run_sb)
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    per_instance = mod.BENCHMARK.score(
        preds=tmp_path / "preds.json",
        run_id="node-a",
        model_name="gemma-4-12b",
        workers=2,
        cwd=tmp_path,
        node_dir=node_dir,
        instances_dir=node_dir / "instances",
    )
    assert per_instance == {"a__b-1": True, "c__d-2": False}
    assert (node_dir / "gemma-4-12b.node-a.json").exists()
    assert calls[-1][:3] == ["eval", "verified", "-p"]
    assert calls[-1][calls[-1].index("--run-id") + 1] == "node-a"
    assert calls[-1][calls[-1].index("--task-repo") + 1] == str(repo)


def test_persist_eval_artifacts(tmp_path: Path) -> None:
    src = tmp_path / "logs" / "run_evaluation" / "node-a" / "gemma-4-12b"
    for iid in ("a__b-1", "c__d-2"):
        d = src / iid
        d.mkdir(parents=True)
        (d / "instance.log").write_text(f"log for {iid}")
    instances_dir = tmp_path / "node" / "instances"
    assert mod.persist_eval_artifacts(src=src, instances_dir=instances_dir) == 2
    assert (instances_dir / "a__b-1" / "eval" / "instance.log").read_text() == "log for a__b-1"
    assert mod.persist_eval_artifacts(src=tmp_path / "missing", instances_dir=instances_dir) == 0


def test_eval_builds_only_the_missing_images(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_build(
        client: object, image_specs: list, force_rebuild: bool, max_workers: int, dry_run: bool
    ) -> tuple[list, list]:
        calls.append({"force_rebuild": force_rebuild, "max_workers": max_workers})
        return [], []

    monkeypatch.setattr(docker_build, "build_instance_images", fake_build)
    cli = Path(mod.__file__).with_name("swebench_cli.py")
    runpy.run_path(str(cli), run_name="swebench_cli")
    docker_build.build_instance_images(client=None, image_specs=[], force_rebuild=True, max_workers=2)
    assert calls == [{"force_rebuild": False, "max_workers": 2}]
    assert mod.SB_COMMAND[-1] == str(cli)
