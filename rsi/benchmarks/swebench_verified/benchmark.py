import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from rsi import benchmarks, core, paths

log = logging.getLogger(__name__)

_TASKS_REPO_URL = "https://github.com/SWE-bench/swe-bench-tasks.git"
# swebench's CLI, but building only missing eval images
SB_COMMAND = [sys.executable, str(Path(__file__).with_name("swebench_cli.py"))]


def run_sb(args: list[str], cwd: Path) -> None:
    cmd = SB_COMMAND + args
    log.info("$ %s", " ".join(cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # stream swebench's output to our log
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        log.info("  %s", line.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def tasks_repo(clone: bool = False) -> Path:
    """The swe-bench-tasks repo the images are built from, cloned once"""
    repo = paths.rsi_root() / "swe-bench-tasks"
    if repo.exists() and any(repo.iterdir()):
        return repo
    if not clone:
        raise FileNotFoundError("swe-bench-tasks repo not present; pass --clone-tasks")
    log.info("cloning SWE-bench/swe-bench-tasks into %s (one-time) ...", repo)
    subprocess.run(["git", "clone", "--depth", "1", _TASKS_REPO_URL, str(repo)], check=True)
    return repo


def report_path(cwd: Path, model_name: str, run_id: str) -> Path:
    """swebench v5's report path (/ in the model name becomes __)"""
    return cwd / f"{model_name.replace('/', '__')}.{run_id}.json"


def read_report(cwd: Path, model_name: str, run_id: str) -> dict[str, bool]:
    """Whether each graded task resolved, leaving out failed gradings"""
    data = json.loads(report_path(cwd=cwd, model_name=model_name, run_id=run_id).read_text())
    resolved_ids, error_ids = set(data["resolved_ids"]), set(data["error_ids"])
    return {iid: iid in resolved_ids for iid in data["submitted_ids"] if iid not in error_ids}


def persist_eval_artifacts(src: Path, instances_dir: Path) -> int:
    """Copy each instance's eval logs into the node"""
    copied = 0
    if not src.exists():
        return 0
    for inst_dir in sorted(src.iterdir()):
        if not inst_dir.is_dir():
            continue
        shutil.copytree(inst_dir, instances_dir / inst_dir.name / "eval", dirs_exist_ok=True)
        copied += 1
    return copied


def _instance(row: dict) -> core.Instance:
    return core.Instance(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
    )


class SweBenchVerified:
    name = "swebench-verified"
    alias = "verified"
    dataset = core.DatasetSpec(
        name="swebench-verified",
        dataset_id="princeton-nlp/SWE-bench_Verified",
        revision="c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    )
    protocol_terms = (
        "swebench",
        "swe-bench",
        "swe_bench",
        "fail_to_pass",
        "pass_to_pass",
        "fail-to-pass",
        "pass-to-pass",
        "base commit",
        "benchmark",
        "resolved rate",
        "testbed",  # every task's repo path
        # the tasks' repos (not pytest or requests, which any project may use)
        "astropy",
        "django",
        "flask",
        "matplotlib",
        "pylint",
        "scikit-learn",
        "sklearn",
        "seaborn",
        "sphinx",
        "sympy",
        "xarray",
    )

    def ensure_assets(self, clone: bool) -> None:
        tasks_repo(clone=clone)

    def select(self, tasks: int, seed: int) -> list[core.Instance]:
        """A seeded sample of `tasks` rows (0 means all), the same on every machine"""
        picked = benchmarks.rows(spec=self.dataset)
        if 0 < tasks < len(picked):
            shuffled = picked[:]
            random.Random(seed).shuffle(shuffled)
            picked = shuffled[:tasks]
        return [_instance(row=row) for row in picked]

    def by_ids(self, ids: list[str]) -> list[core.Instance]:
        rows = {r["instance_id"]: r for r in benchmarks.rows(spec=self.dataset)}
        unknown = [i for i in ids if i not in rows]
        if unknown:
            raise ValueError(f"unknown instance id(s) for {self.name}: {unknown[:5]}")
        return [_instance(row=rows[i]) for i in ids]

    def build_images(self, instances: list[core.Instance], workers: int, cwd: Path) -> None:
        args = ["images", "build", str(tasks_repo()), "-d", self.alias, "-j", str(workers)]
        for instance in instances:
            args += ["-i", instance.instance_id]
        run_sb(args=args, cwd=cwd)

    def score(
        self,
        preds: Path,
        run_id: str,
        model_name: str,
        workers: int,
        cwd: Path,
        node_dir: Path,
        instances_dir: Path,
    ) -> dict[str, bool]:
        # swebench takes the model name from the preds file
        args = ["eval", self.alias, "-p", str(preds), "--run-id", run_id, "-j", str(workers)]
        run_sb(args=args + ["--task-repo", str(tasks_repo())], cwd=cwd)
        per_instance = read_report(cwd=cwd, model_name=model_name, run_id=run_id)
        persist_eval_artifacts(src=cwd / "logs" / "run_evaluation" / run_id / model_name, instances_dir=instances_dir)
        report = report_path(cwd=cwd, model_name=model_name, run_id=run_id)
        if report.exists():
            shutil.copy2(report, node_dir / report.name)
        return per_instance

    def gold(self, instance_id: str, run_id: str, cwd: Path) -> None:
        args = ["eval", self.alias, "--gold", "-i", instance_id, "--run-id", run_id]
        run_sb(args=args + ["--task-repo", str(tasks_repo())], cwd=cwd)

    def clean_images(self, cwd: Path) -> None:
        out = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True, check=False
        )
        stale = [line for line in out.stdout.splitlines() if line.split(":")[0].startswith("swebench/sweb.eval.")]
        if not stale:
            log.info("no swebench eval images to remove")
            return
        subprocess.run(["docker", "rmi", *stale], capture_output=True, text=True, check=False)
        log.info("removed %d swebench eval image(s): %s", len(stale), ", ".join(sorted(stale)))


BENCHMARK = SweBenchVerified()
