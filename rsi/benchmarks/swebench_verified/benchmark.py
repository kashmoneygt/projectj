import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import docker
from swebench.task import checks

from rsi import benchmarks, core, paths, views

log = logging.getLogger(__name__)

_TASKS_REPO_URL = "https://github.com/SWE-bench/swe-bench-tasks.git"
TASKS_COMMIT = "3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d"  # swe-bench-tasks commit that builds and grades every task
NAMESPACE = "swebench"  # prefix of the image names swebench's eval looks for
WORKDIR = "/testbed"  # where each task's repo is in its image
# swebench's CLI, but building only missing eval images
SB_COMMAND = [sys.executable, str(Path(__file__).with_name("swebench_cli.py"))]
EVAL_LOG_MAX = 18_000  # characters of the test log the viewer shows (failures first, then the end)
# the grader's two groups of hidden tests, each with what a failing test in it means
_CHECKS = (
    ("FAIL_TO_PASS", "should pass after the fix (FAIL_TO_PASS)", "still failing"),
    ("PASS_TO_PASS", "should keep passing (PASS_TO_PASS)", "broken"),
)


def run_sb(args: list[str], cwd: Path) -> None:
    cmd = SB_COMMAND + args
    log.info(f"$ {' '.join(cmd)}")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # stream swebench's output to our log
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        log.info(f"  {line.rstrip()}")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def tasks_repo() -> Path:
    return paths.rsi_root() / "swe-bench-tasks"


def task_image(instance_id: str) -> str:
    """The image a task runs and gets graded in"""
    return checks.expected_image(instance_id=instance_id, namespace=NAMESPACE)


def unbuilt(preds: Path) -> list[str]:
    """Tasks with a patch to grade but no image built"""
    client = docker.from_env()
    missing = []
    for instance_id, pred in json.loads(preds.read_text()).items():
        if not pred["model_patch"]:
            continue
        try:
            client.images.get(task_image(instance_id=instance_id))
        except docker.errors.ImageNotFound:
            missing.append(instance_id)
    return missing


def report_path(cwd: Path, model_name: str, run_id: str) -> Path:
    """swebench v5's report path (/ in the model name becomes __)"""
    return cwd / f"{model_name.replace('/', '__')}.{run_id}.json"


def read_report(cwd: Path, model_name: str, run_id: str) -> dict[str, bool]:
    """Whether each task resolved, leaving out the ones the grader failed on"""
    data = json.loads(report_path(cwd=cwd, model_name=model_name, run_id=run_id).read_text())
    resolved_ids = set(data["resolved_ids"])
    ungraded = set(data["error_ids"]) | set(data["infra_failure_ids"])
    return {iid: iid in resolved_ids for iid in data["submitted_ids"] if iid not in ungraded}


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


def _instance(row: dict, request: str) -> core.Instance:
    return core.Instance(
        instance_id=row["instance_id"],
        repo=row["repo"],
        image=task_image(instance_id=row["instance_id"]),
        workdir=WORKDIR,
        task=request.replace("{task}", row["problem_statement"].strip()),
    )


class SweBenchVerified:
    name = "swebench-verified"
    alias = "verified"
    dataset = core.DatasetSpec(
        name="swebench-verified",
        dataset_id="princeton-nlp/SWE-bench_Verified",
        revision="c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
        request="Please resolve this issue:\n\n{task}",
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
        "hidden tests",  # how its tasks are graded
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
        """Clone swe-bench-tasks the first time, and check it's at the pinned commit"""
        repo = tasks_repo()
        if not repo.is_dir():
            if not clone:
                raise FileNotFoundError("swe-bench-tasks repo not present; pass --clone-tasks")
            log.info(f"cloning SWE-bench/swe-bench-tasks at {TASKS_COMMIT[:12]} into {repo} (one-time) ...")
            subprocess.run(["git", "clone", "--quiet", "--depth", "1", _TASKS_REPO_URL, str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "fetch", "--quiet", "--depth", "1", "origin", TASKS_COMMIT], check=True
            )
            subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", TASKS_COMMIT], check=True)
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        if head.stdout.strip() != TASKS_COMMIT:
            raise RuntimeError(f"{repo} is at {head.stdout.strip()[:12] or 'no commit'}, not {TASKS_COMMIT[:12]}")

    def select(self, tasks: int, seed: int) -> list[core.Instance]:
        """A seeded sample of `tasks` rows spread across the repos (0 means all), the same on every machine"""
        rows = {row["instance_id"]: row for row in benchmarks.rows(spec=self.dataset)}
        ids = list(rows)
        if 0 < tasks < len(ids):
            repos = {iid: row["repo"] for iid, row in rows.items()}
            ids = benchmarks.spread(ids=ids, repos=repos, count=tasks, seed=seed)
        return [_instance(row=rows[iid], request=self.dataset.request) for iid in ids]

    def by_ids(self, ids: list[str]) -> list[core.Instance]:
        rows = {r["instance_id"]: r for r in benchmarks.rows(spec=self.dataset)}
        unknown = [i for i in ids if i not in rows]
        if unknown:
            raise ValueError(f"unknown instance id(s) for {self.name}: {unknown[:5]}")
        return [_instance(row=rows[i], request=self.dataset.request) for i in ids]

    def build_images(self, instances: list[core.Instance], workers: int, cwd: Path) -> None:
        args = ["images", "build", str(tasks_repo()), "-d", self.alias, "-n", NAMESPACE, "-j", str(workers)]
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
        missing = unbuilt(preds=preds)
        if missing:
            # otherwise the eval builds its own, and grades in an image the agent never ran in
            raise RuntimeError(f"no image for {', '.join(missing)}; build them with `rsi images build`")
        # swebench takes the model name from the preds file
        args = ["eval", self.alias, "-p", str(preds), "--run-id", run_id, "-j", str(workers)]
        run_sb(args=args + ["--task-repo", str(tasks_repo())], cwd=cwd)
        report = report_path(cwd=cwd, model_name=model_name, run_id=run_id)
        logs = cwd / "logs" / "run_evaluation" / run_id / model_name
        for instance_id in json.loads(report.read_text())["infra_failure_ids"]:
            # swebench skips tasks it already has a report for, so remove it to grade it again
            shutil.rmtree(logs / instance_id, ignore_errors=True)
        per_instance = read_report(cwd=cwd, model_name=model_name, run_id=run_id)
        persist_eval_artifacts(src=logs, instances_dir=instances_dir)
        shutil.copy2(report, node_dir / report.name)
        return per_instance

    def verdict(self, task_dir: Path) -> views.Verdict | None:
        try:
            report = json.loads((task_dir / "eval" / "report.json").read_text())
        except (OSError, ValueError):
            return None
        if not report:
            return None
        record = next(iter(report.values()))
        log_text = ""
        log_path = task_dir / "eval" / "test_output.txt"
        if log_path.is_file():
            lines = log_path.read_text(errors="replace").splitlines()
            failures = [x for x in lines if x.startswith(("FAILED", "ERROR")) or "Test Exit Code" in x]
            log_text = "\n".join([*failures, "", f"{'=' * 30} end of log {'=' * 30}", *lines[-60:]])[-EVAL_LOG_MAX:]
        tests = record.get("tests_status") or {}
        checks = []
        for key, label, failing in _CHECKS:
            status = tests.get(key) or {}
            passed, failed = len(status.get("success") or []), list(status.get("failure") or [])
            checks.append(
                views.Checks(
                    label=label, failing=failing, total=passed + len(failed), passed=passed, failed=failed[:50]
                )
            )
        return views.Verdict(
            resolved=bool(record.get("resolved")),
            applied=bool(record.get("patch_successfully_applied", True)),
            infra=bool(record.get("infra_failure")),
            checks=checks,
            log=log_text,
        )

    def gold(self, instance_id: str, run_id: str, cwd: Path) -> None:
        args = ["eval", self.alias, "--gold", "-i", instance_id, "--run-id", run_id]
        run_sb(args=args + ["--task-repo", str(tasks_repo())], cwd=cwd)


BENCHMARK = SweBenchVerified()
