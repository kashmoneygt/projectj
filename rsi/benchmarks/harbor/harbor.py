import concurrent.futures
import dataclasses
import functools
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import tomllib
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

import docker
import huggingface_hub

from rsi import benchmarks, core, harness, paths, runs, views

log = logging.getLogger(__name__)

HF = "hf"  # a Hugging Face dataset
GITHUB = "github"  # a GitHub repo
VERIFY = "chmod +x /tests/test.sh && /tests/test.sh"  # the command Harbor runs to grade a task
EVAL_LOG_MAX = 18_000  # max characters of grader output the viewer shows (FAILED/ERROR lines first, then the end)
# timeout for pulling or building a task image, set high since amd64 images take longer on an arm64 host
IMAGE_TIMEOUT_S = 2400
_fetch_lock = threading.Lock()


def _online(mode: str | None, default: bool) -> bool:
    """Whether a Harbor network_mode allows the internet (an allowlist counts as online, since we can't enforce it)"""
    return default if mode is None else mode != "no-network"


def _resolve(value: object) -> str:
    """Fill in ${VAR} and ${VAR:-default} in a Harbor env value from env vars on the host machine"""
    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}", lambda m: os.environ.get(m[1], m[2] or ""), str(value))


def listed(path: Path) -> list[str]:
    """The task names in a file, one per line, skipping # comments"""
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


@dataclasses.dataclass(frozen=True)
class Source:
    """A folder of Harbor tasks in a repository, pinned to a commit"""

    name: str  # task ids are "<name>.<task name>"
    kind: str  # kind of work in a few words, e.g. "bug fixes"
    host: str  # HF or GITHUB
    repo: str  # e.g. "harborframework/terminal-bench-2.1"
    revision: str  # the pinned commit
    folder: str  # folder in the repo that holds the task folders
    internet: bool | None = None  # turn internet on or off for the agent and grader, None to follow each task.toml
    leave_out: str = ""  # regex for task names to skip
    pick: Callable[[], Iterable[str]] | None = None  # returns the task names to use, None to use the whole folder
    repo_of: str = ""  # regex for the repo in a task name (group 1), used to spread samples. Falls back to the name
    request: str = "{task}"  # template for the message the agent gets, where {task} is the task's instruction.md
    # folder with our tests/ and solution/ to swap in for each task's. Both get the task's solution/ as reference/
    grader: str = ""

    @property
    def cache(self) -> Path:
        return paths.rsi_root() / ".cache" / "tasks" / self.name / self.revision[:12]

    @functools.cached_property
    def changes(self) -> str:
        """Short hash of our request template and grader files, or "" if we use the tasks as they are"""
        if self.request == "{task}" and not self.grader:
            return ""
        digest = hashlib.sha256(self.request.encode())
        for path in sorted(Path(self.grader).rglob("*")) if self.grader else []:
            parts = path.relative_to(self.grader).parts
            # skip __pycache__ and dotfiles since they change on their own, and _graded doesn't copy them either
            if path.is_file() and "__pycache__" not in parts and not any(part.startswith(".") for part in parts):
                digest.update(f"{path.relative_to(self.grader)}\0".encode() + path.read_bytes())
        return digest.hexdigest()[:8]

    def names(self) -> list[str]:
        """Sorted task names from pick or the whole folder, minus leave_out"""
        names = sorted(self.pick()) if self.pick else self._listing()
        return [n for n in names if not (self.leave_out and re.search(self.leave_out, n))]

    def _listing(self) -> list[str]:
        listing = self.cache / "names.json"
        if not listing.is_file():
            runs.atomic_write(path=listing, text=json.dumps(self._list(), indent=0) + "\n")
        return json.loads(listing.read_text())

    def code_of(self, name: str) -> str:
        found = re.search(self.repo_of, name) if self.repo_of else None
        return found[1] if found else name

    def task_dir(self, name: str) -> Path:
        return self.cache / self.folder / name

    def fetch(self, names: list[str]) -> None:
        """Download any of these tasks that aren't cached yet"""
        with _fetch_lock:
            missing = [n for n in names if not (self.task_dir(name=n) / "task.toml").is_file()]
            if not missing:
                return
            if self.host == GITHUB:
                self._download_repo()
                return
            log.info(f"downloading {len(missing)} tasks of {self.repo} at {self.revision[:12]}")
            huggingface_hub.snapshot_download(
                repo_id=self.repo,
                repo_type="dataset",
                revision=self.revision,
                allow_patterns=[f"{self.folder}/{name}/**" for name in missing],
                local_dir=self.cache,
            )

    def _list(self) -> list[str]:
        if self.host == GITHUB:
            self._download_repo()
            return sorted(p.parent.name for p in (self.cache / self.folder).glob("*/task.toml"))
        api = huggingface_hub.HfApi()
        found = api.list_repo_tree(self.repo, path_in_repo=self.folder, repo_type="dataset", revision=self.revision)
        return sorted(
            PurePosixPath(entry.path).name for entry in found if isinstance(entry, huggingface_hub.RepoFolder)
        )

    def _download_repo(self) -> None:
        """Download the task folder out of the GitHub tarball at the pinned commit, once"""
        done = self.cache / ".downloaded"
        if done.is_file():
            return
        url = f"https://codeload.github.com/{self.repo}/tar.gz/{self.revision}"
        log.info(f"downloading {self.folder}/ of {self.repo} at {self.revision[:12]} ...")
        with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                # drop the tarball's top folder, which is <repo>-<commit>
                inside = member.name.partition("/")[2]
                if inside.startswith(f"{self.folder}/") and (member.isfile() or member.isdir()):
                    member.name = inside
                    tar.extract(member, self.cache, filter="data")
        done.touch()

    def instance(self, name: str) -> core.Instance:
        folder = self._graded(name=name) if self.grader else self.task_dir(name=name)
        config = tomllib.loads((folder / "task.toml").read_text())
        environment = config.get("environment", {})
        verifier = config.get("verifier", {})
        online = _online(mode=environment.get("network_mode"), default=bool(environment.get("allow_internet", True)))
        agent_online = _online(mode=config.get("agent", {}).get("network_mode"), default=online)
        verifier_online = _online(mode=verifier.get("network_mode"), default=online)
        if self.internet is not None:
            agent_online = verifier_online = self.internet
        return core.Instance(
            instance_id=f"{self.name}.{name}",
            repo=self.code_of(name=name),
            image=environment.get("docker_image") or self.built_image(name=name),
            workdir=environment.get("workdir", ""),
            task=self.request.replace("{task}", (folder / "instruction.md").read_text().strip()),
            internet=agent_online,
            cpus=float(environment.get("cpus") or 0),
            memory_mb=int(environment.get("memory_mb") or 0),
            env={key: _resolve(value=value) for key, value in environment.get("env", {}).items()},
            verifier=core.Verifier(
                tests=str(folder / "tests"),
                command=VERIFY,
                timeout_s=int(verifier.get("timeout_sec") or 600),
                internet=verifier_online,
                env={key: _resolve(value=value) for key, value in verifier.get("env", {}).items()},
            ),
            solution=str(folder / "solution") if (folder / "solution" / "solve.sh").is_file() else "",
            kind=self.kind,
        )

    def built_image(self, name: str) -> str:
        """Tag for the image we build from environment/ when task.toml names no docker_image"""
        return f"rsi/{self.name}-{self.revision[:8]}:{name}"

    def _graded(self, name: str) -> Path:
        """Copy of the task with our grader swapped in, made once per version of our changes"""
        folder = self.cache / f"graded-{self.changes}" / name
        if (folder / "task.toml").is_file():
            return folder
        task = self.task_dir(name=name)
        folder.parent.mkdir(parents=True, exist_ok=True)
        # build in a temp folder and rename it into place, so a run reading it never sees half a task
        tmp = Path(tempfile.mkdtemp(dir=folder.parent, prefix=f".{name}-"))
        shutil.copytree(task, tmp, ignore=shutil.ignore_patterns("tests", "solution"), dirs_exist_ok=True)
        for part in ("tests", "solution"):
            shutil.copytree(Path(self.grader) / part, tmp / part, ignore=shutil.ignore_patterns("__pycache__", ".*"))
            shutil.copytree(task / "solution", tmp / part / "reference")
        try:
            tmp.rename(folder)
        except OSError:
            # another process made it first, so drop ours
            shutil.rmtree(tmp)
        return folder


def _reward(folder: Path) -> float | None:
    """The reward from reward.json ("reward", else the mean of its numbers) or reward.txt, None if neither has one"""
    for name in ("reward.json", "reward.txt"):
        path = folder / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(data, dict):
            numbers = [v for v in data.values() if isinstance(v, int | float)]
            data = data.get("reward", sum(numbers) / len(numbers) if numbers else None)
        if isinstance(data, int | float) and not isinstance(data, bool):
            return float(data)
    return None


def _checks(folder: Path) -> list[views.Checks]:
    """Test results from the grader's CTRF report (ctrf.json), or [] if there isn't one"""
    try:
        tests = json.loads((folder / "ctrf.json").read_text())["results"]["tests"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    failed = [str(t.get("name")) for t in tests if t.get("status") not in ("passed", "skipped")]
    passed = sum(t.get("status") == "passed" for t in tests)
    checks = views.Checks(
        label="the task's tests", failing="failing", total=passed + len(failed), passed=passed, failed=failed[:50]
    )
    return [checks]


def _log(folder: Path) -> str:
    path = folder / "test-stdout.txt"
    if not path.is_file():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    failures = [x for x in lines if x.startswith(("FAILED", "ERROR"))]
    return "\n".join([*failures, "", f"{'=' * 30} end of log {'=' * 30}", *lines[-80:]])[-EVAL_LOG_MAX:]


@dataclasses.dataclass(frozen=True)
class HarborBenchmark:
    """Harbor tasks from one or more sources, each graded inside its own container"""

    name: str
    sources: tuple[Source, ...]
    work: str  # a few words for the teacher on what the tasks are, without giving the benchmark away
    terms: tuple[str, ...]  # words that would give the benchmark away, so variants can't use them

    @property
    def protocol_terms(self) -> tuple[str, ...]:
        """The terms plus the fixed lines of each request template, since only these tasks start or end with them"""
        lines = [line.strip() for s in self.sources for part in s.request.split("{task}") for line in part.splitlines()]
        return self.terms + tuple(dict.fromkeys(line for line in lines if line))

    @property
    def dataset(self) -> core.DatasetSpec:
        # put each source's network policy and our changes in the dataset id, since they change how the tasks run
        # and runs made with different ones don't compare
        policy = {None: "", True: " online", False: " offline"}
        return core.DatasetSpec(
            name=self.name,
            dataset_id=",".join(
                f"{s.host}:{s.repo}/{s.folder}{policy[s.internet]}{f' as {s.changes}' if s.changes else ''}"
                for s in self.sources
            ),
            revision=",".join(s.revision for s in self.sources),
        )

    def _source(self, instance_id: str) -> tuple[Source, str]:
        prefix, _, name = instance_id.partition(".")
        source = next((s for s in self.sources if s.name == prefix), None)
        if source is None or not name:
            raise ValueError(f"unknown task id for {self.name}: {instance_id!r}")
        return source, name

    def ensure_assets(self, clone: bool) -> None:
        """Nothing to set up, since tasks are downloaded when first used"""

    def select(self, tasks: int, seed: int) -> list[core.Instance]:
        """Seeded sample of `tasks` tasks (0 for all), split evenly by kind of work, then by source within a kind"""
        kinds = list(dict.fromkeys(s.kind for s in self.sources))
        ids: list[str] = []
        for k, kind in enumerate(kinds):
            share = tasks // len(kinds) + (k < tasks % len(kinds))
            mine = [s for s in self.sources if s.kind == kind]
            for j, source in enumerate(mine):
                names = source.names()
                count = share // len(mine) + (j < share % len(mine)) if tasks else len(names)
                found = [f"{source.name}.{n}" for n in names]
                repos = {f"{source.name}.{n}": source.code_of(name=n) for n in names}
                ids += benchmarks.spread(ids=found, repos=repos, count=count, seed=seed)
        return self.by_ids(ids=ids)

    def by_ids(self, ids: list[str]) -> list[core.Instance]:
        wanted = [self._source(instance_id=i) for i in ids]
        for source in self.sources:
            source.fetch(names=[name for s, name in wanted if s is source])
        return [source.instance(name=name) for source, name in wanted]

    def build_images(self, instances: list[core.Instance], workers: int, cwd: Path) -> None:
        """Pull or build each missing task image for amd64 (build from environment/ when the task names no image)"""
        client = docker.from_env()
        todo: dict[str, core.Instance] = {}
        for instance in instances:
            try:
                client.images.get(instance.image)
            except docker.errors.ImageNotFound:
                todo.setdefault(instance.image, instance)
        if not todo:
            return
        log.info(f"getting {len(todo)} task images, {workers} at a time")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
            got = list(pool.map(self._get_image, todo.values()))
        failed = [image for image, ok in zip(todo, got, strict=True) if not ok]
        if failed:
            raise RuntimeError(f"{len(failed)} task images failed to pull or build, like {failed[0]}")

    def _get_image(self, instance: core.Instance) -> bool:
        source, name = self._source(instance_id=instance.instance_id)
        if instance.image == source.built_image(name=name):
            context = str(source.task_dir(name=name) / "environment")
            command = ["docker", "build", "--platform", "linux/amd64", "-q", "-t", instance.image, context]
        else:
            command = ["docker", "pull", "--platform", "linux/amd64", "-q", instance.image]
        try:
            done = subprocess.run(command, capture_output=True, text=True, check=False, timeout=IMAGE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.error(f"docker {command[1]} for {instance.instance_id} didn't finish within {IMAGE_TIMEOUT_S} s")
            return False
        if done.returncode != 0:
            log.error(f"docker {command[1]} failed for {instance.instance_id}: {done.stderr.strip()[-800:]}")
        return done.returncode == 0

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
        """Read the grade each task got in its container, leaving out tasks where grading failed"""
        graded: dict[str, bool] = {}
        for instance_id in json.loads(preds.read_text()):
            found = self.verdict(task_dir=instances_dir / instance_id)
            if found is not None:
                graded[instance_id] = found.resolved
        return graded

    def verdict(self, task_dir: Path) -> views.Verdict | None:
        folder = task_dir / harness.VERIFIER_DIR
        try:
            grading = json.loads((task_dir / harness.GRADING_FILE).read_text())
        except (OSError, ValueError):
            return None
        reward = _reward(folder=folder)
        # count a grading timeout as a fail, since grading again would just time out again
        if reward is None and not grading.get("timed_out"):
            return None
        log_text = _log(folder=folder)
        if grading.get("timed_out"):
            log_text = f"grading ran out of time after {grading.get('seconds')} s\n{log_text}"
        return views.Verdict(
            resolved=reward is not None and reward >= 1.0,
            applied=True,
            infra=False,
            checks=_checks(folder=folder),
            log=log_text,
        )

    def gold(self, instance_id: str, run_id: str, cwd: Path) -> None:
        raise NotImplementedError(f"{self.name} grades tasks in their containers, so `rsi gold` checks them in place")
