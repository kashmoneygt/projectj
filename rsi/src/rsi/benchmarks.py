import math
import random
from pathlib import Path
from typing import Protocol

import datasets

from rsi import core, folders, paths, views


class Benchmark(Protocol):
    """Task selection, environment images and official scoring"""

    name: str
    dataset: core.DatasetSpec
    protocol_terms: tuple[str, ...]  # words for the benchmark, banned from files the agent reads

    def ensure_assets(self, clone: bool) -> None: ...
    def select(self, tasks: int, seed: int) -> list[core.Instance]: ...
    def by_ids(self, ids: list[str]) -> list[core.Instance]: ...
    def build_images(self, instances: list[core.Instance], workers: int, cwd: Path) -> None: ...

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
        """Grade each task, leaving out failed gradings for a second call in the same cwd"""

    def verdict(self, task_dir: Path) -> views.Verdict | None:
        """How a task was judged, from what scoring left in its folder (None before scoring)"""

    def gold(self, instance_id: str, run_id: str, cwd: Path) -> None: ...
    def clean_images(self, cwd: Path) -> None: ...


def rows(spec: core.DatasetSpec) -> list[dict]:
    return list(datasets.load_dataset(spec.dataset_id, split=spec.split, revision=spec.revision or None))


def spread(ids: list[str], repos: dict[str, str], count: int, seed: int) -> list[str]:
    """A seeded sample of count tasks, each repo's share by the square root of its size so none crowds out the rest"""
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    if count >= len(shuffled):
        return shuffled
    by_repo: dict[str, list[str]] = {}
    for iid in shuffled:
        by_repo.setdefault(repos[iid], []).append(iid)
    weights = {repo: math.sqrt(len(found)) for repo, found in by_repo.items()}
    shares = {repo: count * weight / sum(weights.values()) for repo, weight in weights.items()}
    taken = {repo: min(int(share), len(by_repo[repo])) for repo, share in shares.items()}
    while sum(taken.values()) < count:
        # the rest go to the repos furthest under their share, among those with tasks left
        repo = max((r for r in by_repo if taken[r] < len(by_repo[r])), key=lambda r: shares[r] - taken[r])
        taken[repo] += 1
    picked = {iid for repo, found in by_repo.items() for iid in found[: taken[repo]]}
    return [iid for iid in shuffled if iid in picked]


def banned_terms(benchmark: Benchmark) -> tuple[str, ...]:
    """Words a variant can't put in front of the agent, which are the benchmark's own and how its task message starts"""
    lead = benchmark.dataset.request.partition("{task}")[0].strip()
    return benchmark.protocol_terms + ((lead,) if lead else ())


def get(name: str) -> Benchmark:
    return folders.module(root=paths.benchmarks_dir(), name=name, file="benchmark.py", kind="benchmark").BENCHMARK
