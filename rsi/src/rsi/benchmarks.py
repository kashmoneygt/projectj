from pathlib import Path
from typing import Protocol

import datasets

from rsi import core, folders, paths


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

    def gold(self, instance_id: str, run_id: str, cwd: Path) -> None: ...
    def clean_images(self, cwd: Path) -> None: ...


def rows(spec: core.DatasetSpec) -> list[dict]:
    return list(datasets.load_dataset(spec.dataset_id, split=spec.split, revision=spec.revision or None))


def get(name: str) -> Benchmark:
    return folders.module(root=paths.benchmarks_dir(), name=name, file="benchmark.py", kind="benchmark").BENCHMARK
