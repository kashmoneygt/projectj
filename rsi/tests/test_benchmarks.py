import pytest

from rsi import benchmarks, core, folders, paths


def test_get_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        benchmarks.get(name="nope")


def test_registry_lists_benchmark_folders() -> None:
    assert folders.names(root=paths.benchmarks_dir(), file="benchmark.py") == ["swebench_verified"]
    mod = folders.module(root=paths.benchmarks_dir(), name="swebench-verified", file="benchmark.py")
    assert benchmarks.get(name="swebench-verified") is mod.BENCHMARK


def test_a_variant_cant_key_on_the_start_of_the_task_message() -> None:
    bench = benchmarks.get(name="swebench-verified")
    terms = benchmarks.banned_terms(benchmark=bench)
    assert terms[: len(bench.protocol_terms)] == bench.protocol_terms
    assert terms[-1] == "Please resolve this issue:"


def test_verified_benchmark_is_pinned() -> None:
    bm = benchmarks.get(name="swebench-verified")
    assert bm.name == "swebench-verified"
    assert bm.dataset.dataset_id == "princeton-nlp/SWE-bench_Verified"
    assert bm.alias == "verified"
    assert len(bm.dataset.revision) == 40


def test_select_is_deterministic_and_random(monkeypatch) -> None:
    rows = [
        {"instance_id": f"repo__repo-{i}", "repo": f"repo{i % 5}", "base_commit": "abc", "problem_statement": "p"}
        for i in range(50)
    ]
    monkeypatch.setattr(benchmarks, "rows", lambda spec: rows)
    bm = benchmarks.get(name="swebench-verified")
    first = bm.select(tasks=10, seed=7)
    again = bm.select(tasks=10, seed=7)
    other = bm.select(tasks=10, seed=8)
    assert [i.instance_id for i in first] == [i.instance_id for i in again]
    assert [i.instance_id for i in first] != [i.instance_id for i in other]
    assert len({i.instance_id for i in first}) == 10
    for instance in first:
        assert isinstance(instance, core.Instance)
        assert instance.instance_id and instance.repo and instance.task


def test_select_zero_and_oversize_returns_all(monkeypatch) -> None:
    rows = [{"instance_id": f"r__r-{i}", "repo": "r", "base_commit": "c", "problem_statement": "p"} for i in range(8)]
    monkeypatch.setattr(benchmarks, "rows", lambda spec: rows)
    bm = benchmarks.get(name="swebench-verified")
    assert len(bm.select(tasks=0, seed=1)) == 8
    assert len(bm.select(tasks=99, seed=1)) == 8
