import math
import statistics

import pytest

from rsi import core, stats


def _run(name: str, passed: set[str], tasks: list[str]) -> core.Node:
    node = core.Node(
        name=name,
        dataset=core.DatasetSpec(name="swebench-verified", dataset_id="princeton-nlp/SWE-bench_Verified"),
        harness="pi",
        harness_version="1.4.0",
        model=core.ModelSpec(name="m", source="path:/tmp/x.gguf", model_id="m"),
        budget=core.Budget(),
        tasks=len(tasks),
        seed=1,
    )
    node.runs = {t: core.TaskRun(instance_id=t, status="ok", resolved=t in passed) for t in tasks}
    return node


def test_one_run_each_is_mcnemars_test() -> None:
    # 5 fixed and 3 broken: +2 tasks, se sqrt(8), z 0.71
    tasks = [f"t{i}" for i in range(38)]
    both = set(tasks[:17])
    parent = _run(name="parent", passed=both | {"t20", "t21", "t22"}, tasks=tasks)
    variant = _run(name="variant", passed=both | {"t30", "t31", "t32", "t33", "t34"}, tasks=tasks)
    found = stats.evidence(variant=[variant], reference=[parent], tasks=tasks)
    assert found.gain == 2 and found.se == pytest.approx(8**0.5)
    assert found.z == pytest.approx(2 / 8**0.5)
    assert found.line() == "+2.0 ± 2.8 tasks per run (z 0.71; 1 run vs 1 of the parent, 38 tasks)"
    assert found.line(reference="a") == "+2.0 ± 2.8 tasks per run (z 0.71; 1 run vs 1 of a, 38 tasks)"


def test_more_runs_shrink_the_error_and_settled_tasks_add_nothing() -> None:
    tasks = ["flip", "always", "never"]
    parent = [_run(name=f"p{i}", passed={"always"}, tasks=tasks) for i in range(3)]
    variant = [_run(name=f"v{i}", passed={"always", "flip"}, tasks=tasks) for i in range(3)]
    found = stats.evidence(variant=variant, reference=parent, tasks=tasks)
    # only "flip" differs: 3/3 vs 0/3, pooled rate 1/2, times 6/5 for its 6 runs
    assert found.gain == 1 and found.se == pytest.approx((0.25 * 6 / 5 * (1 / 3 + 1 / 3)) ** 0.5)
    one = stats.evidence(variant=variant[:1], reference=parent, tasks=tasks)
    assert one.se > found.se and (one.variant_runs, one.reference_runs) == (1, 3)
    assert stats.evidence(variant=variant, reference=parent, tasks=["missing"]).tasks == 0


def test_identical_runs_give_no_signal_either_way() -> None:
    tasks = ["a", "b"]
    variant, parent = _run(name="v", passed={"a"}, tasks=tasks), _run(name="p", passed={"a"}, tasks=tasks)
    same = stats.evidence(variant=[variant], reference=[parent], tasks=tasks)
    assert (same.gain, same.se, same.z) == (0, 0, 0)


def test_a_verdict_calls_z_2_either_way_beyond_noise() -> None:
    assert stats.verdict(z=2.0) == stats.verdict(z=8.5) == "better, beyond noise"
    assert stats.verdict(z=-2.0) == stats.verdict(z=-5.2) == "worse, beyond noise"
    assert stats.verdict(z=1.99) == stats.verdict(z=0.0) == stats.verdict(z=-1.99) == "within noise"


def test_a_gain_that_cancels_out_is_exactly_zero() -> None:
    # +1/3 and -1/3 sum to 5.6e-17 in floats
    tasks = ["fixed", "broken"]
    variant = [_run(name=f"v{i}", passed={"fixed"}, tasks=tasks) for i in range(3)]
    parent = [
        _run(name="p0", passed={"fixed", "broken"}, tasks=tasks),
        _run(name="p1", passed={"fixed"}, tasks=tasks),
        _run(name="p2", passed=set(), tasks=tasks),
    ]
    assert stats.evidence(variant=variant, reference=parent, tasks=tasks).gain == 0


def _spent(name: str, tokens: dict[str, int]) -> core.Node:
    node = _run(name=name, passed=set(), tasks=list(tokens))
    for task, count in tokens.items():
        node.runs[task].metrics.output_tokens = count
    return node


def test_tokens_compare_by_the_share_a_variant_saves_on_each_task() -> None:
    parent = [_spent(name=f"p{i}", tokens={"a": 1000, "b": 4000, "c": 9000}) for i in range(2)]
    variant = [
        _spent(name="v0", tokens={"a": 500, "b": 2000, "c": 3000}),
        _spent(name="v1", tokens={"a": 500, "b": 2000, "c": 9000}),
    ]
    found = stats.token_evidence(variant=variant, reference=parent, tasks=["a", "b", "c"])
    diffs = [math.log(0.5), math.log(0.5), (math.log(3000) + math.log(9000)) / 2 - math.log(9000)]
    assert (found.tasks, found.variant_runs, found.reference_runs) == (3, 2, 2)
    assert found.saving == pytest.approx(-statistics.fmean(diffs))
    assert found.se == pytest.approx(statistics.stdev(diffs) / math.sqrt(3))
    assert found.z == pytest.approx(found.saving / found.se)


def test_only_tasks_every_run_finished_with_output_count_toward_tokens() -> None:
    parent = [_spent(name="p", tokens={"a": 1000, "b": 1000, "c": 1000, "d": 1000})]
    variant = [_spent(name="v", tokens={"a": 500, "b": 0, "c": 400, "d": 300})]
    variant[0].runs["c"].status = "error"
    variant[0].runs["d"].status = "timeout"
    found = stats.token_evidence(variant=variant, reference=parent, tasks=["a", "b", "c", "d", "missing"])
    assert found.tasks == 2 and found.saving == pytest.approx(-(math.log(0.5) + math.log(0.3)) / 2)
    one = stats.token_evidence(variant=variant, reference=parent, tasks=["a"])
    assert (one.tasks, one.se, one.z) == (1, 0, 0)
    halved = stats.token_evidence(
        variant=[_spent(name="v", tokens={"a": 500, "b": 500})],
        reference=[_spent(name="p", tokens={"a": 1000, "b": 1000})],
        tasks=["a", "b"],
    )
    assert halved.saving == pytest.approx(math.log(2)) and halved.z == 0


def test_the_token_line_reads_as_a_share_of_the_parents_tokens() -> None:
    found = stats.TokenEvidence(variant_runs=3, reference_runs=3, tasks=44, saving=-math.log(0.77), se=0.09)
    assert found.line() == "tokens per task -23% ± 9% (z 2.90; 3 runs vs 3 of the parent, 44 tasks)"
    costlier = stats.TokenEvidence(variant_runs=1, reference_runs=3, tasks=40, saving=-math.log(1.5), se=0.2)
    assert costlier.line(reference="a") == "tokens per task +50% ± 20% (z -2.03; 1 run vs 3 of a, 40 tasks)"
