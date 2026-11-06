import dataclasses
import math
import statistics
from fractions import Fraction

from rsi import core


def _basis(variant_runs: int, reference_runs: int, reference: str, tasks: int) -> str:
    return f"{variant_runs} run{'s' if variant_runs != 1 else ''} vs {reference_runs} of {reference}, {tasks} tasks"


@dataclasses.dataclass(frozen=True)
class Evidence:
    """Solves vs the parent on the same tasks: gain per run and its standard error"""

    variant_runs: int
    reference_runs: int
    tasks: int
    gain: float
    se: float

    @property
    def z(self) -> float:
        # se is 0 only when gain is 0 too
        return self.gain / self.se if self.se else 0.0

    def line(self, reference: str = "the parent") -> str:
        basis = _basis(
            variant_runs=self.variant_runs, reference_runs=self.reference_runs, reference=reference, tasks=self.tasks
        )
        return f"{self.gain:+.1f} ± {self.se:.1f} tasks per run, {verdict(z=self.z)} ({basis})"


@dataclasses.dataclass(frozen=True)
class TokenEvidence:
    """Output tokens vs the parent on the same tasks: mean log saving and its standard error"""

    variant_runs: int
    reference_runs: int
    tasks: int
    saving: float
    se: float

    @property
    def z(self) -> float:
        return self.saving / self.se if self.se else 0.0  # 0 with fewer than 2 tasks or no spread

    def line(self, reference: str = "the parent") -> str:
        basis = _basis(
            variant_runs=self.variant_runs, reference_runs=self.reference_runs, reference=reference, tasks=self.tasks
        )
        return f"tokens per task {math.exp(-self.saving) - 1:+.0%} ± {self.se:.0%}, {verdict(z=self.z)} ({basis})"


def evidence(variant: list[core.Node], reference: list[core.Node], tasks: list[str]) -> Evidence:
    """Paired per-task test of pass rates (McNemar's test with one run a side)"""
    gain = Fraction(0)  # exact, so gains that cancel out sum to 0
    variance = 0.0
    counted = 0
    for task in tasks:
        v = [n.runs[task].resolved for n in variant if task in n.runs]
        r = [n.runs[task].resolved for n in reference if task in n.runs]
        if not v or not r:
            continue
        counted += 1
        m = len(v) + len(r)
        pooled = (sum(v) + sum(r)) / m
        gain += Fraction(sum(v), len(v)) - Fraction(sum(r), len(r))
        variance += pooled * (1 - pooled) * m / (m - 1) * (1 / len(v) + 1 / len(r))
    return Evidence(
        variant_runs=len(variant),
        reference_runs=len(reference),
        tasks=counted,
        gain=float(gain),
        se=math.sqrt(variance),
    )


def token_evidence(variant: list[core.Node], reference: list[core.Node], tasks: list[str]) -> TokenEvidence:
    """Paired per-task test of log output tokens, over tasks every run finished"""
    diffs: list[float] = []
    for task in tasks:
        v = [n.runs[task] for n in variant if task in n.runs]
        r = [n.runs[task] for n in reference if task in n.runs]
        if v and r and all(run.status in ("ok", "timeout") and run.metrics.output_tokens > 0 for run in v + r):
            diffs.append(_mean_log_tokens(runs=v) - _mean_log_tokens(runs=r))
    return TokenEvidence(
        variant_runs=len(variant),
        reference_runs=len(reference),
        tasks=len(diffs),
        saving=-statistics.fmean(diffs) if diffs else 0.0,
        se=statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0,
    )


def _mean_log_tokens(runs: list[core.TaskRun]) -> float:
    return statistics.fmean(math.log(run.metrics.output_tokens) for run in runs)


NOISE_Z = 2.0  # a no-better variant reaches it ~2% of the time (the best of four ~9%)


def verdict(z: float) -> str:
    if z >= NOISE_Z:
        return "better, beyond noise"
    if z <= -NOISE_Z:
        return "worse, beyond noise"
    return "within noise"
