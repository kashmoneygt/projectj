import dataclasses
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from rsi import core, folders, paths, views

TASK_FILE = "task.md"  # where every run's folder keeps the task message the agent got


@dataclasses.dataclass(frozen=True)
class RunContext:
    node: core.Node
    instance: core.Instance
    out_dir: Path
    model: core.ModelSpec
    endpoint: core.ModelEndpoint
    budget: core.Budget
    variant: Path | None = None  # variant package folder, None for the baseline


class CheckUnavailable(Exception):
    """A check couldn't run (the model or docker failed), so it says nothing about the variant"""


class Harness(Protocol):
    name: str
    version: str
    package_dir: Path  # the baseline package
    doc: Path  # how to write a variant (the teacher's harness.md)
    protocol_terms: tuple[str, ...]  # words about how runs are judged and where the runner puts things

    def prepare(self, instances: list[core.Instance]) -> None: ...
    def run(self, ctx: RunContext) -> core.TaskRun:
        """Run one task, saving its message as TASK_FILE in ctx.out_dir as it starts"""

    def contract(self) -> str:
        """Hash of what every run gets besides the variant"""

    def transcript(self, *, instance_dir: Path, full: bool) -> str:
        """The run turn by turn, compact (to read many) or full (to search)"""

    def trajectory(self, *, instance_dir: Path) -> views.Trajectory:
        """The run as the viewer shows it, read again while it's going"""

    def system_context(self, *, instance_dir: Path) -> str:
        """The run's system prompt and tool definitions, or "" if not saved"""

    def teacher_container(
        self,
        *,
        workspace: Path,
        out_dir: Path,
        model: core.ModelSpec,
        endpoint: core.ModelEndpoint,
        name: str,
        terms: tuple[str, ...],
    ) -> AbstractContextManager[str]:
        """Start the teacher's container: workspace at /work (read-only), out_dir at /out"""

    def validate_variant(self, *, variant: Path, out_dir: Path, terms: tuple[str, ...]) -> list[str]:
        """A variant package's problems (the log goes in out_dir)"""

    def question_problems(
        self, *, variant: Path, out_dir: Path, model: core.ModelSpec, endpoint: core.ModelEndpoint
    ) -> list[str]:
        """How the variant gets in the way of a question that needs no change (CheckUnavailable if it can't ask)"""

    def cleanup(self) -> None:
        """Stop whatever is still running after an interrupt"""


def load(name: str) -> Harness:
    return folders.module(root=paths.harnesses_dir(), name=name, file="harness.py", kind="harness").HARNESS
