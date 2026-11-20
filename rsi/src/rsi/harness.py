import dataclasses
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from rsi import core, folders, paths, views

TASK_FILE = "task.md"  # the task message the agent got, saved in each run folder
VERIFIER_DIR = "verifier"  # subfolder of a run folder for the grader's logs, copied out of the task's container
GRADING_FILE = "grading.json"  # result of grading in the task's container, e.g. exit code and whether it timed out
# where the agent's final answer goes in the task's container after the run, for graders that check an answer
REPLY_FILE = "/logs/agent/response.txt"


@dataclasses.dataclass(frozen=True)
class RunContext:
    instance: core.Instance
    out_dir: Path
    model: core.ModelSpec
    endpoint: core.ModelEndpoint
    budget: core.Budget
    variant: Path | None = None  # variant package folder, None for the baseline


class CheckUnavailable(Exception):
    """Raised when a check couldn't run (e.g. the model or Docker failed), so it says nothing about the variant"""


class Harness(Protocol):
    name: str
    version: str
    package_dir: Path  # the baseline package
    doc: Path  # guide to writing a variant, which the teacher reads as harness.md
    protocol_terms: tuple[str, ...]  # words about grading and the runner's paths, banned from files the agent reads

    def prepare(self, instances: list[core.Instance]) -> None: ...
    def run(self, ctx: RunContext) -> core.TaskRun:
        """Run one task after its setup commands, and save the task message as TASK_FILE in ctx.out_dir"""
        # if the task has a verifier, grade it in the task's container once the agent is done. Write the agent's final
        # answer to REPLY_FILE first, and save the grader's logs in ctx.out_dir/VERIFIER_DIR

    def check_grader(self, *, instance: core.Instance, out_dir: Path, solve: bool) -> None:
        """Run the task's grader after its reference solution (if solve) or after doing nothing, same as in a run"""

    def contract(self) -> str:
        """Hash of what every run gets besides the variant"""

    def records_activations(self, *, package: Path) -> bool:
        """Check if a variant logs when it kicks in, so we can count how often it did"""

    def run_files(self, *, instance_dir: Path) -> dict[str, Path]:
        """Run files the teacher reads, keyed by the name the teacher sees, with nothing that says if the run passed"""

    def trajectory(self, *, instance_dir: Path) -> views.Trajectory:
        """The run as the viewer shows it, also called on a run that's still going"""

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
        """Start the teacher's container with check-variant and try-variant, and yield its name"""
        # workspace is mounted read-only at /work, and out_dir at /out

    def validate_variant(self, *, variant: Path, out_dir: Path, terms: tuple[str, ...]) -> list[str]:
        """Check a variant package and return its problems, saving the check's log in out_dir"""

    def question_problems(
        self, *, variant: Path, out_dir: Path, model: core.ModelSpec, endpoint: core.ModelEndpoint
    ) -> list[str]:
        """Problems the variant causes on a question that needs no change (raises CheckUnavailable if it can't ask)"""

    def cleanup(self) -> None:
        """Stop whatever is still running after an interrupt"""


def load(name: str) -> Harness:
    return folders.module(root=paths.harnesses_dir(), name=name, file="harness.py", kind="harness").HARNESS
