import dataclasses
from pathlib import Path
from typing import Protocol

from rsi import folders, paths, views


@dataclasses.dataclass(frozen=True)
class TeacherRun:
    status: str  # ok, timeout or error
    seconds: float
    error: str | None = None
    model_calls: int = 0
    tool_calls: int = 0  # 0 means the session did no work
    output_tokens: int = 0
    cost: str = ""  # in the teacher's own units


class Teacher(Protocol):
    """The agent that writes variants, with tools only in the teacher container"""

    name: str

    def describe(self, *, model: str, effort: str) -> str:
        """The teacher, its version, model and effort"""

    def run(self, *, prompt: str, model: str, effort: str, container: str, out_dir: Path, timeout_s: int) -> TeacherRun:
        """Run one session in the container, saving its record in out_dir"""

    def transcript(self, *, session_dir: Path) -> str:
        """The session turn by turn, for people to read"""

    def trajectory(self, *, session_dir: Path) -> views.Trajectory:
        """The session as the viewer shows it, read again while it's going"""


def load(name: str) -> Teacher:
    return folders.module(root=paths.teachers_dir(), name=name, file="teacher.py", kind="teacher").TEACHER
