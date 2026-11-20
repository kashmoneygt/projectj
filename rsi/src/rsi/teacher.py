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
    """The agent that writes variants, using tools that only act inside the teacher's container"""

    name: str

    def describe(self, *, model: str, effort: str) -> str:
        """Short description of the teacher with its version, model and effort"""

    def run(self, *, prompt: str, model: str, effort: str, container: str, out_dir: Path, timeout_s: int) -> TeacherRun:
        """Run one session in the container and save its record, ending each tool result with the time left"""

    def transcript(self, *, session_dir: Path) -> str:
        """The session turn by turn as markdown, for people to read"""

    def trajectory(self, *, session_dir: Path) -> views.Trajectory:
        """The session as the viewer shows it, also called on a session that's still going"""


def load(name: str) -> Teacher:
    return folders.module(root=paths.teachers_dir(), name=name, file="teacher.py", kind="teacher").TEACHER
