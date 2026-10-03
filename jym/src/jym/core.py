import dataclasses
import math
from collections.abc import Callable
from typing import Any, ClassVar, Literal, Protocol

import pydantic
from pydantic import BaseModel, ConfigDict, Field


class Unavailable(RuntimeError):
    """Raised when a player or environment can't run on this machine"""


class InvalidAnswer(RuntimeError):
    def __init__(self, message: str, reply: Any = None):
        super().__init__(message)
        self.reply = reply


class Task(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    env: str
    id: str
    title: str
    goal: str
    version: str
    milestones: dict[str, int] = Field(default_factory=dict)
    max_steps: int = Field(gt=0)
    max_seconds: float = Field(gt=0)
    settings: dict[str, Any] = Field(default_factory=dict)

    @pydantic.model_validator(mode="after")
    def _check_milestones(self):
        if any(points <= 0 for points in self.milestones.values()) or sum(self.milestones.values()) > 100:
            raise ValueError("Milestone points must be positive and total at most 100")
        return self


class Evaluation(BaseModel):
    success: bool = False
    milestones: set[str] = Field(default_factory=set)
    failure: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


@dataclasses.dataclass
class Reply:
    answers: dict[str, dict]
    info: dict[str, Any] = dataclasses.field(default_factory=dict)  # logged with the decision


Status = Literal["success", "failure", "truncated", "stopped", "error"]


class Result(BaseModel):
    run_id: str
    env: str
    task: str
    task_version: str
    policy: str
    seed: int
    status: Status
    score: int = Field(ge=0, le=100)
    steps: int
    seconds: float
    milestones: list[str] = Field(default_factory=list)
    reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class Policy(Protocol):
    name: str

    def decide(self, state: dict, questions: dict[str, dict]) -> Reply: ...


class Environment(Protocol):
    """An environment that continues to advance even if the policy has not acted on it"""

    version: str
    HZ: int  # physics ticks per second
    ticks: int  # ticks since reset, for exact replays
    RELEASED: ClassVar[dict]  # controls when a person lets go

    def reset(self, task: Task, seed: int) -> None: ...
    def state(self) -> dict: ...
    def questions(self) -> dict[str, dict]: ...
    def apply(self, choices: dict[str, str]) -> None: ...  # held until the next choices
    def hold(self) -> None: ...  # hold steady when the policy hasn't answered
    def control(self, values: dict) -> None: ...  # a person's controls
    def advance(self, seconds: float) -> bool: ...  # True once the episode is over
    def evaluate(self) -> Evaluation: ...  # scored from the game's own state
    def scene(self) -> dict: ...  # what the page draws
    def reference(self) -> Policy: ...  # the environment's scripted player
    def close(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class EnvironmentSpec:
    """An environment plug-in, registered under the jym.environments entry point"""

    id: str
    title: str
    about: str
    environment: Callable[[], Environment]
    tasks: list[Task]
    objective: str  # default objective in Play
    describe: Callable[[], dict]  # state fields, questions and timing
    question_sets: Callable[[str], list[dict]]  # all question sets of an objective, for warming prompt caches
    reference: dict[str, str]  # title, role, trained and deployed text of the scripted player
    pose: Callable[[dict], list[float]]  # a scene's time, position and attitude, for drawing ghosts
    catalog: dict = dataclasses.field(default_factory=dict)  # extra fields for the page's catalog


def choice(instructions: str, options: dict[str, str | None]) -> dict:
    if not 1 <= len(options) <= 8:
        raise ValueError(f"A choice needs 1-8 options, got {len(options)}")
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def picked(**choices: str) -> Reply:
    return Reply({qid: {"type": "choice", "choice": option} for qid, option in choices.items()})


def check_answers(questions: dict[str, dict], answers: Any) -> list[str]:
    """Validate answers, and warn about choices that weren't the most probable"""
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise InvalidAnswer(message=f"answers do not match questions {sorted(questions)}", reply=answers)
    warnings = []
    for qid, question in questions.items():
        answer = answers[qid]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise InvalidAnswer(message=f"{qid}: expected a choice answer", reply=answer)
        probabilities = answer.get("probabilities") or {}
        if not all(_is_probability(p) for p in probabilities.values()):
            raise InvalidAnswer(message=f"{qid}: probabilities must be numbers in [0, 1]", reply=answer)
        if answer.get("choice") not in question["criteria"]:
            raise InvalidAnswer(message=f"{qid}: {answer.get('choice')!r} is not an offered option", reply=answer)
        best = max(probabilities, key=probabilities.get, default=answer["choice"])
        if probabilities.get(answer["choice"], 1.0) < probabilities.get(best, 1.0):
            warnings.append(f"{qid}: chose {answer['choice']!r} although {best!r} is more probable")
    return warnings


def _is_probability(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def score(task: Task, evaluation: Evaluation) -> int:
    if evaluation.success:
        return 100
    return min(99, sum(task.milestones[name] for name in evaluation.milestones))


def status(evaluation: Evaluation | None, ending: str | None) -> Status:
    if evaluation is None or ending == "error":
        return "error"
    if evaluation.success:
        return "success"
    if evaluation.failure or ending is None:
        return "failure"
    return ending
