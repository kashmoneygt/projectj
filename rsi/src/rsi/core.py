import dataclasses
import time
import tomllib
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class DatasetSpec(Frozen):
    name: str
    dataset_id: str
    split: str = "test"
    revision: str = ""  # pinned dataset revision, e.g. an HF revision or a git commit
    request: str = "{task}"  # template for the message the agent gets; {task} becomes the task text
    setup: str = ""  # shell commands run in the task's repo before the agent starts

    @property
    def lead(self) -> str:
        """Text every task message starts with, i.e. the request up to {task}"""
        return self.request.partition("{task}")[0]


class ModelSpec(Frozen):
    """Model serving parameters (from models.toml)"""

    name: str
    # weights to serve, as "hf:<org>/<name>/<file>", "path:/abs/file.gguf" or "splash:OWNER/REPO[:VARIANT]"
    source: str
    model_id: str  # name the server serves the model under
    backend: str = "llama.cpp"  # a folder under backends/
    port: int = 11500
    ctx: int = 32768  # context window in tokens (input + output)
    parallel: int = 1  # requests the server runs at once
    memory_gb: int = 0  # memory cap for the server, 0 = no cap
    seed: int = 42  # sampling seed
    thinking: bool = False
    max_output_tokens: int = 0  # cap per response, 0 = no cap beyond the context window
    sampling: dict[str, Any] = Field(default_factory=dict)  # sent with every request
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)  # for the server's chat template

    @property
    def output_cap(self) -> int:
        return self.max_output_tokens or self.ctx


@dataclasses.dataclass(frozen=True)
class ModelEndpoint:
    model_id: str
    base_url: str  # OpenAI-compatible URL the agent calls
    root_url: str = ""  # server root, for /tokenize


class ModelBackend(Protocol):
    """A model server on the host machine"""

    spec: ModelSpec

    def is_running(self) -> bool: ...
    def start(self) -> None: ...
    def wait_ready(self, timeout: float = 300.0) -> None: ...
    def stop(self) -> None: ...
    def host_url(self) -> str: ...
    def endpoint(self) -> ModelEndpoint: ...
    def status_lines(self) -> list[str]: ...
    def identity(self) -> dict[str, str]: ...  # version, weights and build, saved on every node
    def healthy(self, timeout_s: float) -> bool: ...  # True if the server answers a one-token request in time


class Budget(Frozen):
    timeout_s: int = 3600  # per task, then stop the agent and grade what it left
    max_calls: int = 0  # model requests per task, then stop the agent and grade what it left (0 = no limit)


class Verifier(Frozen):
    """Harbor-style grading that runs in the task's container after the agent is done"""

    tests: str  # host folder copied to /tests right before grading
    command: str  # shell command run in the workdir, with its output saved in logs
    timeout_s: int = 600
    internet: bool = False
    env: dict[str, str] = Field(default_factory=dict)
    logs: str = "/logs/verifier"  # container folder copied back into verifier/ in the task's run folder
    results: tuple[str, ...] = ("reward.json", "reward.txt")  # grader leaves one of these in logs, or grading failed


class Instance(Frozen):
    instance_id: str
    repo: str  # repo the task's code comes from, used to spread samples across repos
    image: str  # Docker image the task runs and is graded in
    workdir: str  # folder the agent starts in, "" = the image's WORKDIR
    task: str  # message the agent gets, worded like a user's request
    setup: str = ""  # shell commands run in the workdir before the agent starts
    login: bool = False  # start the agent in a login shell, for images that set up tools in shell profiles
    internet: bool = False  # whether the agent can reach the internet
    cpus: float = 0.0  # 0 = no limit
    memory_mb: int = 0  # 0 = no limit
    env: dict[str, str] = Field(default_factory=dict)  # env vars for the task's container
    verifier: Verifier | None = None  # None if the benchmark grades the diff itself
    solution: str = ""  # host folder with a reference solution, copied to /solution only to check the grader
    kind: str = ""  # kind of work, so results can be split by kind


class ToolStat(BaseModel):
    count: int = 0
    errors: int = 0
    ms: int = 0


class TaskMetrics(BaseModel):
    """Metrics for one task, across all model calls in its run"""

    requests: int = 0
    input_tokens: int = 0  # uncached prompt tokens
    output_tokens: int = 0  # thinking included
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    tool_ms: int = 0
    tool_stats: dict[str, ToolStat] = Field(default_factory=dict)
    tok_s: float = 0.0  # output tokens per second, not counting tool time
    context_pct: float = 0.0  # largest prompt as a % of the context window
    # token breakdown of the largest request
    system_tokens: int = 0
    tool_def_tokens: int = 0
    message_tokens: int = 0
    ttft_first_ms: float = 0.0  # TTFT of the first request, cold start included
    ttft_avg_ms: float = 0.0
    ttft_max_ms: float = 0.0
    model_ms: int = 0  # generation time, thinking included
    think_ms: int = 0  # part of model_ms before the first content token
    stop_reason: str = ""  # stop reason of the last request (stop, length, toolUse, ...)
    patch_lines: int = 0  # non-empty lines in the final diff
    peak_output_tokens: int = 0  # largest single response, thinking included
    capped_requests: int = 0  # responses cut off at the output cap
    acted: int = 0  # times the variant logged that it kicked in
    # ways the model requests broke the pinned settings or the call budget
    contract_violations: list[str] = Field(default_factory=list)


class TaskRun(BaseModel):
    instance_id: str
    status: str = "pending"  # pending, ok, timeout or error
    patch: str = ""
    seconds: float = 0.0
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    images: dict[str, str] = Field(default_factory=dict)  # image tag -> sha256 digest
    error: str | None = None
    resolved: bool = False


class Node(BaseModel):
    """One run of a variant on a set of tasks, saved in runs/<name>/"""

    name: str
    parent: str | None = None
    title: str | None = None  # title from the variant card
    dataset: DatasetSpec
    harness: str
    harness_version: str
    variant: str = ""  # name of a variant in variants/, a package path, or "" for the baseline
    model: ModelSpec
    budget: Budget
    tasks: int
    seed: int
    task_ids: list[str] = Field(default_factory=list)  # selected tasks, checked again on resume
    kinds: dict[str, str] = Field(default_factory=dict)  # task id -> kind of work, for tasks that have one
    machine: str = ""
    # server identity, harness contract and package digest, so runs made differently are never compared
    server: dict[str, str] = Field(default_factory=dict)
    contract: str = ""
    variant_digest: str = ""
    resolved: int = 0
    total: int = 0
    score: float = 0.0
    runs: dict[str, TaskRun] = Field(default_factory=dict)
    created: float = Field(default_factory=time.time)


class TeacherSpec(Frozen):
    """Teacher settings (from models.toml)"""

    name: str
    agent: str  # agent CLI, a folder under teachers/
    model: str
    effort: str = ""  # reasoning effort


def load_teachers(path: Path) -> list[TeacherSpec]:
    tables = tomllib.loads(path.read_text()).get("teachers", {})
    return [TeacherSpec(name=name, **cfg) for name, cfg in tables.items()]


def load_models(path: Path) -> list[ModelSpec]:
    tables = tomllib.loads(path.read_text()).get("models", {})
    return [ModelSpec.model_validate({"model_id": name, **cfg, "name": name}) for name, cfg in tables.items()]
