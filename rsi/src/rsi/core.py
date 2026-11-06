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
    revision: str = ""  # pinned HF dataset revision
    request: str = "{task}"  # how a task is put to the agent; {task} is the task's own text


class ModelSpec(Frozen):
    """Model serving parameters (from models.toml)"""

    name: str
    # "hf:<org>/<name>/<file>", "path:/abs/file.gguf" or "splash:OWNER/REPO[:VARIANT]"
    source: str
    model_id: str  # the name the server serves it under
    backend: str = "llama.cpp"  # a folder under backends/
    port: int = 11500
    ctx: int = 32768  # context window (input + output)
    parallel: int = 1  # requests the server runs at once
    seed: int = 42  # sampling seed
    thinking: bool = False  # thinking mode
    max_output_tokens: int = 0  # per response, 0 = the rest of the context window
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


class ServerStats(BaseModel):
    requests: int = 0
    cache_hits: list[float] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tok_s: float = 0.0

    @property
    def avg_cache_hit(self) -> float | None:
        return sum(self.cache_hits) / len(self.cache_hits) if self.cache_hits else None


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
    def log_stats(self) -> ServerStats | None: ...
    def identity(self) -> dict[str, str]: ...  # version, weights and build, saved on every node
    def healthy(self, timeout_s: float) -> bool: ...  # answers a one-token request in time


class Budget(Frozen):
    timeout_s: int = 3600  # per task, after which the run is scored as is


class Instance(Frozen):
    instance_id: str
    repo: str
    base_commit: str
    task: str  # the message the agent gets, as a user would write it


class ToolStat(BaseModel):
    count: int = 0
    errors: int = 0
    ms: int = 0


class TaskMetrics(BaseModel):
    """One task's metrics, over every model call its run made"""

    requests: int = 0
    input_tokens: int = 0  # uncached prompt tokens
    output_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    tool_ms: int = 0
    tool_stats: dict[str, ToolStat] = Field(default_factory=dict)
    final_event: str = ""  # last event type in the harness's stream
    tok_s: float = 0.0  # output tokens per second of non-tool time
    context_pct: float = 0.0  # largest prompt as a % of the context window
    # token breakdown of the largest request
    system_tokens: int = 0
    tool_def_tokens: int = 0
    message_tokens: int = 0
    ttft_first_ms: float = 0.0  # first request's, cold start included
    ttft_avg_ms: float = 0.0
    ttft_max_ms: float = 0.0
    model_ms: int = 0  # generation time, thinking included
    think_ms: int = 0  # part of model_ms before the first content token
    stop_reason: str = ""  # last request's (stop, length, toolUse, ...)
    patch_lines: int = 0  # non-empty lines in the final diff
    peak_output_tokens: int = 0  # largest single response, thinking included
    capped_requests: int = 0  # responses cut at the output cap
    acted: int = 0  # times the variant called pi.appendEntry("acted", ...)
    contract_violations: list[str] = Field(default_factory=list)  # requests off the pinned model settings


class TaskRun(BaseModel):
    instance_id: str
    status: str = "pending"  # pending, ok, timeout or error
    patch: str = ""
    trajectory: str = ""
    seconds: float = 0.0
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    images: dict[str, str] = Field(default_factory=dict)  # image tag -> sha256 digest
    error: str | None = None
    resolved: bool = False


class Node(BaseModel):
    """One run of a variant on a set of tasks, saved in runs/<name>/"""

    name: str
    parent: str | None = None
    title: str | None = None  # the variant card's title
    diff_summary: str | None = None  # files changed vs the parent's package
    dataset: DatasetSpec
    harness: str
    harness_version: str
    variant: str = ""  # a registered variant's name, another package's dir, or "" for the baseline
    model: ModelSpec
    budget: Budget
    tasks: int
    seed: int
    task_ids: list[str] = Field(default_factory=list)  # checked on resume
    machine: str = ""
    # what produced the runs, so runs made differently are never compared
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
