import importlib.metadata

import conftest

from jym import core, environments

TASK = core.Task(
    env="e", id="t", title="T", goal="g", version="1", milestones={"a": 40, "b": 60}, max_steps=5, max_seconds=5
)


def test_only_verified_success_scores_100():
    assert core.score(TASK, core.Evaluation(success=True)) == 100
    assert core.score(TASK, core.Evaluation(milestones={"a", "b"})) == 99
    assert core.score(TASK, core.Evaluation(milestones={"a"})) == 40


def test_status_prefers_success_then_failure():
    assert core.status(evaluation=core.Evaluation(success=True), ending="stopped") == "success"
    assert core.status(evaluation=core.Evaluation(failure="crashed"), ending="truncated") == "failure"
    assert core.status(evaluation=core.Evaluation(), ending="stopped") == "stopped"
    assert core.status(evaluation=core.Evaluation(), ending=None) == "failure"
    assert core.status(evaluation=None, ending=None) == "error"


def test_environments_are_plugins(monkeypatch):
    installed = importlib.metadata.entry_points
    plugin = importlib.metadata.EntryPoint("counter", "conftest:SPEC", "jym.environments")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [*installed(group=group), plugin])
    monkeypatch.setattr(environments, "ENVIRONMENTS", environments.discover())
    assert list(environments.ENVIRONMENTS) == ["flight", "counter"] and environments.default() == "flight"
    assert environments.get_task(env="counter") == conftest.COUNTER  # its default objective
