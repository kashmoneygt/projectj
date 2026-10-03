import time

import conftest

from jym import core, environments, library, runner, runs
from jym.environments.flight import cessna


class Script:
    name = "script"

    def __init__(self, *moves):
        self.moves = list(moves)
        self.metadata = {"url": "https://api.example/?key=sk-very-secret-value"}

    def decide(self, state, questions):
        return core.picked(step=self.moves.pop(0))


def test_secrets_are_redacted_as_escaped_in_json(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TOKEN", 'multi\nline "secret"')
    assert runs.RunLog(tmp_path).redact(data={"token": 'multi\nline "secret"'}) == '{"token": "[REDACTED]"}'


def test_lockstep_run_logs_every_exchange(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "sk-very-secret-value")
    counter, log = conftest.Counter(), runs.RunLog(tmp_path)
    result = runner.run_lockstep(
        environment=counter, policy=Script("up", "stay", "up", "up"), task=conftest.COUNTER, seed=3, log=log
    )
    assert (result.status, result.score, result.steps, result.milestones) == ("success", 100, 4, ["one", "two"])
    run = runs.load_run(run_id=log.run_id, runs_dir=tmp_path)
    assert (run["manifest"]["format"], run["manifest"]["seed"]) == (runs.FORMAT, 3)
    assert run["result"] == result.model_dump(mode="json")
    assert [event["type"] for event in run["events"]] == ["reset", *["decision", "step"] * 4, "evaluation", "end"]
    decision = run["events"][1]
    assert decision["state"] == {"goal": conftest.COUNTER.goal, "value": 0}
    assert decision["answers"] == {"step": {"type": "choice", "choice": "up"}}
    manifest = (log.path / "manifest.json").read_text()
    assert "sk-very-secret-value" not in manifest and "[REDACTED]" in manifest and counter.closed
    with (log.path / "events.jsonl").open("a") as file:
        file.write('{"type": "sce')  # a live run's line still being written
    assert runs.load_run(run_id=log.run_id, runs_dir=tmp_path)["events"] == run["events"]


class Late:
    name = "late"

    def __init__(self):
        self.answered = 0

    def decide(self, state, questions):
        time.sleep(0.3 if self.answered == 0 else 0.1)  # the first answer comes after the watchdog steps in
        self.answered += 1
        answers = {
            "brakes": "off",
            "throttle": "full",
            "turn": "straight",
            "lift_off": "lift off" if state["airspeed_kt"] >= cessna.LIFTOFF_KT else "not yet",
            "climb": "climb",
            "speed": "80 kt",
        }
        return core.picked(**{qid: answers[qid] for qid in questions})


def test_realtime_never_waits_for_the_policy(tmp_path):
    log = runs.RunLog(tmp_path)
    task = environments.get_task(env="flight", task="takeoff")
    runner.run_realtime(
        environment=environments.create(env="flight"), policy=Late(), task=task, seed=1, log=log, speed=20
    )
    events = runs.load_run(run_id=log.run_id, runs_dir=tmp_path)["events"]
    applied = [event for event in events if event["type"] == "decision" and event["applied_at"] is not None]
    assert all(decision["applied_at"] > decision["asked_at"] for decision in applied)
    assert "watchdog" in [event["type"] for event in events]
    entry = library.record(run_id=log.run_id, runs_dir=tmp_path)
    assert entry["result"]["milestones"] == ["takeoff"] and library.verify(entry) is None
