import json
import shutil

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from jym import cli, environments, library, runs, web

RECORDS = web.LIBRARY / "flight"


@pytest.mark.parametrize("name", ["circuit-v6-seed2-jev.json", "drone-rings-v6-seed0-jev.json"])
def test_library_runs_replay_to_their_score(name):
    assert library.verify(library.load_file(RECORDS / name)) is None


def test_recorded_runs_go_on_the_leaderboard(tmp_path):
    arguments = ["library", "record", "--tasks", "drone-takeoff", "--seeds", "0,1,2"]
    arguments += ["--out", str(tmp_path / "library"), "--runs-dir", str(tmp_path / "runs")]
    first = CliRunner().invoke(cli.app, arguments)
    assert first.exit_code == 0 and "reference drone-takeoff seed 2: success 100" in first.output
    again = CliRunner().invoke(cli.app, arguments)
    assert again.exit_code == 0 and "seed 0:" not in again.output  # nothing left to record
    folder, version = tmp_path / "library" / "flight", environments.get_task(env="flight", task="drone-takeoff").version
    names = [f"drone-takeoff-v{version}-seed{seed}-reference.json" for seed in range(3)]
    assert sorted(path.name for path in folder.iterdir()) == names
    board = library.Library(runs_dir=tmp_path / "empty", folders=[tmp_path / "library"]).leaderboard(
        env="flight", title=str, models={}
    )
    (row,) = next(task for task in board["tasks"] if task["task"] == "drone-takeoff")["rows"]
    assert (row["player"], row["seeds"], row["ranked"], row["imported"]) == ("reference", [0, 1, 2], True, True)
    entry = library.load_file(folder / names[2])
    changed = {**entry, "result": {**entry["result"], "score": 99}}
    assert library.verify(changed) == "replayed it scores 100 ['takeoff'], recorded 99"  # so export refuses it


def finished(runs_dir, player, seed, score, seconds=0.5, waited=0.0, model=None) -> None:
    created = len(list(runs_dir.iterdir()))  # each run newer than the ones before
    folder = runs_dir / f"{created:032x}"
    folder.mkdir()
    manifest = {
        "format": runs.FORMAT,
        "mode": "realtime",
        "created": float(created),
        "task": environments.get_task(env="flight", task="takeoff").model_dump(),
        "seed": seed,
        "policy": {"name": player, "metadata": {"model": model}},
    }
    metrics = {"decision_seconds_p50": seconds, "waited_seconds": waited}
    result = {"policy": player, "status": "success" if score == 100 else "failure", "score": score, "steps": 60}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "result.json").write_text(json.dumps({**result, "metrics": metrics}))


def test_leaderboard_ranks_players(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for seed in library.LEADERBOARD_SEEDS:
        finished(runs_dir=runs_dir, player="reference", seed=seed, score=100, seconds=0.001)
        finished(
            runs_dir=runs_dir, player="maverick@cpu", seed=seed, score=100, waited=0.004, model="maverick@new"
        )  # lock overhead isn't waiting
        finished(runs_dir=runs_dir, player="qwen-0.6b@cpu", seed=seed, score=20 if seed == 2 else 100)
        finished(runs_dir=runs_dir, player="human", seed=seed, score=100, seconds=None)
    finished(runs_dir=runs_dir, player="jev", seed=0, score=100)
    finished(
        runs_dir=runs_dir, player="maverick@cpu", seed=1, score=0, model="maverick@old"
    )  # a version it no longer serves
    finished(
        runs_dir=runs_dir, player="qwen-0.6b@cpu", seed=2, score=100, waited=2.0
    )  # slowed down by another model on the same CPU
    finished(runs_dir=runs_dir, player="human", seed=0, score=20)  # people keep their best run
    board = library.Library(runs_dir).leaderboard(env="flight", title=str, models={"maverick@cpu": "maverick@new"})
    rows = next(task for task in board["tasks"] if task["task"] == "takeoff")["rows"]
    ranking = [(row["player"], row["ranked"], row["successes"]) for row in rows]
    assert ranking == [
        ("reference", True, 3),
        ("maverick@cpu", True, 3),
        ("human", True, 3),
        ("qwen-0.6b@cpu", True, 2),
        ("jev", False, 1),
    ]


def test_library_run_is_replayed(tmp_path):
    record = RECORDS / "drone-takeoff-v6-seed0-jev.json"
    (tmp_path / "library").mkdir()
    shutil.copy(record, tmp_path / "library")
    entry = library.load_file(record)
    jym = web.Jym(settings=web.Settings(players=[]), runs_dir=tmp_path / "runs", library_folders=[tmp_path / "library"])
    client = TestClient(web.create_app(jym=jym, hosts={"testserver"}))
    run = client.get(f"/api/runs/{entry['run_id']}").json()
    decisions = [event for event in run["events"] if event["type"] == "decision"]
    assert len(decisions) == sum(not decision.get("hold") for decision in entry["decisions"])
    assert all(decision["state"]["goal"] == entry["task"]["goal"] for decision in decisions)
    last = next(event["scene"] for event in reversed(run["events"]) if event["type"] == "scene")
    position = [last["position"][key] for key in ("north_m", "east_m", "altitude_m")]
    assert position == pytest.approx(entry["trajectory"][-1][1:4], abs=0.06)  # the recorded run's last position
    assert client.get(f"/api/runs/{'0' * 32}").status_code == 404
