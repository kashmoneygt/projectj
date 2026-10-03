import threading
import time
import types
from pathlib import Path

import conftest
import pytest
from fastapi.testclient import TestClient

from jym import core, environments, library, live, logits, players, web

ORIGIN = {"origin": "https://jym.test"}
SETTINGS = Path(__file__).parents[1] / "deploy" / "jym.toml"


def until(predicate, seconds=30):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def hosted_jym(tmp_path, settings: web.Settings) -> tuple[web.Jym, TestClient]:
    jym = web.Jym(settings=settings, runs_dir=tmp_path / "runs", library_folders=[])
    return jym, TestClient(web.create_app(jym=jym, hosts={"jym.test"}, secure=True), base_url="https://jym.test")


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    monkeypatch.setitem(environments.ENVIRONMENTS, "counter", conftest.SPEC)
    settings = web.Settings(players=["reference"], stream=["reach-3"], hosted=True)
    return hosted_jym(tmp_path, settings)


def headers(client_id):
    return {**ORIGIN, "x-jym-client": client_id}


def test_other_sites_are_refused(hosted):
    _, client = hosted
    response = client.get("/api/catalog")
    assert response.headers["content-security-policy"] == "frame-ancestors 'self'"
    assert client.get("/api/catalog", headers={"host": "evil.example"}).status_code == 400
    assert client.post("/api/stop", headers={"origin": "http://jym.test", "x-jym-client": "a" * 16}).status_code == 403


def test_rate_limit_follows_the_caller(hosted):
    _, client = hosted
    forged = [
        {**headers(client_id=f"forged-{i:010}"), "x-forwarded-for": f"10.0.0.{i % 250}, 203.0.113.9"}
        for i in range(121)
    ]
    assert [client.post("/api/stop", headers=sent).status_code for sent in forged][-2:] == [200, 429]


def test_each_visitor_gets_a_room(hosted):
    jym, client = hosted
    alice, bob = "alice-0123456789", "bob-0123456789ab"
    start = {"mode": "controls", "env": "counter", "task": "reach-3"}
    assert client.post("/api/start", json=start, headers=headers(alice)).status_code == 200
    until(lambda: jym.room(alice).state["session"]["state"] == "running")
    assert jym.room(bob).session is None and jym.live.session is None
    assert client.post("/api/start", json=start, headers=ORIGIN).status_code == 409  # no room without a client id
    assert client.post("/api/start", json={**start, "mode": "questions"}, headers=headers(bob)).status_code == 409
    client.post("/api/stop", headers=headers(client_id="carol-0123456789"))
    assert "carol-0123456789" not in jym.rooms  # stopping claims no seat
    jym.room(alice).session.move({"speed": 1})
    until(lambda: jym.room(alice).state["session"]["state"] == "ended")
    assert jym.room(alice).state["session"]["result"]["status"] == "success"


def test_the_stream_races_its_players(hosted, monkeypatch):
    monkeypatch.setattr(live, "START_SECONDS", 0.1)
    jym, _ = hosted
    heat = jym.director.next_heat()
    assert (heat.task, heat.players, heat.seed in library.LEADERBOARD_SEEDS) == ("reach-3", ["reference"], True)
    assert jym.director.upcoming.seed != heat.seed  # the next heat flies another seed nobody has flown
    jym.director.play(heat)
    assert (heat.entries["reference"]["state"], heat.entries["reference"]["status"]) == ("done", "success")
    decision, step = jym.live.state["decisions"]["reference"]  # the last decision, with the step that applied it
    assert (decision["type"], step["type"], step["step"]) == ("decision", "step", decision["step"])
    again = live.Heat(env="counter", task="reach-3", seed=heat.seed, players=["reference"])
    jym.director.play(again)
    assert again.entries["reference"]["state"] == "recorded"  # a seed it has flown is replayed


def test_heats_fly_the_seeds_recorded_players_have_flown(hosted, monkeypatch):
    monkeypatch.setattr(live, "START_SECONDS", 0.1)
    director = hosted[0].director
    director.play(live.Heat(env="counter", task="reach-3", seed=1, players=["reference"]))
    director.players, director.recorded = ["idler"], ["reference"]
    heat = director.next_heat()
    assert (heat.seed, heat.players) == (1, ["idler", "reference"])  # the reference has only flown seed 1


def test_replays_play_until_the_last_success(hosted, monkeypatch):
    monkeypatch.setattr(live, "START_SECONDS", 0.1)
    director = hosted[0].director
    found = {"run_id": "r", "status": "success", "score": 100, "sim_seconds": 1.0}
    monkeypatch.setattr(director, "recording", lambda task, seed, player: found)
    started = time.monotonic()
    director.play(live.Heat(env="counter", task="reach-3", seed=1, players=["reference"]))
    assert time.monotonic() - started >= 1.1  # nobody flies live, yet the heat lasts as long as the replay


def test_viewers_choose_the_next_objective(tmp_path):
    jym, client = hosted_jym(
        tmp_path, web.Settings(players=["reference"], stream=["rings", "drone-rings"], hosted=True)
    )
    assert {"env": "flight", "task": "drone-rings"} in client.get("/api/catalog").json()["stream"]
    for task in ["drone-rings", "rings", "drone-rings"]:
        assert client.post("/api/next", json={"tasks": [task]}, headers=ORIGIN).status_code == 200
        assert jym.director.upcoming.task == task
    assert client.post("/api/next", json={"tasks": ["circuit"]}, headers=ORIGIN).status_code == 409  # not streamed


class Idler:
    name = "idler"

    def decide(self, state, questions):
        return core.picked(step="stay")


def test_stream_runs_only_while_watched(hosted, monkeypatch):
    monkeypatch.setattr(live, "START_SECONDS", 0.1)
    monkeypatch.setattr(live, "UNWATCHED_SECONDS", 0.5)
    jym, client = hosted
    idler = players.Player(id="idler", title="Idler", make=lambda environment: Idler(), per_run=True)
    monkeypatch.setitem(jym.registry, "idler", idler)
    director = jym.director
    assert not director.watched()
    with client.websocket_connect(f"/ws?client={'v' * 16}&room=live", headers={**ORIGIN, "host": "jym.test"}) as tab:
        assert tab.receive_json()["here"] == 1
        until(director.watched)
        tab.send_json({"type": "visibility", "visible": False})
        until(lambda: not director.watched())  # a tab in the background does not count
    heat = live.Heat(env="counter", task="reach-3", seed=8, players=["idler"])
    started = time.monotonic()
    director.play(heat)
    assert heat.entries["idler"]["reason"] == library.CUT  # nobody watches, so it is cut
    assert time.monotonic() - started < 3  # well within the 5 s its objective allows


def test_heats_off_the_leaderboard_are_cut(hosted, monkeypatch):
    monkeypatch.setattr(live, "GRACE_SECONDS", 1.0)
    director = hosted[0].director
    heat = live.Heat(env="counter", task="reach-3", seed=1234, players=["reference", "maverick@cpu"])
    heat.entries = {"reference": {"status": "success", "sim_seconds": 4.0}, "maverick@cpu": {"state": "running"}}

    def cut(seconds: float, seed: int = 1234) -> bool:
        director.cut.clear()
        heat.seed, heat.started = seed, time.time() - seconds
        director.enforce(heat)
        return director.cut.is_set()

    assert not cut(seconds=5) and cut(seconds=7)  # the first finisher took 4 s, so the others get until 6 s
    assert not cut(seconds=7, seed=0)  # on a leaderboard seed every player uses its whole budget
    director.limits = {"reach-3": 10.0}
    heat.entries = {"reference": {"state": "running"}}
    assert not cut(seconds=9) and cut(seconds=11)  # nobody finished within the objective's time limit


def test_ready_once_the_model_loads(tmp_path, monkeypatch):
    loaded = threading.Event()
    monkeypatch.setattr(logits, "LogitsPolicy", lambda name, *_: loaded.wait(10) and types.SimpleNamespace(name=name))
    jym, client = hosted_jym(tmp_path, web.Settings(players=["qwen-0.6b@cpu", "reference"], hosted=True))
    jym.model(name="qwen-0.6b@cpu")
    probe = {"host": "10.0.0.7:8080"}  # the platform's probe uses the container's address
    assert client.get("/api/health?ready=1", headers=probe).status_code == 503
    assert client.get("/api/health").json()["ready"] is False  # still up
    loaded.set()
    until(jym.loaded)
    assert client.get("/api/health?ready=1", headers=probe).json()["ready"] is True


def test_players_are_named_by_model_or_id(tmp_path):
    jym, _ = hosted_jym(tmp_path, web.Settings(players=["maverick", "qwen-0.6b@gpu", "reference"], recorded=["jev"]))
    assert (jym.director.players, jym.director.recorded) == (["maverick@cpu", "qwen-0.6b@gpu", "reference"], ["jev"])


def test_settings_file_is_checked(tmp_path):
    settings = web.Settings.load(SETTINGS)
    assert settings.hosted
    director = web.Jym(settings=settings, runs_dir=tmp_path / "runs").director
    heats = [director.pick() for _ in range(40)]
    assert {heat.task for heat in heats} == set(settings.stream)
    assert {heat.seed for heat in heats} <= set(library.LEADERBOARD_SEEDS)  # the seeds its recorded players flew
    path = tmp_path / "jym.toml"
    for text, problem in [
        ('players = ["reference"]\ncolour = "red"', "unknown settings"),
        ('players = ["nobody"]', "unknown player"),
        ('players = ["reference"]\nrecorded = ["nobody"]', "unknown player"),
        ('players = ["reference"]\nplay = "loop"', "unknown objective"),
        ('players = ["reference"]\nstream = ["rings", "loop"]', "unknown objective"),
        ('players = ["reference"]\nheat_minutes = { loop = 3 }', "unknown objective"),
        ('players = ["reference"]\nheat_minutes = { rings = 0 }', "must be positive"),
    ]:
        path.write_text(text)
        with pytest.raises(ValueError, match=problem):
            web.Settings.load(path)
