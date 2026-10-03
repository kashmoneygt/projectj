import time

import pytest
from fastapi.testclient import TestClient

from jym import web

CLIENT = "tab-0123456789"
HEADERS = {"origin": "http://testserver", "x-jym-client": CLIENT}


@pytest.fixture
def jym(tmp_path):
    return web.Jym(settings=web.Settings(players=[]), runs_dir=tmp_path, library_folders=[])


@pytest.fixture
def client(jym):
    return TestClient(web.create_app(jym=jym, hosts={"testserver"}))


def until(predicate, seconds=60):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def start(client, mode):
    return client.post("/api/start", json={"mode": mode, "env": "flight", "task": "takeoff"}, headers=HEADERS)


def answer(client, step, choices):
    return client.post("/api/answer", json={"step": step, "choices": choices}, headers=HEADERS)


def stop(jym, client) -> dict:
    assert client.post("/api/stop", headers=HEADERS).json() == {"stopped": True}
    until(lambda: not jym.room(CLIENT).running())
    return jym.room(CLIENT).state["session"]["result"]


def test_answer_the_questions_yourself(jym, client):
    start(client=client, mode="questions")
    room = jym.room(CLIENT)
    until(lambda: room.state["pending"] is not None)
    pending = room.state["pending"]
    assert pending["state"]["goal"] and room.state["session"]["state"] == "waiting for you"
    choices = {qid: next(iter(question["criteria"])) for qid, question in pending["questions"].items()}
    assert answer(client, pending["step"] + 1, choices).status_code == 409  # not the question waiting
    assert answer(client, pending["step"], {**choices, "brakes": "maybe"}).status_code == 409  # not an option
    assert answer(client, pending["step"], choices).json() == {"accepted": True}
    until(lambda: (room.state["pending"] or {}).get("step") == pending["step"] + 1)
    decision, step = room.state["decision"]  # the decision, with the step that applied it
    assert (decision["answers"]["brakes"]["choice"], step["step"]) == (choices["brakes"], pending["step"])
    assert stop(jym, client)["status"] == "stopped"


def test_fly_with_the_controls(jym, client):
    start(client=client, mode="controls")
    room = jym.room(CLIENT)
    until(lambda: room.state["session"]["state"] == "running" and room.state["scene"])
    with client.websocket_connect(f"/ws?client={CLIENT}", headers=HEADERS) as socket:
        assert socket.receive_json()["session"]["mode"] == "controls"
        socket.send_json({"type": "controls", "values": {"brake": 0, "throttle": 1}})
        until(lambda: room.state["scene"]["ground_speed_kt"] > 2)
    assert room.state["scene"]["controls"]["throttle"] == 1
    assert stop(jym, client)["status"] == "stopped"


def test_stop_ends_the_run(jym, client):
    start(client=client, mode="questions")
    until(lambda: jym.room(CLIENT).state["pending"] is not None)
    refused = start(client=client, mode="controls")
    assert refused.status_code == 409 and "in progress" in refused.json()["detail"]
    result = stop(jym, client)
    assert (result["status"], result["reason"]) == ("stopped", "stop requested")
    assert start(client=client, mode="controls").status_code == 200
    assert stop(jym, client)["status"] == "stopped"


def test_runs_list_newest_first(jym, client):
    start(client=client, mode="questions")  # a person's run, and the reference player's to race as a ghost
    until(lambda: [ghost["player"] for ghost in jym.room(CLIENT).state["ghosts"]] == ["reference"])
    stop(jym, client)
    runs = client.get("/api/runs").json()
    assert sorted(run["player"] for run in runs) == ["human", "reference"]
    assert [run["created"] for run in runs] == sorted((run["created"] for run in runs), reverse=True)
    ghost = next(run for run in runs if run["player"] == "reference")
    assert (ghost["status"], ghost["task"]) == ("success", "takeoff")
    assert client.get(f"/api/runs/{ghost['run_id']}").json()["result"]["status"] == "success"
