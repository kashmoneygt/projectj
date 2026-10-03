import itertools
from pathlib import Path

import pytest

from jym import environments, logits, runner
from jym.environments.flight import airfield, cessna, drone, environment, objectives

TASKS = environments.tasks(env="flight")


def reset(name: str, seed: int = 0, **settings) -> environment.FlightEnvironment:
    task = TASKS[name]
    flight_env = environment.FlightEnvironment()
    flight_env.reset(task.model_copy(update={"settings": {**task.settings, **settings}}), seed)
    return flight_env


def act(flight_env: environment.FlightEnvironment, choices: dict) -> None:
    flight_env.apply(choices)
    flight_env.advance(runner.DECISION_SECONDS)


def fly(flight_env: environment.FlightEnvironment, until=lambda: False) -> None:
    autopilot = flight_env.reference()
    for _ in range(flight_env.task.max_steps):
        if until() or flight_env.evaluator.ending:
            return
        reply = autopilot.decide(flight_env.state(), flight_env.questions())
        act(flight_env=flight_env, choices={qid: answer["choice"] for qid, answer in reply.answers.items()})


def sample(t: float, north: float, agl: float, climb: float = 0.0, speed_kt: float = 70.0, struck=()) -> dict:
    return {
        "t": t,
        "north_m": north,
        "east_m": 0.0,
        "altitude_m": airfield.RUNWAY["elevation_m"] + agl,
        "agl_m": agl,
        "airspeed_kt": speed_kt,
        "ground_speed_kt": speed_kt,
        "climb_fpm": climb,
        "bank_deg": 0.0,
        "pitch_deg": 0.0,
        "heading_deg": 0.0,
        "on_ground": agl <= airfield.PARKED_AGL_M,
        "nose_gear_first": False,
        "struck": list(struck),
    }


def touch_down(name: str, north: float, speed_kt: float = 0.0) -> objectives.Evaluator:
    evaluator = (drone.DroneEvaluator if name.startswith("drone") else cessna.CessnaEvaluator)(
        TASKS[name], [], objectives.WAYPOINTS
    )
    evaluator.place_in_air(
        waypoint=len(objectives.WAYPOINTS), s=sample(t=0, north=north - 100, agl=50)
    )  # after flying the circuit's route
    evaluator.update(sample(t=1, north=north, agl=airfield.PARKED_AGL_M + 0.5, climb=-100, speed_kt=speed_kt))
    evaluator.update(sample(t=2, north=north, agl=airfield.PARKED_AGL_M, speed_kt=0.0))
    return evaluator


def test_goals_and_questions_are_as_trained():
    # Maverick 1.0 was trained on exactly these (see `jym describe flight`)
    golden = (Path(__file__).parent / "flight.md").read_text(encoding="utf-8")
    assert environments.markdown(environments.describe(env="flight")) == golden


def test_state_reads_as_trained():
    assert logits.render_state(reset(name="takeoff").state()) == (
        "situation: stopped on the runway\n"
        "airspeed_kt: 6\n"
        "ground_speed_kt: 0\n"
        "height_m: 0\n"
        "vertical_speed_fpm: 0\n"
        "pitch: nose 1 deg down\n"
        "bank: wings level\n"
        "heading_deg: 0\n"
        "stall_warning: no\n"
        "current_controls: brakes=on, throttle=idle, turn=straight, lift_off=not yet\n"
        "target: runway_left_m=1160, centreline=on it, direction=straight ahead\n"
        "wind: from 304 deg at 6.1 kt"
    )
    state = reset(name="drone-rings").state()
    assert logits.render_state({key: state[key] for key in ("current_controls", "target")}) == (
        "current_controls: turn=holding heading 0, climb=level, speed=hover\n"
        "target: ring=1 of 4, distance_m=1660, direction=straight ahead, height_m=90, "
        "height_difference=90 m above you, climb_needed_fpm=210"
    )


def test_brakes_hold_until_released():
    flight_env = reset(name="rings")
    for _ in range(9):
        act(
            flight_env=flight_env,
            choices={"brakes": "on", "throttle": "full", "turn": "straight", "lift_off": "not yet"},
        )
    assert flight_env.sample["ground_speed_kt"] < 1
    for _ in range(10):
        act(
            flight_env=flight_env,
            choices={"brakes": "off", "throttle": "full", "turn": "straight", "lift_off": "not yet"},
        )
    assert flight_env.sample["ground_speed_kt"] > 10
    assert (
        abs(airfield.wrap(flight_env.sample["heading_deg"] - airfield.RUNWAY["heading_deg"])) < 3
    )  # the nosewheel keeps it straight


def test_turn_moves_the_heading_held():
    flight_env = reset(name="rings")
    fly(flight_env=flight_env, until=lambda: flight_env.sample["agl_m"] > 60)
    assert list(flight_env.questions()) == ["turn", "climb", "speed"]
    start = flight_env.aircraft.heading_held()
    for _ in range(2):
        act(flight_env=flight_env, choices={"turn": "right 20°", "climb": "level", "speed": "80 kt"})
    assert flight_env.aircraft.heading_held() == pytest.approx((start + 40) % 360)
    for _ in range(40):
        act(flight_env=flight_env, choices={"turn": "straight", "climb": "level", "speed": "80 kt"})
    assert (
        abs(airfield.wrap(flight_env.sample["heading_deg"] - start - 40)) < 3 and abs(flight_env.sample["bank_deg"]) < 3
    )
    assert flight_env.state()["current_controls"]["turn"] == f"holding heading {round(start + 40) % 360}"
    assert abs(flight_env.sample["airspeed_kt"] - 80) < 5  # the autothrottle holds the chosen airspeed


def test_rings_count_inside_circle():
    evaluator = cessna.CessnaEvaluator(
        TASKS["rings"], objectives.course_rings(task=TASKS["rings"], seed=0), objectives.WAYPOINTS
    )
    ring = objectives.RINGS[0]
    height = airfield.PARKED_AGL_M + ring["height_m"]
    evaluator.update(sample(t=0, north=ring["north_m"] - 100, agl=height))
    evaluator.milestones.add("takeoff")
    evaluator.update(sample(t=1, north=ring["north_m"] - 50, agl=height + 80))
    evaluator.update(sample(t=2, north=ring["north_m"] + 50, agl=height + 80))  # over it
    evaluator.update(sample(t=3, north=ring["north_m"] - 50, agl=height))  # the wrong way through it
    assert evaluator.ring == 0
    evaluator.update(sample(t=4, north=ring["north_m"] + 50, agl=height))
    assert evaluator.ring == 1 and evaluator.milestones == {"takeoff", "ring_1"}


def test_hard_contact_is_a_crash():
    evaluator = cessna.CessnaEvaluator(TASKS["rings"], [], objectives.WAYPOINTS)
    evaluator.update(sample(t=0, north=0, agl=20, climb=-900))
    evaluator.update(sample(t=1, north=0, agl=airfield.PARKED_AGL_M))
    assert (evaluator.ending, evaluator.failure) == ("crashed", "hit the ground at 900 fpm")
    evaluator = cessna.CessnaEvaluator(TASKS["rings"], [], objectives.WAYPOINTS)
    evaluator.update(sample(t=0, north=0, agl=5, struck=["left wingtip"]))
    assert (evaluator.ending, evaluator.failure) == ("crashed", "left wingtip struck the ground")


def test_landing_needs_touchdown_within_limits():
    assert touch_down(name="circuit", north=-400, speed_kt=60).ending == "landed"
    fast = touch_down(name="circuit", north=-400, speed_kt=80)
    assert (fast.ending, fast.failure) == ("landing_failed", "touchdown at 80 kt exceeds 70")


def test_drone_lifts_off_vertically():
    flight_env = reset(name="drone-rings")
    north = flight_env.sample["north_m"]
    for _ in range(10):
        act(flight_env=flight_env, choices={"turn": "straight", "climb": "climb fast", "speed": "hover"})
    assert flight_env.sample["agl_m"] - airfield.PARKED_AGL_M > 15 and flight_env.sample["north_m"] == north
    assert list(flight_env.questions()) == ["turn", "climb", "speed"]  # as on the ground


def test_drone_lands_gently_on_its_spot():
    spot = drone.SPOT["north_m"]
    assert touch_down(name="drone-circuit", north=spot + 5).ending == "landed"
    beside = touch_down(name="drone-circuit", north=spot + 60)
    assert (beside.ending, beside.failure) == ("landing_failed", "touchdown 60 m from the landing spot")
    flight_env = reset(name="drone-rings")
    for _ in range(10):
        act(flight_env=flight_env, choices={"turn": "straight", "climb": "climb fast", "speed": "hover"})
    for _ in range(12):
        act(flight_env=flight_env, choices={"turn": "straight", "climb": "descend fast", "speed": "45 kt"})
    assert flight_env.evaluator.failure == "a propeller struck the ground"  # it touched down flying forward


@pytest.mark.parametrize("name", TASKS)
def test_autopilot_completes_every_objective(name):
    flight_env = reset(name)
    fly(flight_env)
    assert flight_env.evaluate().success, flight_env.evaluator.failure


@pytest.mark.parametrize(
    "name, spec, facing",
    [("course", objectives.COURSE, 0), ("gauntlet", objectives.GAUNTLET, 60)],
)
def test_courses_are_drawn_from_the_seed(name, spec, facing):
    courses = [objectives.course_rings(TASKS[name], seed) for seed in range(50)]
    assert courses[3] == objectives.course_rings(task=TASKS[name], seed=3) != courses[4]
    assert {len(rings) for rings in courses} == {spec["rings"]}
    heights = [ring["height_m"] for rings in courses for ring in rings]
    assert spec["height_m"][0] <= min(heights) and max(heights) <= spec["height_m"][1]
    # course rings face away from the ring before; gauntlet rings can be up to 60 degrees off
    off = [
        abs(airfield.wrap(ring["heading_deg"] - airfield.bearing_to(s=last, point=ring)))
        for rings in courses
        for last, ring in itertools.pairwise(rings)
    ]
    assert max(off) == pytest.approx(facing, abs=10)
    given = {"north_m": 1200.0, "east_m": 0.0, "height_m": 100.0, "heading_deg": 0.0}
    assert reset(name=name, rings=[given]).evaluator.rings == [{"name": "ring 1", **given}]
    with pytest.raises(ValueError, match="within 20 km"):
        reset(name=name, rings=[{**given, "north_m": 90000.0}])


@pytest.mark.parametrize("name", ["rings", "drone-rings"])
def test_same_seed_flies_the_same(name):
    def flown(seed: int, **settings) -> dict:
        flight_env = reset(name=name, seed=seed, wind_max_kt=0.0, **settings)
        fly(flight_env=flight_env, until=lambda: flight_env.sample["t"] >= 30)
        return flight_env.sample

    assert flown(seed=3) == flown(seed=4)  # in calm air, on the same rings, the seed changes nothing
    varied = {"airfield": "varied", "dynamics": 0.2, "gusts_kt": 8.0, "noise": 1.0}  # each drawn from the seed
    assert flown(seed=3, **varied) == flown(seed=3, **varied) != flown(seed=4, **varied)


def test_varied_airfield_turns_every_heading():
    plain, varied = reset(name="circuit", seed=2), reset(name="circuit", seed=2, airfield="varied")
    shown, turned = plain.state(), varied.state()
    assert airfield.wrap(turned["heading_deg"] - shown["heading_deg"] - varied.bearing) == pytest.approx(0, abs=1)
    wind = [float(state["wind"].split()[1]) for state in (shown, turned)]
    assert airfield.wrap(wind[1] - wind[0] - varied.bearing) == pytest.approx(0, abs=1)
    assert not varied.right_hand and all(waypoint["east_m"] <= 0 for waypoint in varied.waypoints)
    right = reset(name="drone-circuit", seed=1, airfield="varied")
    assert right.right_hand and all(waypoint["east_m"] >= 0 for waypoint in right.waypoints)
    fly(right)
    assert right.evaluate().success


@pytest.mark.parametrize("name, start", [("circuit", "downwind"), ("drone-circuit", "final")])
def test_circuit_can_start_in_the_air(name, start):
    flight_env = reset(name=name, seed=4, start=start)
    assert not flight_env.sample["on_ground"] and flight_env.sample["agl_m"] > 30
    assert flight_env.evaluator.milestones == ({"takeoff", "route"} if start == "final" else {"takeoff"})
    state, asked = flight_env.state(), flight_env.question_set()
    held = {qid: flight_env.aircraft.AIRBORNE[start][qid] for qid in asked}
    assert state["current_controls"] == {**held, "turn": f"holding heading {state['heading_deg']}"}
    assert flight_env.sample["climb_fpm"] == pytest.approx(asked["climb"][1][held["climb"]][0], abs=150)
    fly(flight_env)
    assert "landing" in flight_env.evaluator.milestones
    with pytest.raises(ValueError, match="only on the circuit"):
        reset(name=name.replace("circuit", "rings"), seed=4, start=start)


def test_noise_changes_only_the_readings():
    exact, noisy = reset(name="rings", seed=5), reset(name="rings", seed=5, noise=1.0)
    for flight_env in (exact, noisy):
        fly(flight_env=flight_env, until=lambda flight_env=flight_env: flight_env.sample["agl_m"] > 100)
    shown, read = exact.state(), noisy.state()
    assert read == noisy.state() != shown  # the same until the next tick
    assert abs(read["airspeed_kt"] - shown["airspeed_kt"]) <= 8 and abs(read["height_m"] - shown["height_m"]) <= 8
    assert read["current_controls"] == shown["current_controls"]  # its own answers are known exactly
    assert noisy.sample == exact.sample and noisy.scene() == exact.scene()
