from jym import core, runner
from jym.environments.flight import cessna, controls, drone, environment, objectives


def describe() -> dict:
    return {
        "state": environment.STATE,
        "phases": {
            "Cessna 172, on the ground": controls.as_questions(cessna.ON_GROUND),
            "Cessna 172, in the air (flaps on the circuit only)": controls.as_questions(cessna.IN_AIR),
            "Cessna 172, after landing on the circuit": controls.as_questions(cessna.ROLLOUT),
            "Quadcopter drone, on the ground and in the air": controls.as_questions(drone.QUESTIONS),
        },
        "timing": f"Heats run in real time: an answer applies when it arrives and the one before holds meanwhile; "
        f"after {runner.WATCHDOG_SECONDS:g} s without one the wings are levelled. In lockstep runs the simulation "
        f"waits for each decision, then flies {runner.DECISION_SECONDS:g} s. The aircraft's controller holds each "
        f"answer until the next: a heading, a climb rate and a speed in the air; on the ground the Cessna's brakes, "
        f"power and lift-off work directly while the nosewheel holds the heading.",
    }


def question_sets(task: str) -> list[dict]:
    spec = next(t for t in objectives.TASKS if t.id == task)
    return [
        controls.as_questions(questions)
        for questions in environment.AIRCRAFT[objectives.aircraft_of(spec)].question_sets(objectives.objective_of(spec))
    ]


def pose(scene: dict) -> list[float]:
    p, a = scene["position"], scene["attitude"]
    return [
        round(float(scene["sim_time"]), 2),
        *(round(p[key], 1) for key in ("north_m", "east_m", "altitude_m")),
        *(round(a[key], 1) for key in ("roll_deg", "pitch_deg", "heading_deg")),
    ]


SPEC = core.EnvironmentSpec(
    id="flight",
    title="Flight Simulator",
    about="A Cessna 172 on the JSBSim flight dynamics engine, or a quadcopter drone on a simple multirotor model",
    environment=environment.FlightEnvironment,
    tasks=objectives.TASKS,
    objective="rings",
    describe=describe,
    question_sets=question_sets,
    reference={
        "title": "Autopilot",
        "role": "rule-based",
        "trained": "Hand-written flying rules (not AI).",
        "deployed": "Runs inside the simulator and reads its exact values instead of the text models get. "
        "Answers instantly.",
    },
    pose=pose,
    catalog={"aircraft": [{"id": name, "title": about["title"]} for name, about in objectives.BRIEFS.items()]},
)
