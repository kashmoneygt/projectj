import math
import random
from typing import ClassVar

from jym import core
from jym.environments.flight import airfield, cessna, controls, drone, objectives

WIND_MAX_KT = 8.0
# settings only training uses
VARIETY = {"airfield": ("fixed", "varied"), "start": ("runway", "downwind", "final")}
# reading noise (standard deviation at noise 1); scoring and the scripted pilot use true values
READING_NOISE = {
    "north_m": 3.0,
    "east_m": 3.0,
    "agl_m": 2.0,
    "airspeed_kt": 2.0,
    "ground_speed_kt": 1.0,
    "climb_fpm": 60.0,
    "pitch_deg": 1.0,
    "bank_deg": 1.5,
    "heading_deg": 2.0,
    "wind_from_deg": 10.0,
    "wind_speed_kt": 1.5,
}

STATE = {
    "goal": "the objective, in words",
    "situation": "the phase in words: stopped on the runway, rolling down the runway, airborne and climbing out, "
    "flying to ring k of n, flying to a waypoint, on final approach, rolling out on the runway, landed, crashed",
    "airspeed_kt": "indicated airspeed, knots",
    "ground_speed_kt": "speed over the ground, knots",
    "height_m": "height above the runway, metres",
    "vertical_speed_fpm": "climb (positive) or descent (negative), feet per minute",
    "pitch": "the nose's attitude, e.g. 'nose 6 deg up'",
    "bank": "'wings level', or e.g. '15 deg left'",
    "heading_deg": "compass heading, degrees",
    "stall_warning": "whether the stall warning sounds",
    "current_controls": "the answer to each question now in force; for the turn, the heading being held",
    "target": "the next point to fly to, relative to the aircraft. On the runway (the Cessna): runway_left_m, "
    "centreline and direction. Flying: the ring or waypoint (k of n), distance_m, direction (which way to turn to "
    "follow the course: for a ring, its approach line, so as to fly through it in its heading), its height or height "
    "band, height_difference, and climb_needed_fpm to arrive at its height. On final approach, the Cessna: "
    "runway_threshold_ahead_m, centreline, direction, glide_path and approach_speed_kt; the drone: the landing_spot, "
    "distance_m, direction, height_difference and climb_needed_fpm to touch down on it",
    "wind": "the direction it blows from and its speed",
}
AIRCRAFT = {"cessna": cessna.Cessna, "drone": drone.Drone}


class FlightEnvironment:
    """The flight environment: a Cessna or a drone flying objectives at an airfield"""

    version = objectives.VERSION
    RELEASED: ClassVar[dict] = {"elevator": 0.0, "aileron": 0.0, "pedals": 0.0}

    def reset(self, task: core.Task, seed: int) -> None:
        self.task, self.seed, self.objective = task, seed, objectives.objective_of(task)
        rng = random.Random(seed)
        strongest = float(task.settings.get("wind_max_kt", WIND_MAX_KT))
        if not 0 <= strongest <= 25:
            raise ValueError("wind_max_kt must be 0-25")
        self.wind = {
            "from_deg": round(rng.uniform(0, 360)) % 360,
            "speed_kt": round(rng.uniform(0, strongest), 1),
        }
        self.lay_out(task, seed)
        dynamics, gusts = float(task.settings.get("dynamics", 0.0)), float(task.settings.get("gusts_kt", 0.0))
        self.noise = float(task.settings.get("noise", 0.0))
        if not (0 <= dynamics <= 0.3 and 0 <= gusts <= 15 and 0 <= self.noise <= 2):
            raise ValueError("dynamics must be 0-0.3, gusts_kt 0-15 and noise 0-2")
        self.aircraft = AIRCRAFT[objectives.aircraft_of(task)](self.wind, dynamics, gusts, seed)
        self.chosen = dict(self.aircraft.START)
        self.evaluator = self.aircraft.evaluator(
            task, self.mirrored(objectives.course_rings(task, seed)), self.waypoints
        )
        start = task.settings.get("start", "runway")
        if start not in VARIETY["start"] or (start != "runway" and self.objective != "circuit"):
            raise ValueError(f"start must be one of {VARIETY['start']}, and in the air only on the circuit")
        if start != "runway":
            self.start_in_air(start, seed)
        self.sample = self.aircraft.sample
        self.evaluator.update(self.sample)

    def lay_out(self, task: core.Task, seed: int) -> None:
        """Lay out the airfield, rotated, mirrored and stretched when varied"""
        airfield_setting = task.settings.get("airfield", "fixed")
        if airfield_setting not in VARIETY["airfield"]:
            raise ValueError(f"airfield must be one of {VARIETY['airfield']}")
        varied = airfield_setting == "varied"
        rng = random.Random(f"airfield-{seed}")
        self.bearing = rng.uniform(0, 360) if varied else 0.0  # added to every heading shown
        self.right_hand = varied and rng.random() < 0.5
        north, east = (rng.uniform(0.85, 1.15), rng.uniform(0.8, 1.25)) if varied else (1.0, 1.0)
        side = -1.0 if self.right_hand else 1.0
        self.waypoints = [
            {**w, "north_m": w["north_m"] * north, "east_m": w["east_m"] * east * side} for w in objectives.WAYPOINTS
        ]

    def start_in_air(self, start: str, seed: int) -> None:
        """Start the circuit in the air on downwind or final, with the earlier legs counted as flown"""
        rng = random.Random(f"start-{seed}")
        if start == "final":
            before = rng.uniform(150.0, 3000.0)  # metres before the threshold
            north, east = airfield.THRESHOLD_NORTH_M - before, rng.uniform(-1.0, 1.0) * (10.0 + 0.05 * before)
            slope = math.tan(math.radians(airfield.GLIDE_SLOPE_DEG))
            height = slope * (before + airfield.AIM_PAST_THRESHOLD_M) * rng.uniform(0.8, 1.2)
            off = rng.uniform(-1.0, 1.0) * (2.0 + 0.003 * before)
            heading, done = (airfield.RUNWAY["heading_deg"] + off) % 360, len(self.waypoints)
        else:
            crosswind, downwind = self.waypoints[1], self.waypoints[2]
            north = downwind["north_m"] + rng.uniform(0.35, 0.8) * (crosswind["north_m"] - downwind["north_m"])
            east, height = downwind["east_m"], rng.uniform(200.0, 300.0)
            heading, done = (airfield.RUNWAY["heading_deg"] + 180.0) % 360, 2
        answers = self.aircraft.AIRBORNE[start]
        self.aircraft.fly_from(north, east, height, heading, answers)
        self.evaluator.place_in_air(waypoint=done, s=self.aircraft.sample)
        self.chosen |= answers
        self.apply({name: self.chosen[name] for name in self.question_set()})  # so current_controls shows them

    def mirrored(self, rings: list[dict]) -> list[dict]:
        if not self.right_hand:
            return rings
        return [{**ring, "east_m": -ring["east_m"], "heading_deg": (360 - ring["heading_deg"]) % 360} for ring in rings]

    @property
    def HZ(self) -> int:
        return self.aircraft.HZ

    @property
    def ticks(self) -> int:
        return self.aircraft.ticks

    def reading(self) -> dict:
        """Return the sample and wind as the policy sees them, with noise seeded by tick"""
        s = {**self.sample, "wind_from_deg": self.wind["from_deg"], "wind_speed_kt": self.wind["speed_kt"]}
        if self.noise:
            rng = random.Random(f"noise-{self.seed}-{self.ticks}")
            for key, spread in READING_NOISE.items():
                s[key] += rng.gauss(0.0, spread * self.noise)
            s["wind_speed_kt"] = max(0.0, s["wind_speed_kt"])
        return s

    def state(self) -> dict:
        s = self.reading()
        return {
            "situation": self.situation(),
            "airspeed_kt": round(s["airspeed_kt"]),
            "ground_speed_kt": round(s["ground_speed_kt"]),
            "height_m": max(0, round(s["agl_m"] - airfield.PARKED_AGL_M)),
            "vertical_speed_fpm": int(round(s["climb_fpm"], -1)),
            "pitch": f"nose {abs(s['pitch_deg']):.0f} deg {'up' if s['pitch_deg'] >= 0 else 'down'}",
            "bank": "wings level"
            if abs(s["bank_deg"]) < 3
            else f"{abs(s['bank_deg']):.0f} deg {airfield.side_of(s['bank_deg'])}",
            "heading_deg": round(s["heading_deg"] + self.bearing) % 360,
            "stall_warning": self.aircraft.stall_warning(),
            "current_controls": self.current_controls(),
            "target": self.target(s),
            "wind": f"from {round(s['wind_from_deg'] + self.bearing) % 360} deg at {round(s['wind_speed_kt'], 1):g} kt",
        }

    def current_controls(self) -> dict:
        shown = {name: self.chosen[name] for name in self.question_set()}
        if "turn" in shown and (held := self.aircraft.heading_held()) is not None:
            shown["turn"] = f"holding heading {round(held + self.bearing) % 360}"
        return shown

    def question_set(self) -> dict:
        return self.aircraft.questions(self.objective, "route" in self.evaluator.milestones)

    def questions(self) -> dict:
        return controls.as_questions(self.question_set())

    def apply(self, choices: dict[str, str]) -> None:
        asked = self.question_set()
        if set(choices) != set(asked):
            raise ValueError(f"These choices answer {sorted(choices)}, but the aircraft now asks {sorted(asked)}")
        self.chosen |= choices
        self.aircraft.apply({name: asked[name][1][option][0] for name, option in choices.items()})

    def hold(self) -> None:
        self.aircraft.hold()

    def control(self, values: dict[str, float]) -> None:
        self.aircraft.control(values)

    def advance(self, seconds: float) -> bool:
        for _ in range(max(1, round(seconds * self.aircraft.HZ))):
            if self.evaluator.ending:
                break
            self.sample = self.aircraft.tick()
            self.evaluator.update(self.sample)
        return self.evaluator.ending is not None

    def instruments(self) -> dict:
        return self.aircraft.instruments()

    def situation(self) -> str:
        evaluator, s = self.evaluator, self.sample
        if evaluator.phase == "takeoff":
            if not s["on_ground"]:
                return "airborne, climbing out"
            return "stopped on the runway" if s["ground_speed_kt"] < 3 else "rolling down the runway"
        if evaluator.phase.startswith("ring "):  # not rings_complete
            return f"flying to ring {evaluator.ring + 1} of {len(evaluator.rings)}"
        endings = {
            "crashed": f"crashed: {evaluator.failure}",
            "airborne": "takeoff complete",
            "rings_complete": "all rings passed",
            "landed": "landed",
            "landing_failed": "landing failed",
            "final": "on final approach",
            "rollout": "rolling out on the runway",
        }
        return endings.get(evaluator.phase, f"flying to the {evaluator.phase}")

    def target(self, s: dict) -> dict | None:
        """Return the next point to fly to, and which way to turn for it"""
        evaluator = self.evaluator
        along, right = airfield.runway_frame(s["north_m"], s["east_m"])
        height = s["agl_m"] - airfield.PARKED_AGL_M
        direction = airfield.turn_text(airfield.wrap(self.course(s) - s["heading_deg"]))
        if s["on_ground"] and evaluator.phase in ("takeoff", "rollout") and self.aircraft.ROLLS:
            return {
                "runway_left_m": round(airfield.RUNWAY["length_m"] / 2 - along),
                "centreline": airfield.offset_text(right),
                "direction": direction,
            }
        if evaluator.rings and evaluator.ring < len(evaluator.rings):
            ring = evaluator.rings[evaluator.ring]
            return {
                "ring": f"{evaluator.ring + 1} of {len(evaluator.rings)}",
                "distance_m": airfield.distance(s=s, point=ring),
                "direction": direction,
                "height_m": round(ring["height_m"]),
                "height_difference": airfield.vertical_text(ring["height_m"] - height),
                "climb_needed_fpm": self.climb_needed(s=s, point=ring, metres=ring["height_m"] - height),
            }
        if self.objective == "circuit" and "route" not in evaluator.milestones:
            waypoint = self.waypoints[evaluator.waypoint]
            return {
                "waypoint": f"{waypoint['name']} ({evaluator.waypoint + 1} of {len(self.waypoints)})",
                "distance_m": airfield.distance(s, waypoint),
                "direction": direction,
                "height_band_m": [round(metres) for metres in waypoint["height_m"]],
                "climb_needed_fpm": self.climb_needed(
                    s=s, point=waypoint, metres=sum(waypoint["height_m"]) / 2 - height
                ),
            }
        if evaluator.phase == "takeoff":
            return {"climb_to_m": round(objectives.TAKEOFF["height_m"]), "direction": direction}
        if evaluator.phase == "final":
            return self.aircraft.final_target(s, direction)
        return None

    def climb_needed(self, s: dict, point: dict, metres: float) -> int:
        """Return the climb rate that reaches the target's height on arrival"""
        seconds = airfield.distance(s, point) / max(s["ground_speed_kt"] * airfield.KT, 20.0)
        return int(round(metres / max(seconds, 5.0) * airfield.FPM, -1))

    def course(self, s: dict | None = None) -> float:
        s, evaluator = self.sample if s is None else s, self.evaluator
        if evaluator.rings:
            return objectives.ring_course(s, evaluator.rings[min(evaluator.ring, len(evaluator.rings) - 1)])
        if self.objective == "circuit" and "route" not in evaluator.milestones:
            return airfield.bearing_to(s, self.waypoints[evaluator.waypoint])
        if evaluator.phase == "takeoff":
            return airfield.RUNWAY["heading_deg"]
        return self.aircraft.final_course(s)

    def navigation(self, height: float) -> tuple[float, float]:
        """Return the scripted pilot's course and climb rate"""
        evaluator, course = self.evaluator, self.course()
        if evaluator.phase == "takeoff":
            return course, 1000.0
        if evaluator.rings:
            return course, 15 * (evaluator.rings[min(evaluator.ring, len(evaluator.rings) - 1)]["height_m"] - height)
        return course, 15 * (sum(self.waypoints[evaluator.waypoint]["height_m"]) / 2 - height)

    def evaluate(self) -> core.Evaluation:
        return self.evaluator.evaluation(
            {"sim_seconds": round(self.sample["t"], 2), "seed": self.seed, "wind": self.wind, **self.aircraft.metrics()}
        )

    def scene(self) -> dict:
        s, evaluator = self.sample, self.evaluator
        rings = [
            {
                **ring,
                "radius_m": objectives.RING_RADIUS_M,
                "altitude_m": airfield.RUNWAY["elevation_m"] + ring["height_m"],
                "passed": n < evaluator.ring,
            }
            for n, ring in enumerate(evaluator.rings)
        ]
        waypoints = [{key: w[key] for key in ("name", "north_m", "east_m", "radius_m")} for w in self.waypoints]
        return {
            "position": {key: s[key] for key in ("north_m", "east_m", "altitude_m", "agl_m")},
            "attitude": {
                "roll_deg": s["bank_deg"],
                "pitch_deg": s["pitch_deg"],
                "heading_deg": s["heading_deg"],
            },
            "airspeed_kt": s["airspeed_kt"],
            "ground_speed_kt": s["ground_speed_kt"],
            "vertical_speed_fpm": s["climb_fpm"],
            "aircraft": objectives.aircraft_of(self.task),
            **self.aircraft.scene(),
            "weather": {"wind_direction_deg": self.wind["from_deg"], "wind_speed_kt": self.wind["speed_kt"]},
            "runway": airfield.RUNWAY,
            "waypoints": rings or (waypoints if self.objective == "circuit" else []),
            "goal": {"kind": "rings", "next": evaluator.ring, "rings": rings} if rings else None,
            "next_waypoint": evaluator.waypoint if self.objective == "circuit" else None,
            "target": self.target(self.sample),
            "sim_time": round(s["t"], 4),
            "status": {
                "phase": self.situation(),
                "on_ground": s["on_ground"],
                "crashed": evaluator.ending == "crashed",
            },
            "impact": evaluator.impact,
        }

    def reference(self) -> "ScriptedPilot":
        return ScriptedPilot(self)

    def close(self) -> None:
        self.aircraft.close()


class ScriptedPilot:
    """Hand-written flying rules that read the simulator directly"""

    name = "reference"
    synchronous = True  # reads the simulator, so it's asked between physics ticks

    def __init__(self, environment: FlightEnvironment):
        self.environment = environment

    def decide(self, state: dict, questions: dict) -> core.Reply:
        answers = self.environment.aircraft.scripted_answers(self.environment)
        return core.picked(**{name: answers[name] for name in questions})
