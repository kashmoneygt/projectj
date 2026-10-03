import math
import random
from abc import ABC, abstractmethod

from jym import core
from jym.environments.flight import airfield

TAKEOFF = {"height_m": 30.0, "airspeed_kt": 50.0, "distance_m": 200.0}
APPROACH_SECONDS = 15.0
RING_RADIUS_M = 60.0
CRASH_SINK_FPM = 600.0
RINGS = [
    {"name": "climb-out", "north_m": 1100.0, "east_m": 0.0, "height_m": 90.0, "heading_deg": 0.0},
    {"name": "crosswind", "north_m": 2161.0, "east_m": -1061.0, "height_m": 160.0, "heading_deg": 315.0},
    {"name": "west", "north_m": 2161.0, "east_m": -2561.0, "height_m": 200.0, "heading_deg": 270.0},
    {"name": "descent", "north_m": 1100.0, "east_m": -3621.0, "height_m": 150.0, "heading_deg": 225.0},
]
COURSE = {
    "rings": 5,
    "distance_m": (1300.0, 2000.0),
    "turn_deg": 60.0,
    "height_m": (80.0, 300.0),
    "step_m": (-60.0, 80.0),
}
GAUNTLET = {
    "rings": 5,
    "first_m": (1500.0, 2500.0),
    "first_turn_deg": 40.0,
    "first_height_m": (80.0, 180.0),
    "first_facing_deg": 30.0,
    "distance_m": (1500.0, 3000.0),
    "turn_deg": 120.0,
    "step_m": 150.0,
    "height_m": (60.0, 450.0),
    "facing_deg": 60.0,
}
WAYPOINTS = [
    {"name": "upwind", "north_m": 1500.0, "east_m": 0.0, "radius_m": 300.0, "height_m": (60.0, 400.0)},
    {
        "name": "crosswind",
        "north_m": 1800.0,
        "east_m": -1400.0,
        "radius_m": 300.0,
        "height_m": (150.0, 400.0),
    },
    {
        "name": "downwind",
        "north_m": -1500.0,
        "east_m": -1400.0,
        "radius_m": 300.0,
        "height_m": (180.0, 350.0),
    },
    {"name": "base", "north_m": -2400.0, "east_m": -700.0, "radius_m": 300.0, "height_m": (100.0, 320.0)},
]

VERSION = "6"
# goals describe the aircraft, so one policy can fly any of them
BRIEFS = {
    "cessna": {
        "title": "Cessna 172",
        "brief": "The aircraft is a Cessna 172 airplane: it lifts off at about 55 kt after a takeoff roll, stalls "
        "below about 48 kt and lands on a runway at 55-80 kt.",
        "takeoff": f" at {TAKEOFF['airspeed_kt']:g} kt or more",
        "landing": "make a stable approach and land on the same runway, stopping on it",
    },
    "drone": {
        "title": "Quadcopter drone",
        "brief": "The aircraft is a quadcopter drone: it takes off and lands vertically, hovers in place and flies "
        "forward at up to 45 kt.",
        "takeoff": "",
        "landing": (
            f"make a stable approach to the landing spot on the runway centreline, "
            f"{airfield.AIM_PAST_THRESHOLD_M:g} m past its threshold, and land on it"
        ),
    },
}


OBJECTIVES = {  # title, milestones, max decisions, max seconds
    "takeoff": ("Take off", {"takeoff": 100}, 300, 600),
    "rings": ("Fly through rings", {"takeoff": 20, "ring_1": 20, "ring_2": 20, "ring_3": 20, "ring_4": 20}, 1500, 3600),
    "course": (
        "Random ring course",
        {"takeoff": 20, **{f"ring_{n}": 16 for n in range(1, COURSE["rings"] + 1)}},
        1800,
        3600,
    ),
    "gauntlet": (
        "Ring gauntlet",
        {"takeoff": 20, **{f"ring_{n}": 16 for n in range(1, GAUNTLET["rings"] + 1)}},
        1800,
        3600,
    ),
    "circuit": ("Circuit and landing", {"takeoff": 20, "route": 30, "approach": 20, "landing": 30}, 1500, 3600),
}


def objectives(aircraft: str) -> list[core.Task]:
    """Return this aircraft's objectives; only the Cessna's ids have no aircraft prefix"""
    about = BRIEFS[aircraft]
    rings = f"Each ring is a vertical circle of {RING_RADIUS_M:g} m radius, passed by flying through it in its heading."
    goals = {
        "takeoff": f"Take off from the runway and climb to {TAKEOFF['height_m']:g} m above it{about['takeoff']}.",
        "rings": f"Take off, then fly through {len(RINGS)} rings in order. {rings} No landing needed.",
        "course": f"Take off, then fly through {COURSE['rings']} rings in order. {rings} The course changes with the "
        "seed. No landing needed.",
        "gauntlet": f"Take off, then fly through {GAUNTLET['rings']} rings in order. {rings} They change with the "
        "seed: anywhere around the airfield, at very different heights, each facing its own way, so each needs a turn "
        "to line up with it. No landing needed.",
        "circuit": f"Take off, fly the four circuit waypoints in order, {about['landing']}.",
    }
    return [
        core.Task(
            env="flight",
            id=name if aircraft == "cessna" else f"{aircraft}-{name}",
            title=title,
            version=VERSION,
            goal=f"{goals[name]} {about['brief']}",
            milestones=milestones,
            max_steps=steps,
            max_seconds=seconds,
            settings={"aircraft": aircraft},
        )
        for name, (title, milestones, steps, seconds) in OBJECTIVES.items()
    ]


TASKS = [task for aircraft in BRIEFS for task in objectives(aircraft)]


def aircraft_of(task: core.Task) -> str:
    return task.settings["aircraft"]


def objective_of(task: core.Task) -> str:
    return task.id.removeprefix(f"{aircraft_of(task)}-")


def course_rings(task: core.Task, seed: int) -> list[dict]:
    """Return an objective's rings from its settings, the fixed set, or the seed"""
    if task.settings.get("rings"):
        return validate_rings(task.settings["rings"])
    if objective_of(task) == "rings":
        return [dict(ring) for ring in RINGS]
    if objective_of(task) == "gauntlet":
        return gauntlet_rings(seed)
    if objective_of(task) != "course":
        return []
    rng = random.Random(f"course-{seed}")
    rings = [{**RINGS[0], "name": "ring 1"}]
    for number in range(2, COURSE["rings"] + 1):
        last = rings[-1]
        heading = (last["heading_deg"] + rng.uniform(-COURSE["turn_deg"], COURSE["turn_deg"])) % 360
        metres = rng.uniform(*COURSE["distance_m"])
        height = airfield.clamp(last["height_m"] + rng.uniform(*COURSE["step_m"]), *COURSE["height_m"])
        rings.append(
            {
                "name": f"ring {number}",
                "north_m": round(last["north_m"] + metres * math.cos(math.radians(heading)), 1),
                "east_m": round(last["east_m"] + metres * math.sin(math.radians(heading)), 1),
                "height_m": round(height),
                "heading_deg": round(heading, 1),
            }
        )
    return rings


def gauntlet_rings(seed: int) -> list[dict]:
    """Place each ring relative to the last, each facing its own way"""
    spec, rng = GAUNTLET, random.Random(f"gauntlet-{seed}")
    last = {
        "north_m": airfield.THRESHOLD_NORTH_M + airfield.RUNWAY["length_m"],
        "east_m": 0.0,
        "heading_deg": airfield.RUNWAY["heading_deg"],
    }
    low, high = spec["height_m"]
    rings = []
    for number in range(1, spec["rings"] + 1):
        first = number == 1
        turn = rng.uniform(-1, 1) * spec["first_turn_deg" if first else "turn_deg"]
        metres = rng.uniform(*spec["first_m" if first else "distance_m"])
        height = (
            rng.uniform(*spec["first_height_m"]) if first else last["height_m"] + rng.uniform(-1, 1) * spec["step_m"]
        )
        height = 2 * low - height if height < low else 2 * high - height if height > high else height  # bounce back
        facing = rng.uniform(-1, 1) * spec["first_facing_deg" if first else "facing_deg"]
        if facing * turn < 0:  # keep S-turns no tighter than a gentle turn
            facing *= min(1.0, (180 - abs(turn)) / 3 / spec["facing_deg"])
        bearing = last["heading_deg"] + turn
        last = {
            "name": f"ring {number}",
            "north_m": round(last["north_m"] + metres * math.cos(math.radians(bearing)), 1),
            "east_m": round(last["east_m"] + metres * math.sin(math.radians(bearing)), 1),
            "height_m": round(height),
            "heading_deg": round((bearing + facing) % 360, 1),
        }
        rings.append(last)
    return rings


def validate_rings(rings: list) -> list[dict]:
    if not isinstance(rings, list) or not 1 <= len(rings) <= 8:
        raise ValueError("A course needs 1-8 rings")
    checked = []
    for number, ring in enumerate(rings, 1):
        values = {key: float(ring[key]) for key in ("north_m", "east_m", "height_m", "heading_deg")}
        if not (math.hypot(values["north_m"], values["east_m"]) <= 20000 and 40 <= values["height_m"] <= 1000):
            raise ValueError(f"Ring {number} must be within 20 km of the runway and 40-1000 m up")
        checked.append({"name": str(ring.get("name") or f"ring {number}")[:40], **values})
    return checked


def ring_course(s: dict, ring: dict) -> float:
    """Aim halfway along the ring's axis to line up with it, and come around after a miss"""
    heading = math.radians(ring["heading_deg"])
    north, east = s["north_m"] - ring["north_m"], s["east_m"] - ring["east_m"]
    along = north * math.cos(heading) + east * math.sin(heading)
    back = 1500.0 if along > 0 else -along / 2
    return airfield.bearing_to(
        s=s,
        point={
            "north_m": ring["north_m"] - back * math.cos(heading),
            "east_m": ring["east_m"] - back * math.sin(heading),
        },
    )


class Evaluator(ABC):
    """Score an objective from every physics tick. Only hitting the ground counts as a crash"""

    def __init__(self, task: core.Task, rings: list[dict], waypoints: list[dict]):
        self.task = task
        self.rings = rings
        self.waypoints = waypoints
        self.ring = 0
        self.waypoint = 0
        self.milestones: set[str] = set()
        self.ending: str | None = None
        self.failure: str | None = None
        self.impact: dict | None = None
        self.start: dict | None = None
        self.previous: dict | None = None
        self.lifted_off = False
        self.route_done_at: float | None = None
        self.stable_seconds = 0.0
        self.touchdowns: list[dict] = []
        self.passes: list[dict] = []
        self.times: dict[str, float] = {}
        self.max_bank_deg = 0.0
        self.max_height_m = 0.0

    def place_in_air(self, waypoint: int, s: dict) -> None:
        """Start airborne, already past takeoff and heading for this waypoint"""
        self.milestones.add("takeoff")
        self.lifted_off, self.waypoint, self.start = True, waypoint, s
        if waypoint >= len(self.waypoints):
            self.milestones.add("route")
            self.route_done_at = s["t"]

    @property
    def phase(self) -> str:
        if self.ending:
            return self.ending
        if "takeoff" not in self.milestones:
            return "takeoff"
        if self.rings:
            return f"ring {self.ring + 1}"
        if "route" not in self.milestones:
            return f"{self.waypoints[self.waypoint]['name']} waypoint"
        return "rollout" if self.previous and self.previous["on_ground"] else "final"

    def update(self, s: dict) -> None:
        if self.ending:
            return
        previous, self.previous = self.previous, s
        self.start = self.start or s
        along, right = airfield.runway_frame(s["north_m"], s["east_m"])
        height = s["agl_m"] - airfield.PARKED_AGL_M
        self.max_bank_deg = max(self.max_bank_deg, abs(s["bank_deg"]))
        self.max_height_m = max(self.max_height_m, height)
        if s["struck"]:
            return self.crash(reason=f"{s['struck'][0]} struck the ground", s=s)
        if previous and s["on_ground"] and not previous["on_ground"]:
            touchdown = {
                "t": s["t"],
                "sink_fpm": -previous["climb_fpm"],
                "airspeed_kt": previous["airspeed_kt"],
                "bank_deg": previous["bank_deg"],
                "heading_error_deg": airfield.wrap(previous["heading_deg"] - airfield.RUNWAY["heading_deg"]),
                "along_m": along,
                "right_m": right,
                "nose_first": s["nose_gear_first"],
            }
            self.touchdowns.append(touchdown)
            if touchdown["sink_fpm"] > CRASH_SINK_FPM:
                return self.crash(reason=f"hit the ground at {touchdown['sink_fpm']:.0f} fpm", s=s)
        if previous and previous["on_ground"] and not s["on_ground"]:
            self.lifted_off = True

        if "takeoff" not in self.milestones:
            if self.took_off(s, height):
                self.award(milestone="takeoff", s=s)
                if objective_of(self.task) == "takeoff":
                    self.ending = "airborne"
        elif self.rings:
            self.fly_through_ring(previous, s)
        elif "route" not in self.milestones:
            waypoint = self.waypoints[self.waypoint]
            low, high = waypoint["height_m"]
            near = (
                math.hypot(waypoint["north_m"] - s["north_m"], waypoint["east_m"] - s["east_m"]) <= waypoint["radius_m"]
            )
            if not s["on_ground"] and near and low <= height <= high:
                self.times[waypoint["name"]] = round(s["t"], 2)
                self.waypoint += 1
                if self.waypoint == len(self.waypoints):
                    self.route_done_at = s["t"]
                    self.award(milestone="route", s=s)
        else:
            if "approach" not in self.milestones:
                held = s["t"] - previous["t"] if previous else 0.0  # a flight can start already on the approach
                self.stable_seconds = self.stable_seconds + held if self.stable(s, along, right) else 0.0
                if self.stable_seconds >= APPROACH_SECONDS - 1e-9:
                    self.award(milestone="approach", s=s)
            if s["on_ground"] and s["ground_speed_kt"] < 1.0:
                self.land(s)

    @abstractmethod
    def took_off(self, s: dict, height: float) -> bool: ...

    def fly_through_ring(self, previous: dict, s: dict) -> None:
        """Pass a ring when the path crosses its plane in its direction, inside its circle"""
        ring = self.rings[self.ring]
        (before, right_before), (after, right_after) = (
            airfield.frame(point["north_m"] - ring["north_m"], point["east_m"] - ring["east_m"], ring["heading_deg"])
            for point in (previous, s)
        )
        if not before < 0 <= after:
            return
        fraction = before / (before - after)
        right = right_before + fraction * (right_after - right_before)
        above = (
            previous["agl_m"] + fraction * (s["agl_m"] - previous["agl_m"]) - airfield.PARKED_AGL_M - ring["height_m"]
        )
        if math.hypot(right, above) <= RING_RADIUS_M:
            self.passes.append(
                {
                    "ring": ring["name"],
                    "t": round(s["t"], 2),
                    "right_m": round(right, 1),
                    "above_m": round(above, 1),
                }
            )
            self.ring += 1
            self.award(milestone=f"ring_{self.ring}", s=s)
            if self.ring == len(self.rings):
                self.ending = "rings_complete"

    @abstractmethod
    def stable(self, s: dict, along: float, right: float) -> bool: ...

    def land(self, s: dict) -> None:
        touchdowns = [t for t in self.touchdowns if t["t"] >= self.route_done_at]
        problems = [problem for t in touchdowns for problem in self.problems(t)]
        if not touchdowns:
            problems.append("no touchdown after the circuit")
        if problems:
            self.ending, self.failure = "landing_failed", "; ".join(problems)
        else:
            self.award(milestone="landing", s=s)
            self.ending = "landed"

    @abstractmethod
    def problems(self, t: dict) -> list[str]: ...

    @abstractmethod
    def on_landing_area(self, along: float, right: float) -> bool: ...

    def award(self, milestone: str, s: dict) -> None:
        self.milestones.add(milestone)
        self.times[milestone] = round(s["t"], 2)

    def crash(self, reason: str, s: dict) -> None:
        self.ending, self.failure = "crashed", reason
        self.impact = {
            "sim_time": s["t"],
            "position": {key: s[key] for key in ("north_m", "east_m", "altitude_m")},
            "airspeed_kt": s["airspeed_kt"],
            "reason": reason,
        }

    def evaluation(self, metrics: dict) -> core.Evaluation:
        report = {
            "phase": self.phase,
            "times": self.times,
            "touchdowns": self.touchdowns,
            "rings_passed": self.passes,
            "waypoints_reached": self.waypoint,
            "max_bank_deg": round(self.max_bank_deg, 1),
            "max_height_m": round(self.max_height_m, 1),
            **metrics,
        }
        return core.Evaluation(
            success=self.milestones == set(self.task.milestones),
            milestones=set(self.milestones),
            failure=self.failure,
            metrics=report,
        )
