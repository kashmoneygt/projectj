import math
import random
from typing import TYPE_CHECKING, ClassVar

from jym import core
from jym.environments.flight import airfield, controls, objectives

if TYPE_CHECKING:
    from jym.environments.flight.environment import FlightEnvironment

HZ = 60
G = 9.80665
# response times and limits of a heavy-lift quadcopter
CLIMB_SECONDS, SPEED_SECONDS, MAX_ACCEL = 0.7, 0.5, 4.0
YAW_GAIN, MAX_YAW_DPS = 2.0, 60.0
# gusts change over GUST_SECONDS; the controller corrects within CATCH_SECONDS
GUST_SECONDS, CATCH_SECONDS = 4.0, 1.5
DRAG = 0.008  # display tilt per (m/s)^2 of airspeed
TIP_KT = 10.0  # touching down faster than this tips it over
SPOT = {"north_m": airfield.THRESHOLD_NORTH_M + airfield.AIM_PAST_THRESHOLD_M, "east_m": 0.0}
# the approach gate must hold for 15 s
APPROACH = {"distance_m": 500.0, "height_m": (3.0, 150.0), "ground_speed_kt": 25.0, "climb_fpm": (-1000.0, 100.0)}
LANDING = {"radius_m": 20.0, "sink_fpm": 400.0}

SPEED = (
    "Which ground speed should the drone hold, flying forward?",
    {
        "hover": (0.0, "stop and hold its position"),
        "15 kt": (15.0, "slow, to approach a spot"),
        "45 kt": (45.0, "full speed"),
    },
)
# same questions on the ground and in the air; climbing lifts off
QUESTIONS = {
    "turn": controls.TURN,
    "climb": (controls.CLIMB[0], {**controls.CLIMB[1], "flare": (-100.0, "touch down gently at about -100 ft/min")}),
    "speed": SPEED,
}


class Drone:
    """A simple quadcopter with a position-mode flight controller, at 60 Hz"""

    HZ = HZ
    ROLLS = False  # takes off and lands vertically
    START: ClassVar[dict] = {"turn": "straight", "climb": "level", "speed": "hover"}
    # at 45 kt, "descend slowly" follows the glide path to the spot
    AIRBORNE: ClassVar[dict] = {
        "downwind": {"turn": "straight", "climb": "level", "speed": "45 kt"},
        "final": {"turn": "straight", "climb": "descend slowly", "speed": "45 kt"},
    }
    TOP_SPEED = SPEED[1]["45 kt"][0] * airfield.KT

    def __init__(self, wind: dict, dynamics: float, gusts: float, seed: int):
        rng = random.Random(f"dynamics-{seed}")

        def vary() -> float:
            return rng.uniform(1 - dynamics, 1 + dynamics)

        self.climb_seconds, self.speed_seconds = CLIMB_SECONDS * vary(), SPEED_SECONDS * vary()
        self.max_accel, self.max_yaw = MAX_ACCEL * vary(), MAX_YAW_DPS * vary()
        self.gusts = gusts * airfield.KT / 2  # standard deviation in m/s (gusts_kt is about the peak)
        self.gust_rng = random.Random(f"gusts-{seed}")
        self.gust, self.caught, self.push = [0.0, 0.0], [0.0, 0.0], (0.0, 0.0)  # north and east, m/s
        blowing = math.radians(wind["from_deg"])
        self.air = (
            -wind["speed_kt"] * airfield.KT * math.cos(blowing),
            -wind["speed_kt"] * airfield.KT * math.sin(blowing),
        )
        self.north, self.east, self.agl = airfield.START_NORTH_M, 0.0, airfield.PARKED_AGL_M
        self.heading, self.speed, self.climb, self.yaw_rate, self.accel = (
            airfield.RUNWAY["heading_deg"],
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.heading_target, self.climb_target, self.speed_target = self.heading, 0.0, 0.0
        self.sticks: dict[str, float] | None = None  # a person's sticks while they fly
        self.struck: list[str] = []
        self.ticks = 0
        self.sample = self.measure()

    def evaluator(self, task: core.Task, rings: list[dict], waypoints: list[dict]) -> "DroneEvaluator":
        return DroneEvaluator(task, rings, waypoints)

    def scripted_answers(self, environment: "FlightEnvironment") -> dict[str, str]:
        s, evaluator = self.instruments(), environment.evaluator
        height = s["agl_m"] - airfield.PARKED_AGL_M
        if evaluator.phase in ("final", "rollout"):
            course, climb, speed = landing(s, height)
        else:
            course, climb = environment.navigation(height)
            speed = "45 kt" if environment.objective != "takeoff" and height >= 10 else "hover"
        return {
            "turn": controls.nearest(question=controls.TURN, value=airfield.wrap(course - self.heading_target)),
            "climb": controls.nearest(question=QUESTIONS["climb"], value=airfield.clamp(climb, -1000, 1000)),
            "speed": speed,
        }

    @staticmethod
    def question_sets(objective: str) -> list[dict]:
        return [dict(QUESTIONS)]

    def questions(self, objective: str, route_done: bool) -> dict:
        return dict(QUESTIONS)

    def apply(self, value: dict[str, float]) -> None:
        self.sticks = None
        self.heading_target = controls.select(self.heading_target, value["turn"], self.heading)
        self.climb_target, self.speed_target = value["climb"] / airfield.FPM, value["speed"] * airfield.KT

    def hold(self) -> None:
        self.sticks = None
        self.heading_target, self.climb_target, self.speed_target = self.heading, 0.0, 0.0

    def heading_held(self) -> float | None:
        return None if self.sticks else self.heading_target

    def control(self, values: dict[str, float]) -> None:
        """Set a person's sticks (climb, speed and yaw rate)"""
        if unknown := sorted(set(values) - set(controls.ACTUATORS)):
            raise ValueError(f"Unknown controls {unknown}; choose from {list(controls.ACTUATORS)}")
        sticks = self.sticks or {"throttle": 0.5, "elevator": 0.0, "aileron": 0.0, "pedals": 0.0}
        self.sticks = sticks | {
            name: airfield.clamp(float(value), *controls.ACTUATORS[name]) for name, value in values.items()
        }

    def tick(self) -> dict:
        dt = 1 / HZ
        if sticks := self.sticks:
            lift = sticks["throttle"] - 0.5
            climb_target = 0.0 if abs(lift) < 0.05 else lift * 2 * 1000 / airfield.FPM
            speed_target = max(0.0, sticks["elevator"]) * self.TOP_SPEED
            rate = airfield.clamp((sticks["aileron"] + sticks["pedals"]) * self.max_yaw, -self.max_yaw, self.max_yaw)
        else:
            climb_target, speed_target = self.climb_target, self.speed_target
            rate = airfield.clamp(
                YAW_GAIN * airfield.wrap(self.heading_target - self.heading), -self.max_yaw, self.max_yaw
            )
        grounded = self.agl <= airfield.PARKED_AGL_M
        if grounded and climb_target <= 0:
            self.climb = 0.0
        else:
            self.climb += (climb_target - self.climb) * min(1.0, dt / self.climb_seconds)
        flying = not grounded or self.climb > 0
        self.yaw_rate = rate if flying else 0.0
        self.heading = (self.heading + self.yaw_rate * dt) % 360
        wanted = speed_target if flying else 0.0
        self.accel = airfield.clamp((wanted - self.speed) / self.speed_seconds, -self.max_accel, self.max_accel)
        self.speed = max(0.0, self.speed + self.accel * dt)
        heading = math.radians(self.heading)
        self.push = self.gusting(dt) if flying else (0.0, 0.0)
        self.north += (self.speed * math.cos(heading) + self.push[0]) * dt
        self.east += (self.speed * math.sin(heading) + self.push[1]) * dt
        self.agl += self.climb * dt
        if self.agl <= airfield.PARKED_AGL_M:
            if not grounded and self.speed > TIP_KT * airfield.KT:
                self.struck = ["a propeller"]
            self.agl, self.climb, self.speed, self.yaw_rate, self.accel = airfield.PARKED_AGL_M, 0.0, 0.0, 0.0, 0.0
        self.ticks += 1
        self.sample = self.measure()
        return self.sample

    def gusting(self, dt: float) -> tuple[float, float]:
        """Step the gusts and return the push the controller hasn't corrected yet"""
        if not self.gusts:
            return 0.0, 0.0
        decay = math.exp(-dt / GUST_SECONDS)
        for i in (0, 1):
            kick = self.gusts * math.sqrt(1 - decay * decay) * self.gust_rng.gauss(0.0, 1.0)
            self.gust[i] = self.gust[i] * decay + kick
            self.caught[i] += (self.gust[i] - self.caught[i]) * dt / CATCH_SECONDS
        return self.gust[0] - self.caught[0], self.gust[1] - self.caught[1]

    def fly_from(self, north: float, east: float, height: float, heading: float, answers: dict[str, str]) -> None:
        values = {name: QUESTIONS[name][1][option][0] for name, option in answers.items()}
        self.north, self.east, self.agl = north, east, height + airfield.PARKED_AGL_M
        self.heading = self.heading_target = heading
        self.speed, self.climb = values["speed"] * airfield.KT, values["climb"] / airfield.FPM
        self.sample = self.measure()

    def measure(self) -> dict:
        heading = math.radians(self.heading)
        north = self.speed * math.cos(heading) + self.push[0]
        east = self.speed * math.sin(heading) + self.push[1]
        air = math.hypot(north - self.air[0] - self.gust[0], east - self.air[1] - self.gust[1])
        return {
            "t": self.ticks / HZ,
            "north_m": self.north,
            "east_m": self.east,
            "altitude_m": airfield.RUNWAY["elevation_m"] + self.agl,
            "agl_m": self.agl,
            "airspeed_kt": air / airfield.KT,
            "ground_speed_kt": math.hypot(north, east) / airfield.KT,
            "climb_fpm": self.climb * airfield.FPM,
            # banks into turns, and pitches into the wind and to accelerate
            "bank_deg": math.degrees(math.atan2(self.speed * math.radians(self.yaw_rate), G)),
            "pitch_deg": -math.degrees(math.atan((DRAG * air * air + self.accel) / G)),
            "heading_deg": self.heading,
            "on_ground": self.agl <= airfield.PARKED_AGL_M,
            "nose_gear_first": False,
            "struck": list(self.struck),
        }

    def instruments(self) -> dict:
        return {
            **self.sample,
            "track_deg": self.heading,  # position mode cancels wind drift
            "roll_rate_dps": 0.0,
            "pitch_rate_dps": 0.0,
            "yaw_rate_dps": self.yaw_rate,
            "sideslip_deg": 0.0,
        }

    def stall_warning(self) -> bool:
        return False

    def scene(self) -> dict:
        if self.sticks:
            return {"controls": {"throttle": self.sticks["throttle"]}}
        # mid-throttle holds height, like a real drone's spring-centred stick
        return {"controls": {"throttle": 0.5 + airfield.clamp(self.climb_target * airfield.FPM / 2000, -0.5, 0.5)}}

    def metrics(self) -> dict:
        return {}

    def close(self) -> None:
        pass

    def final_course(self, s: dict) -> float:
        return airfield.bearing_to(s=s, point=SPOT) if spot_distance(s) > 3 else s["heading_deg"]

    def final_target(self, s: dict, direction: str) -> dict:
        height = s["agl_m"] - airfield.PARKED_AGL_M
        seconds = spot_distance(s) / max(s["ground_speed_kt"] * airfield.KT, 3.0)
        return {
            "landing_spot": f"on the runway centreline, {airfield.AIM_PAST_THRESHOLD_M:g} m past its threshold",
            "distance_m": airfield.distance(s=s, point=SPOT),
            "direction": direction,
            "height_difference": airfield.vertical_text(-height),
            "climb_needed_fpm": int(round(-height / max(seconds, 5.0) * airfield.FPM, -1)),
        }


def spot_distance(s: dict) -> float:
    return math.hypot(SPOT["north_m"] - s["north_m"], SPOT["east_m"] - s["east_m"])


def off_spot(along: float, right: float) -> float:
    spot_along, spot_right = airfield.runway_frame(SPOT["north_m"], SPOT["east_m"])
    return math.hypot(along - spot_along, right - spot_right)


class DroneEvaluator(objectives.Evaluator):
    def took_off(self, s: dict, height: float) -> bool:
        return self.lifted_off and not s["on_ground"] and height >= objectives.TAKEOFF["height_m"]

    def stable(self, s: dict, along: float, right: float) -> bool:
        gate, height = APPROACH, s["agl_m"] - airfield.PARKED_AGL_M
        return (
            spot_distance(s) <= gate["distance_m"]
            and gate["height_m"][0] <= height <= gate["height_m"][1]
            and s["ground_speed_kt"] <= gate["ground_speed_kt"]
            and gate["climb_fpm"][0] <= s["climb_fpm"] <= gate["climb_fpm"][1]
        )

    def problems(self, t: dict) -> list[str]:
        found = []
        if t["sink_fpm"] > LANDING["sink_fpm"]:
            found.append(f"touchdown sink {t['sink_fpm']:.0f} fpm exceeds {LANDING['sink_fpm']:.0f}")
        if not self.on_landing_area(t["along_m"], t["right_m"]):
            found.append(f"touchdown {off_spot(t['along_m'], t['right_m']):.0f} m from the landing spot")
        return found

    def on_landing_area(self, along: float, right: float) -> bool:
        return off_spot(along, right) <= LANDING["radius_m"]


def landing(s: dict, height: float) -> tuple[float, float, str]:
    """Steer the scripted pilot to 8 m over the spot, stop there, then descend"""
    left = spot_distance(s)
    course = airfield.bearing_to(s=s, point=SPOT) if left > 3 else s["heading_deg"]
    facing = abs(airfield.wrap(course - s["heading_deg"])) < 30
    if left > 12:
        speed = "45 kt" if left > 250 else "15 kt" if facing else "hover"
        climb = 15 * (min(150.0, 8.0 + 0.1 * (left - 12)) - height)
    elif s["ground_speed_kt"] > 2:
        speed, climb = "hover", 15 * (8.0 - height)
    else:
        speed, climb = "hover", -500.0 if height > 15 else -250.0 if height > 4 else -100.0
    return course, climb, speed
