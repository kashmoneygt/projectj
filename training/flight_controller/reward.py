import math
from dataclasses import dataclass

from jym import runner
from jym.environments.flight import airfield, cessna, environment, objectives


@dataclass(frozen=True)
class Reward:
    """Reward weights for Episode.reward_for, and discounts per half second of flight"""

    progress_weight: float = 0.5
    crash: float = 0.5
    hard_touchdown: float = 0.2
    time: float = 0.0004
    answer_change: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95

    def discount(self, seconds: float, per_half_second: float | None = None) -> float:
        """Discount a decision that held this many seconds by gamma (or per_half_second) per half second"""
        return (self.gamma if per_half_second is None else per_half_second) ** (seconds / runner.DECISION_SECONDS)


REWARD = Reward()


def progress(env: environment.FlightEnvironment, start: dict) -> float:
    """Return the flight's progress through its objective from 0 to 1, tracking its milestone points"""
    s, ev, objective = env.sample, env.evaluator, env.objective
    drone = objectives.aircraft_of(env.task) == "drone"
    height = s["agl_m"] - airfield.PARKED_AGL_M
    lift = min(1.0, max(0.0, height) / 30.0)
    if not drone:  # a take-off roll builds speed first
        lift = 0.5 * min(1.0, s["airspeed_kt"] / cessna.LIFTOFF_KT) + 0.5 * lift
    if "takeoff" not in ev.milestones:
        return lift * (1.0 if objective == "takeoff" else 0.2)
    if objective == "takeoff":
        return 1.0
    if ev.rings:
        return 0.2 + 0.8 * along(s=s, height=height, start=start, points=ev.rings, index=ev.ring)
    if "route" not in ev.milestones:  # the circuit scores take-off 20, route 30, approach 20 and landing 30
        points = [{**w, "height_m": sum(w["height_m"]) / 2} for w in env.waypoints]
        return 0.2 + 0.3 * along(s=s, height=height, start=start, points=points, index=ev.waypoint)
    along_runway, right = airfield.runway_frame(s["north_m"], s["east_m"])
    if s["on_ground"] and ev.on_landing_area(along_runway, right):
        return 0.7 + 0.3 * max(0.0, 1.0 - s["ground_speed_kt"] / 60.0)
    # progress along the glide path to the aim point, or straight down for the drone
    aim = airfield.THRESHOLD_NORTH_M + airfield.AIM_PAST_THRESHOLD_M
    base = env.waypoints[-1]
    total = math.hypot(aim - base["north_m"], base["east_m"])
    glide = height if drone else 2 * abs(airfield.glide_error(along_runway, s["agl_m"]))
    left = math.hypot(aim - s["north_m"], s["east_m"]) + glide
    return 0.5 + 0.2 * max(0.0, 1.0 - left / total)


def along(s: dict, height: float, start: dict, points: list[dict], index: int) -> float:
    """Return the share of a route flown, counting finished legs and part of the current one"""
    if index >= len(points):
        return 1.0
    legs, previous = [], start
    for point in points:
        legs.append(math.hypot(point["north_m"] - previous["north_m"], point["east_m"] - previous["east_m"]))
        previous = point
    target = points[index]
    left = math.hypot(target["north_m"] - s["north_m"], target["east_m"] - s["east_m"])
    left = math.hypot(left, 2.0 * (target["height_m"] - height))
    return min(1.0, (sum(legs[:index]) + max(0.0, legs[index] - left)) / sum(legs))


def target_point(env: environment.FlightEnvironment) -> dict | None:
    """Return the point the state's target describes, such as a ring, waypoint or landing spot"""
    ev = env.evaluator
    if ev.rings:
        return ev.rings[min(ev.ring, len(ev.rings) - 1)]
    if env.objective != "circuit" or "takeoff" not in ev.milestones:
        return None
    if "route" not in ev.milestones:
        waypoint = env.waypoints[ev.waypoint]
        return {**waypoint, "height_m": sum(waypoint["height_m"]) / 2}
    return {"north_m": airfield.THRESHOLD_NORTH_M + airfield.AIM_PAST_THRESHOLD_M, "east_m": 0.0, "height_m": 0.0}


def checkpoints(ev: objectives.Evaluator) -> int:
    """Count the checkpoints passed, including take-off and the circuit's stable approach"""
    return ("takeoff" in ev.milestones) + ev.ring + ev.waypoint + ("approach" in ev.milestones)
