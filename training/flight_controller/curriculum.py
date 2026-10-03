import random
from dataclasses import dataclass, field

from flight_controller import episode
from jym import environments, runner
from jym.environments.flight import objectives

# train only on seeds from TRAIN_SEED up, never on development (0-99) or test seeds
TRAIN_SEED, TEST_SEEDS = 10_000, "100-119"
# train on both aircraft so one model flies either
TRAIN_TASKS = {
    "takeoff": 2.0,  # take-offs are short, so this is still little data
    "rings": 1.5,
    "course": 2.5,
    "circuit": 4.0,
    "drone-takeoff": 2.0,
    "drone-rings": 1.0,
    "drone-course": 2.0,
    "drone-circuit": 3.0,
}
EVAL_TASKS = tuple(TRAIN_TASKS)
# weights for how many checkpoints the scripted pilot flies before handing over
HANDOVERS = {"circuit": {0: 4, 3: 2, 4: 2, 5: 2, 6: 1}, "rings": {0: 3, 1: 1, 2: 1}, "course": {0: 3, 1: 1, 2: 1, 3: 1}}
# a few seconds of random answers teach recovery (as in DART)
TAKEOVER_SHARE, TAKEOVER_SECONDS, TAKEOVER_WHEN = 0.3, (2.0, 8.0), 0.4
VARIETY_DYNAMICS, VARIETY_GUSTS_KT, VARIETY_NOISE = 0.2, 8.0, 1.0
# with variety, some circuits start in the air to practise landings early
AIR_STARTS = {"runway": 0.4, "downwind": 0.3, "final": 0.3}


def task_weights(tasks: str) -> dict[str, float]:
    """Parse weights like "circuit=3,course=1"; missing weights come from TRAIN_TASKS"""
    if not tasks:
        return dict(TRAIN_TASKS)
    pairs = [part.partition("=") for part in tasks.split(",")]
    return {name: float(weight) if weight else TRAIN_TASKS[name] for name, _, weight in pairs}


def span(text: str) -> tuple[float, float]:
    """Parse seconds ("1.5") or a range ("0.1-2.5") that each flight draws from"""
    low, dash, high = text.partition("-")
    return (float(low), float(high)) if dash else (float(low), float(low))


def varied_settings(task: str, rng: random.Random, air_starts: bool = True) -> dict:
    settings = {
        "airfield": "varied",
        "dynamics": VARIETY_DYNAMICS,
        "gusts_kt": round(rng.uniform(0.0, VARIETY_GUSTS_KT), 1) if rng.random() < 0.5 else 0.0,
    }
    if air_starts and objectives.objective_of(environments.get_task(env="flight", task=task)) == "circuit":
        settings["start"] = rng.choices(list(AIR_STARTS), list(AIR_STARTS.values()))[0]
    settings["noise"] = round(rng.uniform(0.0, VARIETY_NOISE), 2)
    return settings


def takeover(task: str, rng: random.Random) -> tuple[float, float]:
    budget = environments.get_task(env="flight", task=task).max_steps * runner.DECISION_SECONDS
    return rng.uniform(*TAKEOVER_SECONDS), rng.uniform(0.0, TAKEOVER_WHEN * budget)


def seed_list(text: str) -> list[int]:
    low, dash, high = text.partition("-")
    return list(range(int(low), int(high) + 1)) if dash else [int(part) for part in text.split(",")]


@dataclass(frozen=True)
class Curriculum:
    tasks: dict[str, float] = field(default_factory=lambda: dict(TRAIN_TASKS))
    delay: tuple[float, float] = (0.1, 2.5)
    interval: tuple[float, float] = (runner.DECISION_SECONDS, runner.DECISION_SECONDS)
    wind: float = 12.0  # max knots, or 0 for each objective's default
    variety: bool = False
    takeovers: bool = True
    handovers: bool = False

    def draw(self, rng: random.Random) -> tuple[episode.Job, float, float]:
        task = rng.choices(list(self.tasks), list(self.tasks.values()))[0]
        points = HANDOVERS.get(objectives.objective_of(environments.get_task(env="flight", task=task)), {0: 1})
        handover = rng.choices(list(points), list(points.values()))[0] if self.handovers else 0
        seconds, at = takeover(task, rng) if self.takeovers and rng.random() < TAKEOVER_SHARE else (0.0, 0.0)
        settings = ({"wind_max_kt": self.wind} if self.wind else {}) | (
            varied_settings(task, rng) if self.variety else {}
        )
        # after a handover, the takeover starts right away
        job = episode.Job(
            task=task,
            seed=TRAIN_SEED + rng.randrange(10**9),
            handover=handover,
            settings=settings or None,
            takeover=seconds,
            takeover_at=0.0 if handover else at,
        )
        return job, rng.uniform(*self.delay), rng.uniform(*self.interval)
