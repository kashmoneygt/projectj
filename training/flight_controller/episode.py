import math
import random
from typing import NamedTuple

from flight_controller import reward
from jym import core, environments, runner
from jym.environments.flight import airfield, environment, objectives

# fixed, so new objectives don't change the critic's inputs
KNOWN_OBJECTIVES = ("takeoff", "rings", "course", "circuit")
FEATURES = len(KNOWN_OBJECTIVES) + len(environment.AIRCRAFT) + 20  # one-hot objective and aircraft, plus 20 more


class Job(NamedTuple):
    task: str
    seed: int
    handover: int = 0  # checkpoints the scripted pilot flies before handing over
    settings: dict | None = None  # added to the objective's own
    takeover: float = 0.0  # seconds of random answers
    takeover_at: float = 0.0  # seconds after the policy takes control


class Episode:
    """One flight of a job, with answers applied after a delay and decisions at least interval apart"""

    def __init__(self, job: Job, delay: float = 0.0, interval: float = runner.DECISION_SECONDS, cap: int | None = None):
        self.job, self.seed, self.delay, self.interval, self.cap = job, job.seed, delay, interval, cap
        self.task = environments.get_task(env="flight", task=job.task)
        if job.settings:
            self.task = self.task.model_copy(update={"settings": {**self.task.settings, **job.settings}})
        self.budget = self.task.max_steps * runner.DECISION_SECONDS  # sim seconds allowed, however often it's asked
        self.env = environment.FlightEnvironment()
        self.env.reset(self.task, job.seed)
        self.start = {"north_m": self.env.sample["north_m"], "east_m": self.env.sample["east_m"]}
        self.expert = self.env.reference()
        self.chosen: dict[str, int] = {}
        self.steps, self.seconds, self.done, self.truncated = 0, 0.0, False, False
        self.observe()
        while reward.checkpoints(self.env.evaluator) < job.handover and not self.done:
            self.env.apply(self.expert_choices())
            self.done = self.env.advance(runner.DECISION_SECONDS)
            self.observe()
        self.takeover_time = self.env.sample["t"] + job.takeover_at if job.takeover else math.inf
        self.progress = reward.progress(self.env, self.start)

    @property
    def name(self) -> str:
        """Return the objective's id tagged with @handover, +takeover or ~air start"""
        start = self.task.settings.get("start", "runway")
        handover = f"@{self.job.handover}" if self.job.handover else ""
        takeover = "+takeover" if self.job.takeover else ""
        return self.task.id + handover + takeover + ("" if start == "runway" else f"~{start}")

    def observe(self) -> None:
        self.state = {"goal": self.task.goal, **self.env.state()}
        self.questions = self.env.questions()

    def expert_choices(self, state: dict | None = None) -> dict[str, str]:
        reply = self.expert.decide(self.state if state is None else state, self.questions)
        return {qid: answer["choice"] for qid, answer in reply.answers.items()}

    def step(self, choices: dict[str, str] | None) -> tuple[float, dict[str, str]]:
        """Apply the answers, or the scripted pilot's if None, fly on, and return the reward and label"""
        asked, ev = self.questions, self.env.evaluator
        label = self.expert_choices()
        before, touchdowns, shown_at = set(ev.milestones), len(ev.touchdowns), self.env.sample["t"]
        ended = self.delay > 0 and self.env.advance(self.delay)
        # label with the scripted pilot's answer at the time it applies, so the policy learns to answer ahead
        if self.delay > 0 and not ended and self.env.questions() == asked:
            label = self.expert_choices({"goal": self.task.goal, **self.env.state()})
        choices = choices or label
        moved = self.moved(asked, choices)
        if not ended:
            try:
                self.env.apply(choices)
            except ValueError:  # the questions changed before it applied, so drop it (as in a live heat)
                if not self.delay:
                    raise
            else:
                ended = self.env.advance(max(self.interval - self.delay, 1 / self.env.HZ))
        if not ended and self.env.sample["t"] >= self.takeover_time:  # a takeover can start mid-decision, like a gust
            ended = self.take_over()
        self.steps += 1
        self.seconds = self.env.sample["t"] - shown_at
        reward_value = self.reward_for(before, touchdowns, moved, ended)
        self.done = ended or self.env.sample["t"] >= self.budget or (self.cap is not None and self.steps >= self.cap)
        self.truncated = self.done and not ended
        if not self.done:
            self.observe()
        return reward_value, label

    def moved(self, asked: dict, choices: dict[str, str]) -> float:
        """Measure how far the answers moved from the last ones, each as a share of its options"""
        indices = {q: list(asked[q]["criteria"]).index(o) for q, o in choices.items()}
        moved = sum(
            abs(i - self.chosen[q]) / max(1, len(asked[q]["criteria"]) - 1)
            for q, i in indices.items()
            if q in self.chosen
        )
        self.chosen = indices
        return moved

    def reward_for(self, before: set, touchdowns: int, moved: float, ended: bool) -> float:
        """Reward the milestones and progress this decision earned, minus its costs"""
        r, ev, earlier = reward.REWARD, self.env.evaluator, self.progress
        # progress drops to 0 at the end, so it sums to zero over a flight (potential-based shaping)
        self.progress = 0.0 if ended else reward.progress(self.env, self.start)
        made = r.discount(self.seconds) * self.progress - earlier
        hardness = sum(min(1.0, max(0.0, t["sink_fpm"] - 300.0) / 300.0) for t in ev.touchdowns[touchdowns:])
        return (
            sum(self.task.milestones[name] for name in ev.milestones - before) / 100
            + r.progress_weight * made
            - r.crash * (ev.ending == "crashed")
            - r.hard_touchdown * hardness
            - r.time * self.seconds
            - r.answer_change * moved
        )

    def take_over(self) -> bool:
        self.takeover_time, noise = math.inf, random.Random(f"takeover-{self.seed}")
        for _ in range(round(self.job.takeover / runner.DECISION_SECONDS)):
            self.env.apply({q: noise.choice(list(o["criteria"])) for q, o in self.env.questions().items()})
            if self.env.advance(runner.DECISION_SECONDS):
                return True
        return False

    def score(self) -> int:
        return core.score(task=self.task, evaluation=self.env.evaluate())

    def result(self) -> dict:
        evaluation = self.env.evaluate()
        return {
            "task": self.name,
            "seed": self.seed,
            "delay": round(self.delay, 3),
            "interval": round(self.interval, 3),
            "score": core.score(self.task, evaluation),
            "milestones": sorted(evaluation.milestones),
            "failure": evaluation.failure or (None if evaluation.success else self.env.evaluator.ending or "time out"),
            "decisions": self.steps,
        }

    def features(self) -> list[float]:
        """Return the critic's privileged inputs from the simulator, objective and progress"""
        env, ev = self.env, self.env.evaluator
        s = env.instruments()
        height = s["agl_m"] - airfield.PARKED_AGL_M
        turn = math.radians(airfield.wrap(env.course() - s["heading_deg"]))
        holding = env.aircraft.heading_held()
        held = math.radians(airfield.wrap((s["heading_deg"] if holding is None else holding) - env.course()))
        target = reward.target_point(env)
        distance = math.hypot(target["north_m"] - s["north_m"], target["east_m"] - s["east_m"]) if target else 0.0
        along_runway, right = airfield.runway_frame(s["north_m"], s["east_m"])
        return [
            *(float(env.objective == name) for name in KNOWN_OBJECTIVES),
            *(float(objectives.aircraft_of(self.task) == name) for name in environment.AIRCRAFT),
            float(s["on_ground"]),
            float("takeoff" in ev.milestones),
            float("route" in ev.milestones),
            s["airspeed_kt"] / 100,
            height / 300,
            s["climb_fpm"] / 1000,
            s["bank_deg"] / 45,
            s["pitch_deg"] / 20,
            s["sideslip_deg"] / 10,
            math.sin(turn),
            math.cos(turn),
            min(distance, 4000.0) / 4000,
            ((target["height_m"] - height) / 200) if target else 0.0,
            along_runway / 1000,
            max(-3.0, min(3.0, right / 50)),
            self.progress,
            sum(self.task.milestones[name] for name in ev.milestones) / 100,
            max(env.sample["t"] / self.budget, self.steps / self.cap if self.cap else 0.0),  # the nearer limit
            math.sin(held),  # held heading relative to the course
            math.cos(held),
        ]
