from typing import ClassVar

from jym import core

COUNTER = core.Task(
    env="counter",
    id="reach-3",
    title="Reach 3",
    goal="Count up to 3.",
    version="1",
    milestones={"one": 30, "two": 30},
    max_steps=10,
    max_seconds=30,
)


# a tiny test game: three "up" answers win and "boom" crashes
class Counter:
    version = "test"
    HZ = 1
    RELEASED: ClassVar[dict] = {}

    def __init__(self):
        self.closed = False

    def reset(self, task, seed):
        self.value, self.crashed, self.controls, self.ticks = 0, False, {}, 0

    def state(self):
        return {"value": self.value}

    def questions(self):
        return {
            "step": core.choice(
                instructions="Which way should the counter go?",
                options={"up": "add one", "stay": "do nothing", "boom": "crash"},
            )
        }

    def apply(self, choices):
        self.value += choices["step"] == "up"
        self.crashed = choices["step"] == "boom"

    def hold(self):
        pass

    def control(self, values):
        self.controls |= values

    def advance(self, seconds):
        self.ticks += 1
        self.value += self.controls.get("speed", 0)
        return self.value >= 3 or self.crashed

    def evaluate(self):
        reached = {name for name, n in (("one", 1), ("two", 2)) if self.value >= n}
        return core.Evaluation(success=self.value >= 3, milestones=reached, failure="crashed" if self.crashed else None)

    def scene(self):
        return {"value": self.value, "sim_time": float(self.value)}

    def reference(self):
        return Climber()

    def close(self):
        self.closed = True


class Climber:
    name = "reference"
    synchronous = True

    def decide(self, state, questions):
        return core.picked(step="up")


SPEC = core.EnvironmentSpec(
    id="counter",
    title="Counter",
    about="test",
    environment=Counter,
    tasks=[COUNTER],
    objective="reach-3",
    describe=lambda: {"state": {"value": "the count"}, "phases": {}, "timing": ""},
    question_sets=lambda task: [Counter().questions()],
    reference={"title": "Climber"},
    pose=lambda scene: [scene["sim_time"], scene["value"]],
)
