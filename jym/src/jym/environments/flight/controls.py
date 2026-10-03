from jym import core
from jym.environments.flight import airfield

# controls a person moves directly, and their ranges
ACTUATORS = {
    "throttle": (0.0, 1.0),  # 0 idle, 1 full power
    "elevator": (-1.0, 1.0),  # -1 nose up, +1 nose down
    "aileron": (-1.0, 1.0),  # -1 roll left, +1 roll right
    "pedals": (-1.0, 1.0),  # rudder and nosewheel; +1 yaws right
    "brake": (0.0, 1.0),  # both wheel brakes
    "flaps": (0.0, 1.0),  # 0 up, 1/3 10 deg, 2/3 20 deg, 1 30 deg
    "trim": (-1.0, 1.0),  # added to the elevator; -1 nose up
    "roll_trim": (-1.0, 1.0),  # added to the aileron
}

# options map to (setpoint, description); setpoints hold until the next answer
TURN = (
    "Which way should the aircraft turn now? Each turn moves the heading it holds by that much.",
    {
        "left 60°": (-60.0, "a big turn left"),
        "left 20°": (-20.0, "turn left"),
        "left 5°": (-5.0, "a small correction left"),
        "straight": (0.0, "hold the current heading"),
        "right 5°": (5.0, "a small correction right"),
        "right 20°": (20.0, "turn right"),
        "right 60°": (60.0, "a big turn right"),
    },
)
CLIMB = (
    "Which climb rate or landing flare should the aircraft hold?",
    {
        "climb fast": (1000.0, "+1000 ft/min"),
        "climb": (500.0, "+500 ft/min"),
        "level": (0.0, "hold the current height"),
        "flare": (-100.0, "raise the nose to touch down gently at about -100 ft/min"),
        "descend slowly": (-250.0, "-250 ft/min"),
        "descend": (-500.0, "-500 ft/min"),
        "descend fast": (-1000.0, "-1000 ft/min"),
    },
)


def nearest(question: tuple, value: float) -> str:
    options = question[1]
    return min(options, key=lambda option: abs(options[option][0] - value))


def select(held: float, turn: float, heading: float) -> float:
    """Add a turn to the held heading, at most 90 degrees off the current heading"""
    return (heading + airfield.clamp(airfield.wrap(held + turn - heading), -90.0, 90.0)) % 360


def as_questions(questions: dict) -> dict:
    return {
        name: core.choice(text, {option: about for option, (_, about) in options.items()})
        for name, (text, options) in questions.items()
    }
