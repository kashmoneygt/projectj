import hashlib
import math
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import jsbsim

from jym import core, paths
from jym.environments.flight import airfield, controls, objectives

if TYPE_CHECKING:
    from jym.environments.flight.environment import FlightEnvironment

HZ = 120
# trim for a full-power climb (this c172x rolls right and pitches up hands-off)
START_TRIM = {"trim": 0.15, "roll_trim": -0.12}
LIFTOFF_KT = 55
# autopilot bank per degree of heading error, up to MAX_BANK_DEG
HEADING_GAIN, MAX_BANK_DEG = 1.5, 30.0

# the approach gate must hold for 15 s on final
APPROACH = {
    "before_threshold_m": (150.0, 3000.0),
    "off_centre_m": 60.0,
    "heading_deg": 15.0,
    "airspeed_kt": (55.0, 80.0),
    "climb_fpm": (-1000.0, -150.0),
    "bank_deg": 15.0,
    "height_m": (10.0, 300.0),
}

LANDING = {
    "sink_fpm": 400.0,
    "airspeed_kt": 70.0,
    "bank_deg": 10.0,
    "heading_deg": 10.0,
    "edge_margin_m": 3.0,
}

PROPERTIES = {
    "throttle": ["fcs/throttle-cmd-norm"],
    "elevator": ["fcs/elevator-cmd-norm"],
    "aileron": ["fcs/aileron-cmd-norm"],
    "brake": ["fcs/left-brake-cmd-norm", "fcs/right-brake-cmd-norm"],
    "flaps": ["fcs/flap-cmd-norm"],
    "trim": ["fcs/pitch-trim-cmd-norm"],
    "roll_trim": ["fcs/roll-trim-cmd-norm"],
}
THROTTLE = (
    "Which throttle setting should the pilot select now?",
    {
        "idle": (0.0, "no power"),
        "half": (0.5, "cruise power: slower flight turns tighter"),
        "full": (1.0, "maximum power for takeoff and climbing"),
    },
)
ON_GROUND = {
    "brakes": (
        "Should the pilot hold the wheel brakes now?",
        {
            "off": (0.0, "released: the aircraft can roll forward and take off"),
            "on": (1.0, "held: the aircraft stops or stays still"),
        },
    ),
    "throttle": THROTTLE,
    "turn": controls.TURN,
    "lift_off": (
        f"Should the pilot pull back and lift off now? The aircraft flies from about {LIFTOFF_KT} kt.",
        {
            "not yet": (0.0, "keep the nosewheel on the runway"),
            "lift off": (-0.4, "raise the nose and climb away"),
        },
    ),
}
# after landing the policy only has to stop
ROLLOUT = {
    "brakes": (
        "Should the pilot hold the wheel brakes now to stop on the runway?",
        {
            "off": (0.0, "released: keep rolling"),
            "on": (1.0, "held: slow down and stop on the runway"),
        },
    ),
    "throttle": (
        "Which throttle setting should the pilot select now to slow down on the runway?",
        {
            "idle": (0.0, "idle power for the landing rollout"),
            "half": (0.5, "some power, only if needed to keep moving"),
            "full": (1.0, "maximum power for a go-around"),
        },
    ),
    "turn": controls.TURN,
}
IN_AIR = {
    "turn": controls.TURN,
    "climb": controls.CLIMB,
    "speed": (
        "Which airspeed should the autothrottle hold?",
        {
            "idle": (0.0, "no power: to flare, land or descend steeply"),
            "65 kt": (65.0, "approach and slow flight"),
            "80 kt": (80.0, "climbing"),
            "100 kt": (100.0, "cruise"),
        },
    ),
    "flaps": (
        "Which flap setting should the pilot select now?",
        {
            "up": (0.0, "takeoff, climb and cruise"),
            "10°": (1 / 3, "early approach, below 110 kt"),
            "20°": (2 / 3, "approach, below 85 kt"),
            "30°": (1.0, "final approach and landing, below 85 kt"),
        },
    ),
}

C172X_SHA256 = "b736f4b7560d5aab4d2a69026798be1d369b2b2bb5ddcf627aebfd441ba71840"
# remove the built-in autopilot (it zeroes the nosewheel every frame), auto mixture and CSV logging
REMOVED = (
    r'\n\s*<output name="JSBout172B.csv".*?</output>',
    r'\n\s*<system name="Mixture control">.*?\n    </system>',
    r'\n\s*<system file="GNCUtilities"/>',
    r'\n\s*<system file="Autopilot">.*?</system>',
    r'\n\s*<autopilot file="c172ap"/>',
)
# the flight controls still add these autopilot outputs, so pin them to 0
AUTOPILOT_ZERO = """
    <system name="Autopilot removed">
      <property value="0"> ap/elevator_cmd </property>
      <property value="0"> ap/aileron_cmd </property>
      <property value="0"> ap/roll-cmd-norm-output </property>
    </system>"""


def aircraft_path() -> Path:
    """Write the patched c172x model and return its aircraft folder"""
    source = Path(jsbsim.get_default_root_dir(), "aircraft", "c172x", "c172x.xml")
    text = source.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != C172X_SHA256:
        raise core.Unavailable(f"{source} differs from the pinned c172x model")
    for pattern in REMOVED:
        text = re.sub(pattern, "", text, count=1, flags=re.DOTALL)
    text = text.replace('\n    <flight_control name="c172">', AUTOPILOT_ZERO + '\n    <flight_control name="c172">', 1)
    target = paths.ROOT / ".cache" / "flight-model" / "aircraft" / "c172x" / "c172x.xml"
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return target.parents[1]


def asked(objective: str, phase: dict) -> dict:
    """Return a phase's questions; only the circuit asks about flaps"""
    questions = dict(phase)
    if objective != "circuit":
        questions.pop("flaps", None)
    return questions


class Cessna:
    """Cessna 172 on JSBSim, with an autopilot for heading, climb rate and airspeed"""

    HZ = HZ
    ROLLS = True  # takes off and lands on the runway
    START: ClassVar[dict] = {
        "brakes": "on",
        "throttle": "idle",
        "lift_off": "not yet",
        "turn": "straight",
        "climb": "level",
        "speed": "80 kt",
        "flaps": "up",
    }
    AIRBORNE: ClassVar[dict] = {  # answers in force for circuits that start in the air
        "downwind": {"turn": "straight", "climb": "level", "speed": "80 kt", "flaps": "up", "brakes": "off"},
        "final": {"turn": "straight", "climb": "descend slowly", "speed": "65 kt", "flaps": "20°", "brakes": "off"},
    }

    def __init__(self, wind: dict, dynamics: float, gusts: float, seed: int):
        jsbsim.FGJSBBase().debug_lvl = 0
        fdm = jsbsim.FGFDMExec(None)
        fdm.set_debug_level(0)
        fdm.set_aircraft_path(str(aircraft_path()))
        fdm.set_engine_path(str(Path(jsbsim.get_default_root_dir(), "engine")))
        fdm.set_systems_path(str(Path(jsbsim.get_default_root_dir(), "systems")))
        if not fdm.load_model("c172x"):
            raise core.Unavailable("JSBSim could not load the c172x model")
        fdm.set_dt(1 / self.HZ)
        fdm["ic/lat-geod-deg"], fdm["ic/long-gc-deg"] = airfield.geodetic(north=airfield.START_NORTH_M, east=0.0)
        fdm["ic/terrain-elevation-ft"] = airfield.RUNWAY["elevation_m"] / airfield.FT
        fdm["ic/h-agl-ft"] = airfield.PARKED_AGL_M / airfield.FT
        fdm["ic/psi-true-deg"] = airfield.RUNWAY["heading_deg"]
        fdm["propulsion/set-running"] = -1
        if not fdm.run_ic():
            raise core.Unavailable("JSBSim rejected the initial conditions")
        blowing = math.radians(wind["from_deg"])
        speed = wind["speed_kt"] * airfield.KT / airfield.FT
        self.wind_fps = (-speed * math.cos(blowing), -speed * math.sin(blowing))  # north and east, pointing downwind
        self.fdm = fdm
        self.blow()
        fdm["fcs/mixture-cmd-norm"] = 1.0
        rng = random.Random(f"dynamics-{seed}")
        # vary the weight and weaken the engine
        extra = rng.uniform(-0.5, 0.5) * dynamics * fdm["inertia/weight-lbs"]
        fdm["inertia/pointmass-weight-lbs[1]"] = max(0.0, fdm["inertia/pointmass-weight-lbs[1]"] + extra)
        self.power = 1.0 - rng.uniform(0.0, dynamics)
        if gusts:
            fdm["atmosphere/turb-type"] = 3  # MIL-spec Dryden
            fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = gusts * airfield.KT / airfield.FT
            fdm["atmosphere/turbulence/milspec/severity"] = 1 if gusts <= 6 else 3  # light or moderate
            fdm["atmosphere/randomseed"] = seed  # same gusts on every replay of a seed
        self.controls: dict[str, float] = {}
        self.control(
            {
                "throttle": 0.0,
                "elevator": 0.0,
                "aileron": 0.0,
                "pedals": 0.0,
                "brake": 1.0,
                "flaps": 0.0,
                **START_TRIM,
            }
        )
        for _ in range(2 * self.HZ):  # let the gear settle before time zero
            fdm.run()
        self.ticks = 0
        self.fuel_start = fdm["propulsion/total-fuel-lbs"]
        self.autopilot: Autopilot | None = None
        self.sample = self.measure()

    def evaluator(self, task: core.Task, rings: list[dict], waypoints: list[dict]) -> "CessnaEvaluator":
        return CessnaEvaluator(task, rings, waypoints)

    def scripted_answers(self, environment: "FlightEnvironment") -> dict[str, str]:
        s, evaluator = self.instruments(), environment.evaluator
        along, right = airfield.runway_frame(s["north_m"], s["east_m"])
        height = s["agl_m"] - airfield.PARKED_AGL_M
        held = self.heading_held()
        held = s["heading_deg"] if held is None else held
        steer = controls.nearest(
            question=controls.TURN,
            value=airfield.wrap(airfield.RUNWAY["heading_deg"] + airfield.clamp(-0.5 * right, -5, 5) - held),
        )
        if self.sample["on_ground"] and evaluator.phase in ("rollout", "final"):
            return {"brakes": "on", "throttle": "idle", "turn": steer}
        if self.sample["on_ground"]:
            lift_off = "lift off" if s["airspeed_kt"] >= LIFTOFF_KT else "not yet"
            return {"brakes": "off", "throttle": "full", "turn": steer, "lift_off": lift_off}
        flaps, speed = "up", "80 kt"  # any faster and it turns too wide for the rings
        if evaluator.phase == "final":
            glide = (
                -s["ground_speed_kt"] * airfield.KT * math.tan(math.radians(airfield.GLIDE_SLOPE_DEG)) * airfield.FPM
            )
            climb = (
                0.0 if height < 6 else -250.0 if height < 15 else glide - 15 * airfield.glide_error(along, s["agl_m"])
            )
            flaps = "30°" if airfield.THRESHOLD_NORTH_M - along < 2000 else "10°"
            speed = "idle" if height < 15 else "65 kt"
            course = environment.course()
        else:
            course, climb = environment.navigation(height)
        # correct the heading for wind drift
        return {
            "turn": controls.nearest(
                question=controls.TURN,
                value=airfield.wrap(s["heading_deg"] + airfield.wrap(course - s["track_deg"]) - held),
            ),
            "climb": controls.nearest(question=controls.CLIMB, value=airfield.clamp(climb, -1000, 1000)),
            "speed": speed,
            "flaps": flaps,
        }

    @staticmethod
    def question_sets(objective: str) -> list[dict]:
        phases = [ON_GROUND, IN_AIR, ROLLOUT] if objective == "circuit" else [ON_GROUND, IN_AIR]
        return [asked(objective, phase) for phase in phases]

    def questions(self, objective: str, route_done: bool) -> dict:
        if not self.sample["on_ground"]:
            return asked(objective=objective, phase=IN_AIR)
        if objective == "circuit" and route_done:
            return asked(objective=objective, phase=ROLLOUT)
        return asked(objective=objective, phase=ON_GROUND)

    def apply(self, value: dict[str, float]) -> None:
        heading = self.sample["heading_deg"]
        self.autopilot = self.autopilot or Autopilot(heading)
        self.autopilot.heading_deg = controls.select(self.autopilot.heading_deg, value["turn"], heading)
        if "brakes" in value:
            # on the ground, answers set the throttle, brakes and elevator directly
            self.autopilot.climb_fpm = self.autopilot.speed_kt = None
            self.control(
                {"brake": value["brakes"], "throttle": value["throttle"], "elevator": value.get("lift_off", 0.0)}
            )
        else:
            self.autopilot.climb_fpm, self.autopilot.speed_kt = value["climb"], value["speed"]
            self.control({"brake": 0.0, "flaps": value.get("flaps", self.controls["flaps"])})

    def hold(self) -> None:
        heading = self.sample["heading_deg"]
        self.autopilot = self.autopilot or Autopilot(heading)
        self.autopilot.heading_deg = heading
        if self.sample["on_ground"]:
            self.autopilot.climb_fpm = self.autopilot.speed_kt = None
            self.control(values={"throttle": 0.0, "brake": 1.0, "elevator": 0.0})
        else:
            self.autopilot.climb_fpm = 0.0
            self.autopilot.speed_kt = self.autopilot.speed_kt or 80.0

    def heading_held(self) -> float | None:
        return self.autopilot.heading_deg if self.autopilot else None

    def blow(self) -> None:
        """Set the steady wind (JSBSim's initial conditions reset it)"""
        self.fdm["atmosphere/wind-north-fps"], self.fdm["atmosphere/wind-east-fps"] = self.wind_fps

    def fly_from(self, north: float, east: float, height: float, heading: float, answers: dict[str, str]) -> None:
        values = {name: IN_AIR[name][1][option][0] for name, option in answers.items() if name in IN_AIR}
        fdm = self.fdm
        fdm["ic/lat-geod-deg"], fdm["ic/long-gc-deg"] = airfield.geodetic(north, east)
        fdm["ic/h-agl-ft"] = (height + airfield.PARKED_AGL_M) / airfield.FT
        fdm["ic/psi-true-deg"] = heading
        fdm["ic/vc-kts"] = values["speed"]
        # JSBSim's initial conditions assume still air, so add the wind
        true, sink, pointing = fdm["ic/vt-fps"], -values["climb"] / 60, math.radians(heading)
        level = math.sqrt(max(0.0, true * true - sink * sink))
        fdm["ic/vn-fps"] = level * math.cos(pointing) + self.wind_fps[0]
        fdm["ic/ve-fps"] = level * math.sin(pointing) + self.wind_fps[1]
        fdm["ic/vd-fps"] = sink
        fdm["propulsion/set-running"] = -1
        if not fdm.run_ic():
            raise core.Unavailable("JSBSim rejected the initial conditions in the air")
        self.blow()
        self.control({"brake": 0.0, "throttle": 0.55, "elevator": 0.0, "flaps": values["flaps"]})
        fdm.run()  # step once so the first reading includes the wind
        self.sample = self.measure()

    def control(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            if name not in controls.ACTUATORS:
                raise ValueError(f"Unknown control {name!r}; choose from {list(controls.ACTUATORS)}")
            value = airfield.clamp(float(value), *controls.ACTUATORS[name])
            if name == "pedals":
                self.fdm["fcs/rudder-cmd-norm"] = -value  # the c172x rudder yaws the nose left for +1
                self.fdm["fcs/steer-cmd-norm"] = value
            elif name == "throttle":
                for prop in PROPERTIES[name]:
                    self.fdm[prop] = value * self.power
            else:
                for prop in PROPERTIES[name]:
                    self.fdm[prop] = value
            self.controls[name] = value

    def tick(self) -> dict:
        if self.autopilot and self.ticks % 3 == 0:
            self.control(self.autopilot.steer(s=self.instruments(), dt=3 / self.HZ))
        if not self.fdm.run():
            raise RuntimeError("JSBSim stopped running")
        self.ticks += 1
        self.sample = self.measure()
        return self.sample

    def measure(self) -> dict:
        fdm = self.fdm
        north, east = airfield.local(latitude=fdm["position/lat-geod-deg"], longitude=fdm["position/long-gc-deg"])
        return {
            "t": self.ticks / self.HZ,
            "north_m": north,
            "east_m": east,
            "altitude_m": fdm["position/h-sl-meters"],
            "agl_m": fdm["position/h-agl-ft"] * airfield.FT,
            "airspeed_kt": fdm["velocities/vc-kts"],
            "ground_speed_kt": fdm["velocities/vg-fps"] * airfield.FT / airfield.KT,
            "climb_fpm": fdm["velocities/h-dot-fps"] * 60,
            "bank_deg": fdm["attitude/phi-deg"],
            "pitch_deg": fdm["attitude/theta-deg"],
            "heading_deg": fdm["attitude/psi-deg"] % 360,
            "on_ground": any(fdm[f"gear/unit[{i}]/WOW"] > 0 for i in range(3)),
            "nose_gear_first": fdm["gear/unit[0]/WOW"] > 0,
            "struck": [
                name
                for i, name in ((3, "tail"), (4, "left wingtip"), (5, "right wingtip"))
                if fdm[f"contact/unit[{i}]/WOW"] > 0
            ],
        }

    def instruments(self) -> dict:
        fdm = self.fdm
        return {
            **self.sample,
            "track_deg": math.degrees(fdm["flight-path/psi-gt-rad"]) % 360,
            "roll_rate_dps": math.degrees(fdm["velocities/p-rad_sec"]),
            "pitch_rate_dps": math.degrees(fdm["velocities/q-rad_sec"]),
            "yaw_rate_dps": math.degrees(fdm["velocities/r-rad_sec"]),
            "sideslip_deg": fdm["aero/beta-deg"],
        }

    def stall_warning(self) -> bool:
        return self.fdm["systems/stall-warn-norm"] > 0

    def scene(self) -> dict:
        fdm = self.fdm
        return {
            "controls": {**self.controls, "rudder": -self.controls["pedals"]},
            "instruments": {
                "engine_rpm": fdm["propulsion/engine/engine-rpm"],
                "flap_deg": fdm["fcs/flap-pos-deg"],
                "elevator_deg": math.degrees(fdm["fcs/elevator-pos-rad"]),
            },
        }

    def metrics(self) -> dict:
        return {"fuel_used_kg": round((self.fuel_start - self.fdm["propulsion/total-fuel-lbs"]) * 0.45359237, 3)}

    def close(self) -> None:
        self.fdm = None

    def final_course(self, s: dict) -> float:
        return airfield.RUNWAY["heading_deg"] - airfield.clamp(
            0.3 * airfield.runway_frame(s["north_m"], s["east_m"])[1], -30, 30
        )

    def final_target(self, s: dict, direction: str) -> dict:
        along, right = airfield.runway_frame(s["north_m"], s["east_m"])
        return {
            "runway_threshold_ahead_m": int(round(airfield.THRESHOLD_NORTH_M - along, -1)),
            "centreline": airfield.offset_text(right),
            "direction": direction,
            "glide_path": airfield.vertical_text(-airfield.glide_error(along, s["agl_m"])),
            "approach_speed_kt": [round(knots) for knots in APPROACH["airspeed_kt"]],
        }


class Autopilot:
    """Holds a heading, and in the air a climb rate and airspeed"""

    def __init__(self, heading_deg: float):
        self.heading_deg = heading_deg
        self.climb_fpm: float | None = None
        self.speed_kt: float | None = None
        self.integral = {"bank": 0.0, "climb": 0.0, "pitch": 0.0, "yaw": 0.0, "speed": 0.0}

    def steer(self, s: dict, dt: float) -> dict:
        i = self.integral
        error = airfield.wrap(self.heading_deg - s["heading_deg"])
        if s["on_ground"]:
            return {
                "pedals": airfield.clamp(0.05 * error - 0.05 * s["yaw_rate_dps"], -0.6, 0.6),
                "aileron": airfield.clamp(-0.1 * s["bank_deg"] - 0.03 * s["roll_rate_dps"], -1, 1),
            }
        # bank less near the ground
        limit = airfield.clamp(2.0 + 0.7 * (s["agl_m"] - airfield.PARKED_AGL_M), 5.0, MAX_BANK_DEG)
        bank_error = airfield.clamp(HEADING_GAIN * error, -limit, limit) - s["bank_deg"]
        i["bank"] = airfield.clamp(i["bank"] + 0.01 * bank_error * dt, -0.3, 0.3)
        aileron = 0.03 * bank_error - 0.015 * s["roll_rate_dps"] + i["bank"]
        i["yaw"] = airfield.clamp(i["yaw"] - 0.06 * s["sideslip_deg"] * dt, -0.5, 0.5)
        rudder = -0.02 * s["sideslip_deg"] + 0.02 * s["yaw_rate_dps"] + i["yaw"]
        out = {"aileron": airfield.clamp(aileron, -1, 1), "pedals": -airfield.clamp(rudder, -1, 1)}
        if self.climb_fpm is not None:
            climb_error = (self.climb_fpm - s["climb_fpm"]) / 100
            i["climb"] = airfield.clamp(i["climb"] + 0.3 * climb_error * dt, -6, 6)
            # limit nose-up pitch below 60 kt to avoid a stall
            pitch = airfield.clamp(2.0 + 0.8 * climb_error + i["climb"], -10.0, min(12.0, s["airspeed_kt"] - 48.0))
            pitch_error = pitch - s["pitch_deg"]
            i["pitch"] = airfield.clamp(i["pitch"] - 0.01 * pitch_error * dt, -0.6, 0.6)
            out["elevator"] = airfield.clamp(-0.05 * pitch_error + 0.02 * s["pitch_rate_dps"] + i["pitch"], -1, 1)
        if self.speed_kt is not None:
            speed_error = self.speed_kt - s["airspeed_kt"]
            i["speed"] = airfield.clamp(i["speed"] + 0.03 * speed_error * dt, -0.5, 0.5)
            out["throttle"] = airfield.clamp(0.5 + 0.08 * speed_error + i["speed"], 0, 1)
        return out


class CessnaEvaluator(objectives.Evaluator):
    def took_off(self, s: dict, height: float) -> bool:
        moved = math.hypot(s["north_m"] - self.start["north_m"], s["east_m"] - self.start["east_m"])
        return (
            self.lifted_off
            and not s["on_ground"]
            and height >= objectives.TAKEOFF["height_m"]
            and s["airspeed_kt"] >= objectives.TAKEOFF["airspeed_kt"]
            and moved >= objectives.TAKEOFF["distance_m"]
        )

    def stable(self, s: dict, along: float, right: float) -> bool:
        gate = APPROACH
        before = airfield.THRESHOLD_NORTH_M - along
        return (
            gate["before_threshold_m"][0] <= before <= gate["before_threshold_m"][1]
            and abs(right) <= gate["off_centre_m"]
            and abs(airfield.wrap(s["heading_deg"] - airfield.RUNWAY["heading_deg"])) <= gate["heading_deg"]
            and gate["airspeed_kt"][0] <= s["airspeed_kt"] <= gate["airspeed_kt"][1]
            and gate["climb_fpm"][0] <= s["climb_fpm"] <= gate["climb_fpm"][1]
            and abs(s["bank_deg"]) <= gate["bank_deg"]
            and gate["height_m"][0] <= s["agl_m"] - airfield.PARKED_AGL_M <= gate["height_m"][1]
        )

    def problems(self, t: dict) -> list[str]:
        """Return what keeps a touchdown from counting as a landing"""
        found = []
        if t["sink_fpm"] > LANDING["sink_fpm"]:
            found.append(f"touchdown sink {t['sink_fpm']:.0f} fpm exceeds {LANDING['sink_fpm']:.0f}")
        if t["airspeed_kt"] > LANDING["airspeed_kt"]:
            found.append(f"touchdown at {t['airspeed_kt']:.0f} kt exceeds {LANDING['airspeed_kt']:.0f}")
        if abs(t["bank_deg"]) > LANDING["bank_deg"]:
            found.append(f"touchdown bank {abs(t['bank_deg']):.0f} deg exceeds {LANDING['bank_deg']:.0f}")
        if abs(t["heading_error_deg"]) > LANDING["heading_deg"]:
            found.append(f"touchdown {abs(t['heading_error_deg']):.0f} deg off the runway heading")
        if t["nose_first"]:
            found.append("the nose wheel touched down first")
        if not self.on_landing_area(t["along_m"], t["right_m"]):
            found.append(f"touchdown {t['along_m']:.0f} m along, {t['right_m']:.1f} m right is off the runway")
        return found

    def on_landing_area(self, along: float, right: float) -> bool:
        return (
            airfield.THRESHOLD_NORTH_M <= along <= airfield.RUNWAY["length_m"] / 2
            and abs(right) <= airfield.RUNWAY["width_m"] / 2 - LANDING["edge_margin_m"]
        )
