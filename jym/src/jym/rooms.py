from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Any

from jym import core, environments, runner, runs

if TYPE_CHECKING:
    from jym import web

PARTS = ("session", "scene", "ghosts", "decision", "pending", "live", "scenes", "decisions")
LEAVE_SECONDS = 30.0  # stop a person's run this long after their last tab closes


class Room:
    """What a tab shows: the stream, or one visitor's run. Tabs get only the parts that changed"""

    def __init__(self, jym: web.Jym, name: str):
        self.jym, self.name = jym, name
        self.lock = threading.Lock()
        self.session: Session | None = None
        self.touched = time.monotonic()
        self.viewers = 0  # open WebSockets
        self.hidden = 0  # background tabs, which don't keep the stream running
        parts = {"session": None, "scene": None, "ghosts": [], "decision": None, "pending": None, "live": None}
        self.state: dict[str, Any] = {**parts, "scenes": {}, "decisions": {}, "versions": dict.fromkeys(PARTS, 0)}
        self.encoded: dict[tuple, str] = {}

    def publish(self, owner: Session | None = None, **parts: Any) -> None:
        """Update parts of the state, ignoring replaced sessions"""
        with self.lock:
            if owner is not None and owner is not self.session:
                return
            versions = {part: count + (part in parts) for part, count in self.state["versions"].items()}
            self.state = {**self.state, **parts, "versions": versions}

    def merge(self, part: str, player: str, value: Any) -> None:
        with self.lock:
            self.state = {
                **self.state,
                part: {**self.state[part], player: value},
                "versions": {**self.state["versions"], part: self.state["versions"][part] + 1},
            }

    def encode(self, state: dict, parts: tuple[str, ...], here: int) -> str:
        """Encode these parts once for all tabs"""
        key = (tuple(state["versions"][part] for part in parts), parts, here)
        text = self.encoded.get(key)
        if text is None:
            text = json.dumps({**{part: state[part] for part in parts}, "here": here})
            if len(self.encoded) >= 16:
                self.encoded.clear()
            self.encoded[key] = text
        return text

    def watched(self) -> bool:
        return self.viewers - self.hidden > 0

    def running(self) -> bool:
        session = self.session
        return bool(session and session.thread.is_alive())

    def start(self, mode: str, task: core.Task, seed: int) -> dict:
        with self.lock:
            if self.running():
                raise ValueError("A run is already in progress; stop it first")
            self.session = session = Session(self, task, seed, mode)
        self.publish(session=session.info, scene=None, decision=None, pending=None, ghosts=self.jym.ghosts(task, seed))
        session.thread.start()
        threading.Thread(target=session.race_reference, daemon=True).start()
        return session.info

    def leave(self) -> None:
        """Stop the person's run unless a tab reconnects within LEAVE_SECONDS"""
        session = self.session
        if self.viewers or not session:
            return

        def check() -> None:
            if not self.viewers and self.session is session:
                session.stop.set()

        timer = threading.Timer(LEAVE_SECONDS, check)
        timer.daemon = True
        timer.start()


class Session:
    """A visitor's run in their own room"""

    def __init__(self, room: Room, task: core.Task, seed: int, mode: str):
        self.room, self.jym = room, room.jym
        self.task, self.seed, self.mode = task, seed, mode
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.pending: dict | None = None  # questions waiting for an answer
        self.answer: dict | None = None
        self.answered = threading.Event()
        self.controls: dict = {}  # controls changed since the last frame
        self.moved = time.monotonic()
        self.info = {
            "mode": mode,
            "env": task.env,
            "task": task.id,
            "seed": seed,
            "state": "starting",
            "run_id": None,
            "message": None,
            "result": None,
        }
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"session-{mode}")

    def publish(self, **parts: Any) -> None:
        self.room.publish(self, **parts)

    def status(self, state: str, message: str | None = None) -> None:
        self.info = {**self.info, "state": state, "message": message}
        self.publish(session=self.info)

    def run(self) -> None:
        try:
            log = runs.RunLog(self.jym.runs_dir)
            self.info = {**self.info, "run_id": log.run_id}
            self.status("running")
            environment = environments.create(self.task.env)
            if self.mode == "controls":
                result = runner.run_human(
                    environment=environment,
                    task=self.task,
                    seed=self.seed,
                    log=log,
                    stop=self.stop,
                    inputs=self.take_controls,
                    on_frame=self.show,
                )
            else:
                log.listener = self.heard
                result = runner.run_lockstep(
                    environment=environment,
                    policy=Person(self),
                    task=self.task,
                    seed=self.seed,
                    log=log,
                    stop=self.stop,
                )
            self.jym.library.add_run(log.path)
            self.info = {**self.info, "state": "ended", "message": result.reason, "result": result.model_dump()}
        except Exception as error:  # noqa: BLE001
            self.info = {**self.info, "state": "error", "message": runner.error_text(error)}
        finally:
            self.publish(session=self.info, pending=None)

    def show(self, scene: dict) -> None:
        self.publish(scene=scene)

    def race_reference(self) -> None:
        """Fly the reference player if needed, so there's a ghost to race"""
        if self.jym.fly_reference(self.task, self.seed):
            self.publish(ghosts=self.jym.ghosts(self.task, self.seed))

    def heard(self, event: dict) -> None:
        """Show the person's scenes, decisions and steps"""
        parts = {"scene": event["scene"]} if event.get("scene") else {}
        if event["type"] in ("decision", "step"):
            decision = paired(self.room.state["decision"], event)
            if decision:
                parts["decision"] = decision
        if parts:
            self.publish(**parts)

    def ask(self, state: dict, questions: dict) -> core.Reply:
        step = (self.pending or {}).get("step", 0) + 1
        with self.lock:
            self.pending, self.answer = {"step": step, "state": state, "questions": questions}, None
            self.answered.clear()
        self.publish(pending=self.pending)
        self.status("waiting for you")
        while not self.answered.wait(0.1):
            if self.stop.is_set():
                raise RuntimeError("stopped")
        self.status("running")
        return core.picked(**self.answer)

    def respond(self, step: int, choices: dict[str, str]) -> None:
        with self.lock:
            if not self.pending or self.pending["step"] != step or self.answered.is_set():
                raise ValueError("That question is no longer waiting for an answer")
            questions = self.pending["questions"]
            if set(choices) != set(questions) or any(choices[q] not in questions[q]["criteria"] for q in questions):
                raise ValueError("Answer every question with one of its options")
            self.answer = choices
            self.answered.set()

    def move(self, values: dict) -> None:
        with self.lock:
            self.controls |= values
            self.moved = time.monotonic()

    def take_controls(self) -> tuple[dict, float]:
        """Return the controls changed since the last frame, and seconds since the last change"""
        with self.lock:
            values, self.controls = self.controls, {}
            return values, time.monotonic() - self.moved


class Person:
    name = "human"

    def __init__(self, session: Session):
        self.session = session
        self.metadata = {"input": "questions"}

    def decide(self, state: dict, questions: dict) -> core.Reply:
        return self.session.ask(state, questions)


def paired(latest: list[dict] | None, event: dict) -> list[dict] | None:
    """Return a new decision, or the latest decision with the step that applied it"""
    if event["type"] == "decision":
        return [event]
    if latest and latest[0]["step"] == event["step"]:
        return [latest[0], event]
    return None
