from __future__ import annotations

import dataclasses
import random
import threading
import time
import uuid
from typing import TYPE_CHECKING

from jym import core, environments, library, rooms, runner, runs

if TYPE_CHECKING:
    from jym import web

START_SECONDS = 4.0  # show the lineup this long before a heat
INTERMISSION_SECONDS = 8.0  # show results this long after a heat
LOAD_SECONDS = 600.0  # max wait for the stream's models to load
GRACE_SECONDS = 45.0  # end a heat this long after its first finisher (except on leaderboard seeds)
UNWATCHED_SECONDS = 60.0  # a hosted jym ends a heat nobody has watched for this long


@dataclasses.dataclass
class Heat:
    """A race of several players on one objective and seed"""

    env: str
    task: str
    seed: int
    players: list[str]
    id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float | None = None  # wall-clock start time
    ended: float | None = None
    entries: dict[str, dict] = dataclasses.field(default_factory=dict)  # each player's progress and result

    def public(self, jym: web.Jym) -> dict:
        return {
            "id": self.id,
            "env": self.env,
            "task": self.task,
            "seed": self.seed,
            "started": self.started,
            "ended": self.ended,
            "players": [
                {
                    "player": player,
                    "title": jym.title(player),
                    "role": jym.role(player),
                    **self.entries.get(player, {"state": "waiting"}),
                }
                for player in self.players
            ],
        }


class Director:
    """Runs the stream: one heat after another"""

    def __init__(self, jym: web.Jym, room: rooms.Room, players: list[str], recorded: list[str]):
        self.jym, self.room = jym, room
        self.players = players
        self.recorded = [player for player in recorded if player not in players]  # raced only as replays
        self.objectives = jym.settings.objectives()
        self.seed = jym.settings.seed
        self.limits = {task: minutes * 60 for task, minutes in jym.settings.heat_minutes.items()}
        self.lock = threading.Lock()
        self.current: Heat | None = None
        self.upcoming: Heat | None = None
        self.stopped = threading.Event()
        self.cut = threading.Event()
        self.rng = random.Random()
        self.announced = 0.0  # last time decision counts were published
        self.thread = threading.Thread(target=self.run, daemon=True, name="director")

    def watched(self) -> bool:
        """A hosted jym only runs heats while someone watches"""
        return not self.jym.settings.hosted or self.room.watched()

    def status(self) -> dict:
        with self.lock:
            current, upcoming = self.current, self.upcoming
        return {
            "heat": current.public(self.jym) if current else None,
            "next": upcoming.public(self.jym) if upcoming else None,
            "roster": self.players,
            "loading": [player for player in self.players if self.jym.loading(player)],
            "now": time.time(),
        }

    def publish(self) -> None:
        self.room.publish(live=self.status())

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stopped.set()
        self.cut.set()

    def run(self) -> None:
        self.load()
        while not self.stopped.is_set():
            heat = self.next_heat() if self.watched() else None
            if heat is None:
                self.stopped.wait(1.0)
                continue
            try:
                self.play(heat)
            except Exception as error:  # noqa: BLE001
                print(f"Heat {heat.id} failed: {runner.error_text(error)}", flush=True)
            heat.ended = time.time()
            self.publish()
            self.stopped.wait(INTERMISSION_SECONDS)
            with self.lock:
                self.current = None
            self.room.publish(scenes={}, decisions={}, live=self.status())

    def load(self) -> None:
        deadline = time.monotonic() + LOAD_SECONDS
        for player in self.players:
            if not self.jym.registry[player].per_run:
                future = self.jym.model(player)
                self.publish()
                while not future.done() and time.monotonic() < deadline and not self.stopped.wait(0.5):
                    pass
        self.publish()

    def next_heat(self) -> Heat | None:
        """Return the announced next heat, and pick the one after"""
        heat = self.upcoming or self.pick()
        if heat:
            task = environments.get_task(heat.env, heat.task)
            heat.players += [player for player in self.recorded if self.recording(task, heat.seed, player)]
        following = self.pick(after=heat) if heat else None
        with self.lock:
            self.current, self.upcoming = heat, following
        return heat

    def queue(self, tasks: list[str]) -> None:
        """Make the next heat fly one of these objectives"""
        heat = self.pick(after=self.current, among=tasks)
        if heat is None:
            raise ValueError("The stream has no heat of those objectives")
        with self.lock:
            self.upcoming = heat
        self.publish()

    def pick(self, after: Heat | None = None, among: list[str] | None = None) -> Heat | None:
        """Pick a random objective, preferring a leaderboard seed some player hasn't flown"""
        players = list(self.players)
        objectives = [task for task in self.objectives if among is None or task.id in among]
        if not players or not objectives:
            return None
        if self.seed is not None:
            task = self.rng.choice(objectives)
            return Heat(task.env, task.id, self.seed, players)
        # with recorded players, heats only fly the leaderboard seeds they have all flown
        heats = [
            (task, seed)
            for task in objectives
            for seed in library.LEADERBOARD_SEEDS
            if all(self.recording(task, seed, player) for player in self.recorded)
            and not (after and (after.env, after.task, after.seed) == (task.env, task.id, seed))
        ]
        missing = [(task, seed) for task, seed in heats if any(not self.recording(task, seed, p) for p in players)]
        if missing:
            task, seed = self.rng.choice(missing)
        elif not self.recorded:
            task, seed = self.rng.choice(objectives), self.rng.randrange(100, 1_000_000)
        elif heats:
            task, seed = self.rng.choice(heats)
        else:
            return None
        return Heat(task.env, task.id, seed, players)

    def recording(self, task: core.Task, seed: int, player: str) -> dict | None:
        return self.jym.library.latest(task=task, seed=seed, player=player, model=self.jym.served().get(player))

    def play(self, heat: Heat) -> None:
        """Race every player at once, replaying runs they've already flown"""
        task = environments.get_task(heat.env, heat.task)
        self.cut.clear()
        heat.started = time.time() + START_SECONDS
        threads = []
        for player in heat.players:
            if found := self.recording(task, heat.seed, player):
                fields = ("run_id", "status", "score", "sim_seconds")
                heat.entries[player] = {"state": "recorded", **{key: found[key] for key in fields}}
                continue
            heat.entries[player] = {"state": "starting"}
            threads.append(threading.Thread(target=self.race, args=(heat, task, player), daemon=True))
            threads[-1].start()
        self.publish()
        # replays play until the last one to succeed has finished
        succeeded = [
            entry["sim_seconds"] or 0.0
            for entry in heat.entries.values()
            if entry["state"] == "recorded" and entry["status"] == "success"
        ]
        until = heat.started + max(succeeded, default=0.0)
        seen = time.monotonic()
        while any(thread.is_alive() for thread in threads) or (time.time() < until and not self.cut.is_set()):
            self.enforce(heat)
            if self.watched():
                seen = time.monotonic()
            elif time.monotonic() - seen > UNWATCHED_SECONDS:
                self.cut.set()
            if self.stopped.wait(0.5):
                self.cut.set()
                break
        for thread in threads:
            thread.join(5)

    def enforce(self, heat: Heat) -> None:
        """End the heat soon after its first finisher or at its time limit (except on leaderboard seeds)"""
        if heat.seed in library.LEADERBOARD_SEEDS:
            return
        elapsed, limit = time.time() - heat.started, self.limits.get(heat.task)
        finished = [
            entry["sim_seconds"]
            for entry in list(heat.entries.values())
            if entry.get("status") == "success" and entry.get("sim_seconds") is not None
        ]
        first = min(finished, default=None)
        too_long = limit is not None and elapsed > limit
        stragglers = first is not None and elapsed > max(1.5 * first, first + GRACE_SECONDS)
        if too_long or stragglers:
            self.cut.set()

    def race(self, heat: Heat, task: core.Task, player: str) -> None:
        entry = heat.entries[player]
        try:
            environment = environments.create(heat.env)
            if self.jym.registry[player].per_run:
                policy = self.jym.registry[player].make(environment)
            else:
                entry["state"] = "loading"
                self.publish()
                policy = self.jym.model(player).result()
            log = runs.RunLog(self.jym.runs_dir)
            log.listener = lambda event: self.heard(heat, player, event)
            entry.update(state="running", run_id=log.run_id)
            self.publish()
            result = runner.run_realtime(
                environment=environment,
                policy=policy,
                task=task,
                seed=heat.seed,
                log=log,
                stop=self.cut,
                start_at=heat.started,
                cut_reason=library.CUT,
            )
            self.jym.library.add_run(log.path)
            sim_seconds = result.metrics["evaluation"].get("sim_seconds")
            entry.update(
                state="done", status=result.status, score=result.score, reason=result.reason, sim_seconds=sim_seconds
            )
        except Exception as error:  # noqa: BLE001
            entry.update(state="error", reason=runner.error_text(error))
        self.publish()

    def heard(self, heat: Heat, player: str, event: dict) -> None:
        """Show a player's scenes and decisions as its run logs them"""
        if event.get("scene"):
            self.room.merge("scenes", player, event["scene"])
        if event["type"] in ("decision", "step"):
            decision = rooms.paired(self.room.state["decisions"].get(player), event)
            if decision:
                self.room.merge("decisions", player, decision)
        if event["type"] == "decision":
            entry = heat.entries[player]
            entry["decisions"], entry["last_seconds"] = (
                event["step"],
                runner.own_seconds(seconds=event["seconds"], info=event),
            )
            if time.monotonic() - self.announced >= 0.5:
                self.announced = time.monotonic()
                self.publish()
