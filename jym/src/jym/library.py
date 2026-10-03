import json
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from jym import core, environments, runner, runs

FORMAT = 1
LEADERBOARD_SEEDS = (0, 1, 2)
CUT = "the heat ended"  # reason for runs cut short when a heat ends (not ranked)
GHOSTS = 6  # max ghosts per objective and seed
# runs that waited longer than this for a shared CPU aren't ranked
WAIT_TOLERANCE_SECONDS = 1.0
HUMAN = "human"  # player id for all people (the leaderboard keeps their best run)


def trajectory(env: str, events: list[dict]) -> list[list[float]]:
    pose = environments.ENVIRONMENTS[env].pose
    rows = {row[0]: row for row in (pose(event["scene"]) for event in events if event.get("scene"))}
    return [rows[t] for t in sorted(rows)]


def decisions(events: list[dict]) -> list[dict]:
    """Return a run's applied choices and watchdog holds, with their physics ticks"""
    seconds, applied = {}, []
    for event in events:
        if event["type"] == "decision":
            seconds[event["step"]] = event["seconds"]
        elif event["type"] == "step":
            action = dict(event["action"])
            tick, _ = action.pop("tick"), action.pop("sim_time")
            applied.append({"choices": action, "seconds": seconds[event["step"]], "tick": tick})
        elif event["type"] == "watchdog":
            applied.append({"hold": True, "tick": event["tick"]})
    return applied


def record(run_id: str, runs_dir: Path | str) -> dict:
    run = runs.load_run(run_id, runs_dir)
    manifest, result = run["manifest"], run["result"]
    if not result:
        raise ValueError(f"Run {run_id} has not finished")
    if manifest["mode"] != "realtime":
        raise ValueError(f"Run {run_id} was played in lockstep; only real-time runs go in the library")
    return {
        "format": FORMAT,
        "run_id": run_id,
        "final_tick": result["metrics"]["final_tick"],
        "player": manifest["policy"]["name"],
        "task": manifest["task"],
        "seed": manifest["seed"],
        "result": result,
        "recorded": {key: manifest[key] for key in ("created", "versions", "policy")},
        "decisions": decisions(run["events"]),
        "trajectory": trajectory(manifest["task"]["env"], run["events"]),
    }


def replay(entry: dict, runs_dir: Path | None = None) -> core.Evaluation:
    """Replay a record's decisions and score them, writing the run to runs_dir if given"""
    task = core.Task(**entry["task"])
    environment = environments.create(task.env)
    log = runs.RunLog(runs_dir, entry["run_id"]) if runs_dir else None
    if log:
        manifest = runs.manifest(
            run_id=log.run_id,
            environment=environment,
            policy=entry["player"],
            metadata={},
            task=task,
            seed=entry["seed"],
            mode="realtime",
        )
        log.write(name="manifest.json", data={**manifest, **entry["recorded"], "replayed": "from its library record"})
    try:
        environment.reset(task, entry["seed"])
        rate = environment.HZ
        every = max(1, round(runner.SCENE_SECONDS * rate))
        if log:
            log.event(kind="reset", state=environment.state(), scene=environment.scene())

        def reach(tick: int) -> None:
            while environment.ticks < tick:
                over = environment.advance(min(every, tick - environment.ticks) / rate)
                if log and (environment.ticks % every == 0 or environment.ticks >= tick or over):
                    log.event(kind="scene", scene=environment.scene())
                if over:
                    return

        for step, decision in enumerate(entry["decisions"], 1):
            if decision.get("hold"):
                reach(decision["tick"])
                environment.hold()
                if log:
                    log.event(kind="watchdog", sim_time=round(environment.ticks / rate, 3), tick=environment.ticks)
                continue
            seconds = decision.get("seconds") or 0.0
            if log:  # the state and questions when asked, for the decision panel
                reach(max(environment.ticks, decision["tick"] - round(seconds * rate)))
                asked_at = environment.ticks / rate
                state, questions = {"goal": task.goal, **environment.state()}, environment.questions()
            reach(decision["tick"])
            environment.apply(decision["choices"])
            if log:
                applied_at = round(decision["tick"] / rate, 3)
                answers = {qid: {"type": "choice", "choice": choice} for qid, choice in decision["choices"].items()}
                log.event(
                    kind="decision",
                    step=step,
                    state=state,
                    questions=questions,
                    answers=answers,
                    seconds=seconds,
                    asked_at=round(asked_at, 3),
                    applied_at=applied_at,
                )
                action = {**decision["choices"], "tick": decision["tick"], "sim_time": applied_at}
                log.event(kind="step", step=step, action=action)
        reach(entry["final_tick"])
        evaluation = environment.evaluate()
    finally:
        environment.close()
    if log:
        result = entry["result"]
        log.event(kind="evaluation", evaluation=evaluation)
        log.event(kind="end", status=result["status"], score=result["score"], reason=result.get("reason"))
        log.write(name="result.json", data=result)
    return evaluation


def verify(entry: dict) -> str | None:
    """Return how the record differs when replayed, or None"""
    try:
        evaluation = replay(entry)
    except Exception as error:  # noqa: BLE001
        return f"could not replay it: {runner.error_text(error)}"
    score, milestones = core.score(core.Task(**entry["task"]), evaluation), sorted(evaluation.milestones)
    recorded = entry["result"]
    if score != recorded["score"] or milestones != sorted(recorded["milestones"]):
        return f"replayed it scores {score} {milestones}, recorded {recorded['score']}"
    return None


def load_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_file(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")


class Library:
    """Finished runs and library records, for ghosts, heats and the leaderboard"""

    def __init__(self, runs_dir: Path | str, folders: Sequence[Path] = (), replays: Path | None = None):
        self.runs_dir = Path(runs_dir)
        self.folders = [Path(folder) for folder in folders]
        self.replays = replays  # folder for replayed records
        self.lock = threading.Lock()
        self.replay_lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        self.unfinished: set[str] = set()
        self.records: dict[str, Path] = {}  # record files by run id
        self.trajectories: dict[str, list] = {}
        self.refreshed = 0.0

    def refresh(self, force: bool = False) -> None:
        """Load new runs and records, at most every 10 s"""
        if not force and time.monotonic() - self.refreshed < 10:
            return
        self.refreshed = time.monotonic()
        with self.lock:
            known, files = set(self.rows), set(self.records.values())
        for path in self.runs_dir.iterdir() if self.runs_dir.is_dir() else []:
            if runs.RUN_ID.fullmatch(path.name) and path.name not in known:
                self.add_run(path)
        for folder in self.folders:
            for path in sorted(folder.rglob("*.json")):
                if path not in files:
                    self.add_record(path)

    def add_run(self, path: Path) -> None:
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            result_path = path / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        except (OSError, ValueError):
            return
        if manifest.get("format") != runs.FORMAT:
            return
        with self.lock:
            if result is None:
                self.unfinished.add(path.name)
                return
            self.unfinished.discard(path.name)
            self.rows[path.name] = row(
                run_id=path.name, manifest=manifest, mode=manifest["mode"], result=result, source="server"
            )

    def add_record(self, path: Path) -> None:
        try:
            entry = load_file(path)
        except (OSError, ValueError):
            return
        if entry.get("format") != FORMAT:
            return
        run_id = entry["run_id"]
        with self.lock:
            self.records[run_id] = path
            self.rows[run_id] = row(
                run_id=run_id,
                manifest=entry | entry["recorded"],
                mode="realtime",
                result=entry["result"],
                source="imported",
            )

    def finished(self, task: core.Task, **match: int | str) -> list[dict]:
        self.refresh()
        with self.lock:
            return [
                row
                for row in self.rows.values()
                if (row["env"], row["task"], row["version"]) == (task.env, task.id, task.version)
                and row["status"] not in ("error", "stopped")
                and all(row[key] == value for key, value in match.items())
            ]

    def latest(self, task: core.Task, seed: int, player: str, model: str | None) -> dict | None:
        """Return the player's latest complete run with the model it serves now"""
        found = [
            row
            for row in self.finished(task=task, seed=seed, player=player, mode="realtime")
            if row["reason"] != CUT and (model is None or row["model"] == model)
        ]
        return max(found, key=lambda row: row["created"], default=None)

    def ghosts(self, task: core.Task, seed: int) -> list[dict]:
        """Return each player's best run on this objective and seed, with its poses"""
        best: dict[str, dict] = {}
        for found in self.finished(task=task, seed=seed):
            current = best.get(found["player"])
            if current is None or (found["score"], -found["steps"]) > (current["score"], -current["steps"]):
                best[found["player"]] = found
        chosen = sorted(best.values(), key=lambda found: (-found["score"], found["steps"]))[:GHOSTS]
        fields = ("run_id", "player", "status", "score")
        ghosts = [{key: found[key] for key in fields} | {"trajectory": self.poses(found)} for found in chosen]
        return [ghost for ghost in ghosts if ghost["trajectory"]]

    def poses(self, found: dict) -> list:
        run_id = found["run_id"]
        if run_id not in self.trajectories:
            if found["source"] == "imported":
                points = load_file(self.records[run_id])["trajectory"]
            else:
                try:
                    points = trajectory(found["env"], runs.load_run(run_id, self.runs_dir)["events"])
                except (OSError, ValueError):
                    points = []
            if len(self.trajectories) > 256:
                self.trajectories.pop(next(iter(self.trajectories)))
            self.trajectories[run_id] = points
        return self.trajectories[run_id]

    def replayed(self, run_id: str) -> Path | None:
        """Return a record's replay folder, replaying it on first request"""
        self.refresh()
        with self.lock:
            path = self.records.get(run_id)
        if path is None or self.replays is None:
            return None
        with self.replay_lock:
            if not (self.replays / run_id / "result.json").is_file():
                scratch = self.replays / f".{uuid.uuid4().hex}"
                try:
                    replay(entry=load_file(path), runs_dir=scratch)
                    shutil.rmtree(self.replays / run_id, ignore_errors=True)
                    (scratch / run_id).rename(self.replays / run_id)
                finally:
                    shutil.rmtree(scratch, ignore_errors=True)
        return self.replays

    def leaderboard(self, env: str, title: Callable[[str], str], models: dict[str, str]) -> dict:
        """Rank each objective's players on the leaderboard seeds: successes, then mean score, then time per decision"""
        board = []
        for task in environments.tasks(env).values():
            picked: dict[tuple[str, int], dict] = {}
            for found in self.finished(task=task, mode="realtime"):
                counted = (
                    found["seed"] in LEADERBOARD_SEEDS
                    and found["reason"] != CUT
                    # only count the model a player serves now
                    and (found["player"] not in models or found["model"] == models[found["player"]])
                    and found["waited_seconds"] <= WAIT_TOLERANCE_SECONDS
                )
                current = picked.get((found["player"], found["seed"]))
                if counted and (current is None or better(found, current)):
                    picked[found["player"], found["seed"]] = found
            players: dict[str, list[dict]] = {}
            for (player, _), found in sorted(picked.items()):
                players.setdefault(player, []).append(found)
            rows = [standing(player, flown, title(player)) for player, flown in players.items()]
            rows.sort(key=lambda r: (not r["ranked"], -r["successes"], -r["mean_score"], r["median_seconds"] or 1e9))
            board.append({"task": task.id, "title": task.title, "version": task.version, "rows": rows})
        return {"env": env, "seeds": list(LEADERBOARD_SEEDS), "tasks": board}


def row(run_id: str, manifest: dict, mode: str, result: dict, source: str) -> dict:
    """Return the fields that ghosts, heats and the leaderboard match runs on"""
    task, metrics = manifest["task"], result.get("metrics") or {}
    return {
        "run_id": run_id,
        "source": source,
        "mode": mode,
        "env": task["env"],
        "task": task["id"],
        "version": task["version"],
        "seed": manifest["seed"],
        "player": result["policy"],
        "model": manifest["policy"]["metadata"].get("model"),
        "created": manifest["created"],
        "status": result["status"],
        "reason": result.get("reason"),
        "score": result["score"],
        "steps": result["steps"],
        "seconds_p50": metrics.get("decision_seconds_p50"),
        "waited_seconds": metrics.get("waited_seconds") or 0.0,
        "sim_seconds": (metrics.get("evaluation") or {}).get("sim_seconds"),
    }


def better(found: dict, current: dict) -> bool:
    """People keep their best run; everyone else keeps their latest"""
    if found["player"] == HUMAN:
        return (found["score"], -found["steps"]) > (current["score"], -current["steps"])
    return found["created"] > current["created"]


def standing(player: str, flown: list[dict], title: str) -> dict:
    timings = sorted(found["seconds_p50"] for found in flown if found["seconds_p50"] is not None)
    return {
        "player": player,
        "title": title,
        "seeds": [found["seed"] for found in flown],
        "ranked": len(flown) == len(LEADERBOARD_SEEDS),
        "successes": sum(found["status"] == "success" for found in flown),
        "mean_score": round(sum(found["score"] for found in flown) / len(flown), 1),
        "median_seconds": timings[len(timings) // 2] if timings else None,
        "imported": any(found["source"] == "imported" for found in flown),
        "runs": [{key: found[key] for key in ("seed", "run_id", "status", "score")} for found in flown],
    }


def export(run_ids: list[str], runs_dir: Path | str, folder: Path | str) -> list[Path]:
    """Write finished real-time runs as library records, after checking they replay exactly"""
    written = []
    for run_id in run_ids:
        entry = record(run_id, runs_dir)
        if problem := verify(entry):
            raise ValueError(f"{run_id}: {problem}")
        task, player = entry["task"], entry["player"].replace(":", "_")
        path = Path(folder) / task["env"] / f"{task['id']}-v{task['version']}-seed{entry['seed']}-{player}.json"
        write_file(path, entry)
        written.append(path)
    return written


def latest_runs(runs_dir: Path | str, env: str, players: list[str], seeds: tuple[int, ...]) -> dict[tuple, str]:
    """Return the latest complete run id of each player, objective and seed"""
    shelf, picked = Library(runs_dir), {}
    for task in environments.tasks(env).values():
        for found in shelf.finished(task=task, mode="realtime"):
            key = (found["player"], task.id, found["seed"])
            wanted = found["player"] in players and found["seed"] in seeds and found["reason"] != CUT
            if wanted and (key not in picked or found["created"] > picked[key]["created"]):
                picked[key] = found
    return {key: found["run_id"] for key, found in picked.items()}
