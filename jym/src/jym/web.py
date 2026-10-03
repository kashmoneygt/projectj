import asyncio
import dataclasses
import json
import os
import re
import threading
import time
import tomllib
import uuid
from collections.abc import Sequence
from concurrent import futures
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import RequestResponseEndpoint

from jym import core, environments, library, live, paths, players, rooms, runner, runs

UI = paths.ROOT / "ui" / "dist"
LIBRARY = paths.ROOT / "library"
SEND_SECONDS = 1 / 30  # how often tabs get updates
KEEPALIVE_SECONDS = 25.0  # message tabs at least this often so proxies keep sockets open
CLIENT = re.compile(r"[A-Za-z0-9-]{8,64}")
IDLE_SECONDS = 600.0  # close an unused room after this long, if another visitor needs one


@dataclasses.dataclass
class Settings:
    """Server settings, like deploy/jym.toml"""

    players: list[str]  # who races in every heat
    recorded: list[str] = dataclasses.field(default_factory=list)  # who races only as replays
    stream: list[str] = dataclasses.field(default_factory=list)  # the objectives heats pick from (default: all)
    heat_minutes: dict[str, float] = dataclasses.field(default_factory=dict)  # time limits, except on leaderboard seeds
    seed: int | None = None  # seed for every heat (default: a new one each heat)
    play: str | None = None  # default objective in Play
    rooms: int = 16  # max visitors playing at once
    hosted: bool = False  # served publicly by `jym host`, not locally by `jym ui`

    @classmethod
    def load(cls, path: Path | str) -> "Settings":
        with open(path, "rb") as file:
            values = tomllib.load(file)
        allowed = {field.name for field in dataclasses.fields(cls)} - {"hosted"}
        if unknown := sorted(set(values) - allowed):
            raise ValueError(f"{path}: unknown settings {unknown}; the settings are {sorted(allowed)}")
        settings = cls(**values, hosted=True)
        try:
            settings.check()
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from None
        return settings

    def check(self) -> None:
        ids = [task.id for task in environments.all_tasks()]
        for task in [*([self.play] if self.play else []), *self.stream, *self.heat_minutes]:
            if task not in ids:
                raise ValueError(f"unknown objective {task!r}; choose from {ids}")
        if any(minutes <= 0 for minutes in self.heat_minutes.values()):
            raise ValueError("a heat's time limit must be positive")
        for name in [*self.players, *self.recorded]:
            players.player(name, self.play_objective().env)

    def objectives(self) -> list[core.Task]:
        return [task for task in environments.all_tasks() if not self.stream or task.id in self.stream]

    def play_objective(self) -> core.Task:
        found = (task for task in environments.all_tasks() if task.id == self.play)
        return next(found, None) or environments.get_task(environments.default())


class StartRequest(BaseModel):
    mode: Literal["controls", "questions"]  # fly with controls, or answer a model's questions
    env: str
    task: str
    seed: int = Field(0, ge=0, le=2**31 - 1)


class AnswerRequest(BaseModel):
    step: int
    choices: dict[str, str]


class NextRequest(BaseModel):
    tasks: list[str]  # the objectives the stream's next heat may fly


class Jym:
    """State shared by a jym server's rooms: players, models, runs and the stream"""

    def __init__(self, settings: Settings, runs_dir: Path | str, library_folders: Sequence[Path] = (LIBRARY,)):
        self.settings = settings
        self.runs_dir = Path(runs_dir)
        self.library = library.Library(self.runs_dir, library_folders, self.runs_dir / ".replays")
        self.play = settings.play_objective()
        stream = [players.player(name, self.play.env) for name in settings.players]
        recorded = [players.player(name, self.play.env) for name in settings.recorded]
        self.registry = players.known(self.play.env) | {player.id: player for player in stream + recorded}
        self.lock = threading.Lock()
        self.models: dict[str, futures.Future] = {}
        self.rooms: dict[str, rooms.Room] = {}  # visitor rooms by tab id
        self.live = rooms.Room(jym=self, name="live")
        self.director = live.Director(
            jym=self,
            room=self.live,
            players=[player.id for player in stream],
            recorded=[player.id for player in recorded],
        )
        self.visitors: dict[str, int] = {}  # open tabs per visitor
        self.presence = 0  # bumped when the visitor count changes
        self.reference_lock = threading.Lock()

    def title(self, player: str) -> str:
        known = self.registry.get(player)
        return known.title if known else "People" if player == library.HUMAN else player

    def role(self, player: str) -> str | None:
        if known := self.registry.get(player):
            return known.role
        return "person" if player == library.HUMAN else None

    def room(self, client: str | None) -> rooms.Room:
        if not client or not CLIENT.fullmatch(client):
            raise ValueError("Bad client id")
        with self.lock:
            room = self.rooms.get(client)
            if room is None:
                for name, idle in list(self.rooms.items()):
                    if not idle.running() and time.monotonic() - idle.touched > IDLE_SECONDS:
                        del self.rooms[name]
                if len(self.rooms) >= self.settings.rooms:
                    raise ValueError("Every seat is taken; watch the stream and try again in a few minutes")
                room = self.rooms[client] = rooms.Room(jym=self, name=client)
            room.touched = time.monotonic()
            return room

    def arrive(self, visitor: str) -> None:
        with self.lock:
            self.visitors[visitor] = self.visitors.get(visitor, 0) + 1
            self.presence += self.visitors[visitor] == 1

    def depart(self, visitor: str) -> None:
        with self.lock:
            self.visitors[visitor] -= 1
            if not self.visitors[visitor]:
                del self.visitors[visitor]
                self.presence += 1

    def model(self, name: str) -> futures.Future:
        """Load a model in the background, retrying if the last attempt failed"""
        with self.lock:
            future = self.models.get(name)
            if future is None or (future.done() and future.exception() is not None):
                future = self.models[name] = futures.Future()
                threading.Thread(target=self.load, args=(name, future), daemon=True).start()
            return future

    def load(self, name: str, future: futures.Future) -> None:
        """Load a model and cache all its prompt starts"""
        try:
            policy = self.registry[name].make(None)
            if warm := getattr(policy, "warm", None):
                for spec in environments.ENVIRONMENTS.values():
                    for task in spec.tasks:
                        for questions in spec.question_sets(task.id):
                            warm(task.goal, questions)
            future.set_result(policy)
        except Exception as error:  # noqa: BLE001
            future.set_exception(error)

    def loading(self, name: str) -> bool:
        future = self.models.get(name)
        return future is not None and not future.done()

    def loaded(self) -> bool:
        """Return whether the stream's models have loaded"""
        with self.lock:
            wanted = [self.models.get(name) for name in self.director.players if not self.registry[name].per_run]
        return all(future and future.done() and not future.exception() for future in wanted)

    def served(self) -> dict[str, str]:
        """Return the model each player serves, which its runs must match to count"""
        with self.lock:
            loaded = {name: future for name, future in self.models.items() if future.done() and not future.exception()}
        known = {name: player.model for name, player in self.registry.items() if player.model}
        found = {name: getattr(future.result(), "metadata", {}).get("model") for name, future in loaded.items()}
        return known | {name: model for name, model in found.items() if model}

    def ghosts(self, task: core.Task, seed: int) -> list[dict]:
        return [ghost | {"title": self.title(ghost["player"])} for ghost in self.library.ghosts(task, seed)]

    def fly_reference(self, task: core.Task, seed: int) -> bool:
        """Fly the reference player if it has no run of this objective and seed, and return whether it flew"""
        with self.reference_lock:  # one at a time, to leave CPU for the models
            if self.library.finished(task=task, seed=seed, player="reference"):
                return False
            environment = environments.create(task.env)
            log = runs.RunLog(self.runs_dir)
            runner.run_lockstep(environment=environment, policy=environment.reference(), task=task, seed=seed, log=log)
            self.library.add_run(log.path)
            return True

    def catalog(self) -> dict:
        return {
            "play": {"env": self.play.env, "task": self.play.id},
            "hosted": self.settings.hosted,
            "environments": [
                {
                    "id": env,
                    "title": spec.title,
                    "about": spec.about,
                    **spec.catalog,
                    "tasks": [task.model_dump() for task in spec.tasks],
                }
                for env, spec in environments.ENVIRONMENTS.items()
            ],
            "players": [
                {key: getattr(player, key) for key in ("id", "title", "role", "trained", "deployed")}
                for player in self.registry.values()
            ],
            "leaderboard_seeds": list(library.LEADERBOARD_SEEDS),
            "stream": [{"env": task.env, "task": task.id} for task in self.director.objectives],
        }


class Limiter:
    """Allow each client at most `rate` requests a minute"""

    def __init__(self, rate: int):
        self.rate, self.hits, self.lock = rate, {}, threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            recent = [t for t in self.hits.get(key, []) if now - t < 60]
            if len(recent) >= self.rate:
                self.hits[key] = recent
                return False
            self.hits[key] = [*recent, now]
            if len(self.hits) > 10_000:
                self.hits.clear()
            return True


def create_app(jym: Jym, hosts: set[str], secure: bool = False) -> FastAPI:
    """Build the app that serves the API and the page to these hosts ("*" for any)"""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    limiter = Limiter(rate=120)

    def served_host(host: str | None) -> bool:
        return "*" in hosts or host in hosts

    def own_origin(host: str | None, origin: str | None) -> bool:
        """Return whether a request comes from our page or no page"""
        if origin is None:
            return True
        schemes = ("https",) if secure else ("https", "http")
        return any(origin == f"{scheme}://{host}" for scheme in schemes)

    def client_of(request: Request) -> str | None:
        return request.headers.get("x-jym-client")

    def caller(request: Request) -> str:
        # azure's ingress appends the caller's address last, so it can't be forged
        forwarded = request.headers.get("x-forwarded-for", "").rpartition(",")[2].strip()
        return forwarded or (request.client.host if request.client else "?")

    def session_of(request: Request) -> rooms.Session | None:
        room = jym.rooms.get(client_of(request) or "")
        return room.session if room else None

    @app.middleware("http")
    async def guard(request: Request, call_next: RequestResponseEndpoint):
        host = request.headers.get("host")
        # health checks may use the container's own address as the host
        if not served_host(host) and request.url.path != "/api/health":
            return PlainTextResponse("Host not allowed", status_code=400)
        if request.method != "GET":
            if not own_origin(host, request.headers.get("origin")):
                return PlainTextResponse("Cross-origin request refused", status_code=403)
            if not limiter.allow(caller(request)):
                return PlainTextResponse("Too many requests; slow down", status_code=429)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    @app.exception_handler(ValueError)
    async def refused(_: Request, error: ValueError):
        return JSONResponse({"detail": str(error)}, status_code=409)

    def check_env(env: str) -> None:
        if env not in environments.ENVIRONMENTS:
            raise ValueError(f"{env} is not on this jym; choose from {list(environments.ENVIRONMENTS)}")

    @app.get("/api/catalog")
    def catalog():
        return jym.catalog()

    @app.get("/api/health")
    def health(ready: bool = False):
        """With ?ready=1, return 503 until the stream's models have loaded"""
        body = {"ok": True, "hosted": jym.settings.hosted, "rooms": len(jym.rooms), "ready": jym.loaded()}
        return JSONResponse(body, status_code=503) if ready and not body["ready"] else body

    @app.post("/api/start")
    def start(request: StartRequest, raw: Request):
        if jym.settings.hosted and request.mode == "questions":
            raise ValueError("Questions mode is only available when you run jym locally")
        task = environments.get_task(request.env, request.task)
        return jym.room(client_of(raw)).start(request.mode, task, request.seed)

    @app.post("/api/stop")
    def stop(raw: Request):
        if session := session_of(raw):
            session.stop.set()
        return {"stopped": True}

    @app.post("/api/answer")
    def answer(request: AnswerRequest, raw: Request):
        session = session_of(raw)
        if not session or session.mode != "questions":
            raise ValueError("No question is waiting for an answer")
        session.respond(request.step, request.choices)
        return {"accepted": True}

    @app.post("/api/next")
    def queue(request: NextRequest):
        jym.director.queue(request.tasks)
        return {"queued": True}

    @app.get("/api/runs")
    def list_runs():
        return runs.recent_runs(jym.runs_dir)[:200]

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        if not runs.RUN_ID.fullmatch(run_id):
            raise HTTPException(404, f"Run {run_id} not found")
        folder = jym.runs_dir
        if not (folder / run_id / "manifest.json").is_file():  # a library record, replayed on first request
            try:
                folder = jym.library.replayed(run_id)
            except Exception as error:  # noqa: BLE001
                raise HTTPException(404, f"Run {run_id} could not be replayed: {runner.error_text(error)}") from None
            if folder is None:
                raise HTTPException(404, f"Run {run_id} not found")
        return runs.load_run(run_id=run_id, runs_dir=folder)

    @app.get("/api/leaderboard")
    def leaderboard(env: str = environments.default()):
        check_env(env)
        board = jym.library.leaderboard(env=env, title=jym.title, models=jym.served())
        for task in board["tasks"]:
            for row in task["rows"]:
                row["role"] = jym.role(row["player"])
        return board

    @app.get("/api/ghosts")
    def ghosts(env: str, task: str, seed: int = 0):
        return jym.ghosts(environments.get_task(env, task), seed)

    @app.websocket("/ws")
    async def socket(websocket: WebSocket):
        host, params = websocket.headers.get("host"), websocket.query_params
        if not served_host(host) or not own_origin(host, websocket.headers.get("origin") or ""):
            await websocket.close(code=1008)
            return
        try:
            room = jym.live if params.get("room") == "live" else jym.room(params.get("client"))
        except ValueError:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        with room.lock:
            room.viewers += 1
        # tabs in one browser share a visitor id, so "N here" counts people
        visitor = params.get("visitor") or params.get("client") or ""
        visitor = visitor if CLIENT.fullmatch(visitor) else uuid.uuid4().hex
        jym.arrive(visitor)

        async def send():
            seen, presence, sent = dict.fromkeys(rooms.PARTS, -1), -1, 0.0
            while True:
                state = room.state
                changed = tuple(part for part in rooms.PARTS if state["versions"][part] != seen[part])
                if changed or jym.presence != presence or time.monotonic() - sent > KEEPALIVE_SECONDS:
                    seen, presence, sent = state["versions"], jym.presence, time.monotonic()
                    await websocket.send_text(room.encode(state, changed, len(jym.visitors)))
                room.touched = time.monotonic()
                await asyncio.sleep(SEND_SECONDS)

        sender = asyncio.create_task(send())
        hidden = False
        try:
            while True:
                message = json.loads(await websocket.receive_text())
                session = room.session
                if message.get("type") == "controls" and session and session.mode == "controls":
                    session.move(message.get("values") or {})
                elif message.get("type") == "visibility" and bool(message.get("visible")) == hidden:
                    hidden = not hidden
                    with room.lock:
                        room.hidden += 1 if hidden else -1
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            sender.cancel()
            with room.lock:
                room.viewers -= 1
                room.hidden -= hidden
            jym.depart(visitor)
            room.leave()

    if (UI / "index.html").is_file():
        app.mount("/", StaticFiles(directory=UI, html=True), name="ui")
    else:

        @app.get("/")
        def unbuilt():
            message = "The page is not built; run: npm --prefix ui ci && npm --prefix ui run build"
            return PlainTextResponse(message, status_code=503)

    return app


def serve(settings: Settings, port: int, runs_dir: Path | str) -> None:
    """Serve jym publicly to JYM_HOSTS when hosted, or else locally"""
    jym = Jym(settings, runs_dir)
    jym.director.start()
    if settings.hosted:
        hosts = {host.strip() for host in os.environ.get("JYM_HOSTS", "*").split(",") if host.strip()}
        app = create_app(jym=jym, hosts=hosts, secure=os.environ.get("JYM_INSECURE") != "1")
        print(f"jym on port {port}, answering to {', '.join(sorted(hosts))}", flush=True)
    else:
        app = create_app(jym=jym, hosts={f"{name}:{port}" for name in ("127.0.0.1", "localhost", "[::1]")})
        print(f"jym: http://127.0.0.1:{port}/", flush=True)
    bind = "0.0.0.0" if settings.hosted else "127.0.0.1"  # a hosted jym is behind the platform's ingress
    uvicorn.run(app, host=bind, port=port, log_level="warning", forwarded_allow_ips="*", ws_ping_timeout=60)
