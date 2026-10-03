import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from jym import core, environments, library, paths, players, runner, runs, web

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,  # keeps API keys out of tracebacks
    help="jym: A platform to test AI models in various environments",
)
library_app = typer.Typer(no_args_is_help=True, help="Recorded runs, shown as ghosts, in heats and on the leaderboard.")
app.add_typer(library_app, name="library")

Env = Annotated[str, typer.Option(help=f"Environment: {', '.join(environments.ENVIRONMENTS)}")]
Runs = Annotated[Path, typer.Option(envvar="JYM_RUNS", help="Folder for run logs")]
RUNS = paths.ROOT / "runs"
DEFAULT_ENV = environments.default()
PLAYER = "reference, a model in models.toml (name@gpu on the GPU), or a GGUF file's path"


@app.callback()
def main() -> None:
    load_dotenv(paths.ROOT / ".env")


@app.command()
def ui(
    player_names: Annotated[
        str | None,
        typer.Option(
            "--players",
            help=f"Who races in Watch, comma-separated: {PLAYER} (default: the featured models and reference)",
        ),
    ] = None,
    task: Annotated[str | None, typer.Option(help="Keep the heats on this objective, and start Play on it")] = None,
    seed: Annotated[int | None, typer.Option(help="Keep the heats on this seed (default: a new one each heat)")] = None,
    port: int = 8765,
    runs_dir: Runs = RUNS,
) -> None:
    """Run jym locally: watch models race, or play yourself"""
    chosen = [*players.featured(), "reference"] if player_names is None else player_names.split(",")
    settings = web.Settings(
        players=[name for name in chosen if name], stream=[task] if task else [], seed=seed, play=task
    )
    try:
        settings.check()
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    build_ui()
    web.serve(settings, port, runs_dir)


@app.command()
def host(
    settings: Annotated[Path, typer.Option(help="The hosted jym's settings, like deploy/jym.toml")],
    port: Annotated[int, typer.Option(envvar="PORT")] = 8080,
    runs_dir: Runs = RUNS,
) -> None:
    """Serve jym publicly, with a live stream and a room for each visitor"""
    web.serve(web.Settings.load(settings), port, runs_dir)


@app.command()
def run(
    env: Env = DEFAULT_ENV,
    task: Annotated[str | None, typer.Option(help="Objective (default: the environment's own)")] = None,
    player: Annotated[str | None, typer.Option(help=f"Who plays: {PLAYER} (default: the first featured model)")] = None,
    seed: int = 0,
    runs_dir: Runs = RUNS,
) -> None:
    """Play one episode in lockstep and print its result"""
    try:
        spec = environments.get_task(env, task)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    environment = environments.create(env)
    result = runner.run_lockstep(
        environment=environment,
        policy=make(player or players.featured()[0], env, environment),
        task=spec,
        seed=seed,
        log=runs.RunLog(runs_dir),
    )
    typer.echo(result.model_dump_json(indent=2))
    if result.status == "error":
        raise typer.Exit(1)


@app.command()
def describe(env: str, json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Print the state and questions a model gets in ENV"""
    if env not in environments.ENVIRONMENTS:
        raise typer.BadParameter(f"choose from {list(environments.ENVIRONMENTS)}", param_hint="ENV")
    spec = environments.describe(env)
    typer.echo(json.dumps(spec, indent=2) if json_output else environments.markdown(spec))


@app.command()
def doctor() -> None:
    """Check which models and tools are ready on this machine"""
    typer.echo("Models:")
    for name, entry in players.models()["models"].items():
        typer.echo(f"  {name:<14} {players.readiness(name, entry)}")
    typer.echo(f"llama.cpp: {os.environ.get('LLAMA_SERVER', 'in Docker, started on first use')}")
    typer.echo("Tools: " + ", ".join(f"{tool} {'yes' if shutil.which(tool) else 'no'}" for tool in ("docker", "node")))


@library_app.command("record")
def library_record(
    player_names: Annotated[str, typer.Option("--players", help="Comma-separated players")] = "reference",
    env: Env = DEFAULT_ENV,
    tasks: Annotated[str | None, typer.Option(help="Comma-separated objectives (default: all)")] = None,
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds")] = "0,1,2",
    parallel: Annotated[int, typer.Option(min=1, help="Runs at once, each taking its simulated time")] = 4,
    out: Annotated[Path, typer.Option(help="Library folder")] = web.LIBRARY,
    runs_dir: Runs = RUNS,
) -> None:
    """Record missing real-time runs for each player, objective and seed, then export them"""
    chosen = tuple(int(seed) for seed in seeds.split(","))
    roster = {found.id: found for found in (resolve(name, env) for name in player_names.split(","))}
    specs = [spec for spec in environments.tasks(env).values() if not tasks or spec.id in tasks.split(",")]
    done = library.latest_runs(runs_dir=runs_dir, env=env, players=list(roster), seeds=chosen)
    missing = [
        (name, spec, seed) for name in roster for spec in specs for seed in chosen if (name, spec.id, seed) not in done
    ]

    def play(job: tuple[str, core.Task, int]) -> str:
        name, spec, seed = job
        environment = environments.create(env)
        result = runner.run_realtime(
            environment=environment,
            policy=roster[name].make(environment),
            task=spec,
            seed=seed,
            log=runs.RunLog(runs_dir),
        )
        return f"{name} {spec.id} seed {seed}: {result.status} {result.score}"

    with ThreadPoolExecutor(parallel) as pool:
        for line in pool.map(play, missing):
            typer.echo(line)
    found = library.latest_runs(runs_dir=runs_dir, env=env, players=list(roster), seeds=chosen)
    for path in library.export(run_ids=sorted(found.values()), runs_dir=runs_dir, folder=out):
        typer.echo(f"wrote {path}")


@library_app.command("export")
def library_export(
    run_ids: Annotated[list[str], typer.Argument(help="Finished real-time runs")],
    out: Annotated[Path, typer.Option(help="Library folder")] = web.LIBRARY,
    runs_dir: Runs = RUNS,
) -> None:
    """Check that finished runs replay exactly, then export them as library records"""
    for path in library.export(run_ids=run_ids, runs_dir=runs_dir, folder=out):
        typer.echo(f"wrote {path}")


def resolve(name: str, env: str) -> players.Player:
    try:
        return players.player(name, env)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--player") from None


def make(name: str, env: str, environment: core.Environment) -> core.Policy:
    try:
        return resolve(name, env).make(environment)
    except core.Unavailable as error:
        raise typer.BadParameter(str(error), param_hint="--player") from None


def build_ui() -> None:
    if (web.UI / "index.html").is_file():
        return
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("jym builds its page with npm on first run: install Node.js 22, then run it again")
    typer.echo("Building the page on first run: npm ci, then npm run build, in ui/")
    for command in (["ci"], ["run", "build"]):
        if subprocess.run([npm, *command], cwd=web.UI.parent, check=False).returncode:
            raise SystemExit(f"npm {' '.join(command)} failed in {web.UI.parent}")
