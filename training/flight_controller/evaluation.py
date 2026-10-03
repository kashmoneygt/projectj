import json
import math
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated

import typer

from flight_controller import curriculum, episode, model
from jym import players, runner
from jym.environments.flight import airfield, objectives

VARIANTS = ("standard", "windy", "hard-course", "varied", "takeover")
DELAYS = "0.2,1.5,3.0"  # a GPU or a fast API, a small CPU server, a slow computer


def variant_jobs(variant: str, tasks: Iterable[str], seeds: Sequence[int]) -> list[episode.Job]:
    """Return a variant's jobs for every task and seed, with settings drawn from the seed alone"""
    pairs = [(task, seed) for task in tasks for seed in seeds]
    if variant == "standard":
        return [episode.Job(task, seed) for task, seed in pairs]
    if variant == "windy":
        return [episode.Job(task=task, seed=seed, settings={"wind_max_kt": 15.0}) for task, seed in pairs]
    if variant == "hard-course":
        return [
            episode.Job(task=task, seed=seed, settings={"rings": hard_course(seed)})
            for task, seed in pairs
            if task.endswith("course")
        ]
    if variant == "varied":  # runway starts only, like the objectives
        draws = [(task, seed, random.Random(f"varied-{seed}")) for task, seed in pairs]
        return [
            episode.Job(task=task, seed=seed, settings=curriculum.varied_settings(task=task, rng=rng, air_starts=False))
            for task, seed, rng in draws
        ]
    if variant == "takeover":
        draws = [
            (task, seed, curriculum.takeover(task=task, rng=random.Random(f"takeover-{seed}"))) for task, seed in pairs
        ]
        return [
            episode.Job(task=task, seed=seed, takeover=seconds, takeover_at=at) for task, seed, (seconds, at) in draws
        ]
    raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")


def hard_course(seed: int) -> list[dict]:
    """Draw a random course with rings closer together, turns up to 90 degrees and heights of 60-500 m"""
    rng = random.Random(f"hard-course-{seed}")
    rings = [{**objectives.RINGS[0], "name": "ring 1"}]
    for number in range(2, objectives.COURSE["rings"] + 1):
        last = rings[-1]
        heading = (last["heading_deg"] + rng.uniform(-90, 90)) % 360
        metres, height = rng.uniform(1000, 1600), airfield.clamp(last["height_m"] + rng.uniform(-120, 120), 60, 500)
        rings.append(
            {
                "name": f"ring {number}",
                "north_m": round(last["north_m"] + metres * math.cos(math.radians(heading)), 1),
                "east_m": round(last["east_m"] + metres * math.sin(math.radians(heading)), 1),
                "height_m": round(height),
                "heading_deg": round(heading, 1),
            }
        )
    return rings


def fly(
    pilot: model.Pilot,
    jobs: list[episode.Job],
    timings: list[tuple[float, float]],
    batch: int,
    greedy: bool = True,
    label: bool = True,
    cap: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Fly each job at its (delay, interval) in batches, and return the labelled decisions and results"""
    decisions, results = [], []
    waiting, live = list(reversed(list(zip(jobs, timings, strict=True)))), []
    while live or waiting:
        while waiting and len(live) < batch:
            job, (delay, interval) = waiting.pop()
            live.append(episode.Episode(job, delay, interval, cap))
        flying = [e for e in live if not e.done]  # a handover can end a flight before the model flies it
        prompts = [pilot.prompt(e.state, e.questions) for e in flying]
        for flight, prompt, chosen in zip(flying, prompts, pilot.act(prompts, greedy), strict=True):
            state, questions = flight.state, flight.questions
            _, expert = flight.step(model.choice_names(prompt, chosen))
            if label:
                decisions.append({"state": state, "questions": questions, "expert": expert})
        for flight in [e for e in live if e.done]:
            results.append(flight.result())
            live.remove(flight)
    return decisions, results


def flights(
    pilot: model.Pilot | str | None,
    seeds: Sequence[int],
    tasks: Iterable[str],
    timings: list[tuple[float, float]],
    batch: int = 128,
    greedy: bool = True,
    variant: str = "standard",
) -> list[dict]:
    """Fly every task and seed at each (delay, interval) with a model, a GGUF path, or the scripted pilot if None"""
    runs = [(job, timing) for timing in timings for job in variant_jobs(variant, tasks, seeds)]
    if isinstance(pilot, model.Pilot):
        return fly(
            pilot=pilot,
            jobs=[job for job, _ in runs],
            timings=[timing for _, timing in runs],
            batch=batch,
            greedy=greedy,
            label=False,
        )[1]
    if pilot is None:
        choose = episode.Episode.expert_choices  # its answers apply late too, like any policy's
    else:
        policy = players.player(name=f"{pilot}@gpu", env="flight").make(None)

        def choose(flight: episode.Episode) -> dict[str, str]:
            answers, _ = policy.answer(flight.state, flight.questions)
            return {qid: reply["choice"] for qid, reply in answers.items()}

    results = []
    for job, (delay, interval) in runs:
        flight = episode.Episode(job, delay, interval)
        while not flight.done:
            flight.step(choose(flight))
        results.append(flight.result())
    return results


def check(
    pilot: model.Pilot,
    tasks: Iterable[str],
    seeds: Sequence[int],
    delay: float,
    variants: Iterable[str] = ("standard",),
) -> dict:
    """Fly the development seeds greedily in each variant and summarize each objective"""
    results = {}
    for variant in variants:
        flown = flights(
            pilot=pilot, seeds=seeds, tasks=tasks, timings=[(delay, runner.DECISION_SECONDS)], variant=variant
        )
        suffix = "" if variant in ("standard", "takeover") else f"~{variant}"  # takeover flights are already tagged
        results |= {name + suffix: row for name, row in summarize(flown).items()}
    return results


def summarize(results: list[dict]) -> dict:
    """Summarize each objective: success rate (95% Wilson interval) and mean score"""
    out = {}
    for name in sorted({r["task"] for r in results}):
        rows = [r for r in results if r["task"] == name]
        wins = sum(r["score"] == 100 for r in rows)
        out[name] = {
            "episodes": len(rows),
            "success": round(wins / len(rows), 3),
            "success_95": wilson(wins=wins, n=len(rows)),
            "mean_score": round(sum(r["score"] for r in rows) / len(rows), 1),
        }
    return out


def wilson(wins: int, n: int) -> list[float]:
    """Return the 95% Wilson interval of a success rate, which holds up even at 0/n and n/n"""
    z = 1.96
    rate, spread = wins / n, z * z / n
    middle = (rate + spread / 2) / (1 + spread)
    half = z * math.sqrt(rate * (1 - rate) / n + spread / (4 * n)) / (1 + spread)
    return [round(max(0.0, middle - half), 3), round(min(1.0, middle + half), 3)]


def evaluate(
    adapter: Annotated[Path | None, typer.Option(help="Adapter folder (default: the base model)")] = None,
    scripted: Annotated[bool, typer.Option(help="Fly the scripted pilot instead")] = False,
    gguf: Annotated[Path | None, typer.Option(help="Fly this GGUF on the GPU, as jym's gguf player does")] = None,
    seeds: Annotated[str, typer.Option(help="Comma-separated, or a range like 100-119")] = curriculum.TEST_SEEDS,
    tasks: Annotated[str, typer.Option(help="Comma-separated objectives (default: all eight)")] = "",
    delays: Annotated[str, typer.Option(help="Seconds before each answer applies, one run per value")] = DELAYS,
    intervals: Annotated[str, typer.Option(help="Minimum seconds between decisions, each with every delay")] = "0.5",
    sampled: Annotated[bool, typer.Option(help="Also fly the middle delay with sampled answers")] = True,
    variant: Annotated[str, typer.Option(help=f"One of {', '.join(VARIANTS)}")] = "standard",
    out: Annotated[Path | None, typer.Option(help="Write every flight here as JSON")] = None,
    batch: Annotated[int, typer.Option(help="Flights flown at once")] = 128,
) -> None:
    if scripted and adapter:
        raise typer.BadParameter("--scripted cannot be used with --adapter")
    if scripted and gguf:
        raise typer.BadParameter("--scripted cannot be used with --gguf")
    if gguf and adapter:
        raise typer.BadParameter("--gguf cannot be used with --adapter")
    pilot = None if scripted else str(gguf.resolve()) if gguf else model.Pilot(adapter)
    if isinstance(pilot, model.Pilot):
        pilot.merge()
    chosen = tuple(tasks.split(",")) if tasks else curriculum.EVAL_TASKS
    times = [float(delay) for delay in delays.split(",")]
    timings = [(delay, float(pace)) for pace in intervals.split(",") for delay in times]
    flown = flights(
        pilot=pilot, seeds=curriculum.seed_list(seeds), tasks=chosen, timings=timings, batch=batch, variant=variant
    )
    rows = [{**r, "decoding": "greedy"} for r in flown]
    runs = [(delay, pace, "greedy") for delay, pace in timings]
    if sampled and isinstance(pilot, model.Pilot):
        middle = (times[len(times) // 2], timings[0][1])
        flown = flights(
            pilot=pilot,
            seeds=curriculum.seed_list(seeds),
            tasks=chosen,
            timings=[middle],
            batch=batch,
            greedy=False,
            variant=variant,
        )
        rows += [{**r, "decoding": "sampled"} for r in flown]
        runs.append((*middle, "sampled"))
    who = "scripted pilot" if scripted else gguf or adapter or "base model"
    for delay, pace, decoding in runs:
        key = (round(delay, 3), round(pace, 3), decoding)
        done = [r for r in rows if (r["delay"], r["interval"], r["decoding"]) == key]
        print(f"\n{who}: {variant}, interval {pace:g} s, delay {delay:g} s, {decoding}")
        for task, summary in summarize(done).items():
            low, high = summary["success_95"]
            print(f"  {task:15s} {summary['success']:6.0%} [{low:.0%}-{high:.0%}]  mean {summary['mean_score']:5.1f}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        about = {"adapter": str(adapter) if adapter else None, "gguf": str(gguf) if gguf else None}
        out.write_text(json.dumps({**about, "scripted": scripted, "variant": variant, "flights": rows}))
