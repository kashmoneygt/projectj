import random
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import torch
import typer

from flight_controller import curriculum, episode, evaluation, log, model


def expert_flight(job: episode.Job, delay: float) -> list[dict]:
    """Fly a job with the scripted pilot and return every decision, each labelled for when it applies"""
    flight = episode.Episode(job, delay)
    decisions = []
    while not flight.done:
        state, questions = flight.state, flight.questions
        _, label = flight.step(choices=None)
        decisions.append({"state": state, "questions": questions, "expert": label})
    return decisions


def example(pilot: model.Pilot, decision: dict) -> dict:
    prompt = pilot.prompt(decision["state"], decision["questions"])
    target = model.choice_indices(prompt=prompt, choices=decision["expert"])
    return {**pilot.encode(prompt, target), "target": target}


def imitation_loss(pilot: model.Pilot, batch: list[dict]) -> tuple[torch.Tensor, int]:
    """Return the cross-entropy of the scripted pilot's answers, and how many decisions the model got all right"""
    losses, right = [], 0
    for values, item in zip(pilot.log_probabilities(batch), batch, strict=True):
        target = torch.tensor(item["target"], device=values.device)
        losses.append(-values.gather(1, target[:, None]).sum())
        right += int((values.argmax(-1) == target).all())
    return torch.stack(losses).sum() / sum(len(item["target"]) for item in batch), right


def learn(
    pilot: model.Pilot,
    data: list[dict],
    epochs: float,
    lr: float,
    micro: int,
    accumulate: int,
    write_log: Callable[[dict], None],
) -> None:
    """Fit the labelled decisions with AdamW, warming up briefly and decaying to a tenth"""
    optimizer = torch.optim.AdamW(pilot.trainable(), lr=lr, weight_decay=0.0)
    rng = random.Random(len(data))
    batches: list[list[int]] = []
    while len(batches) * micro < len(data) * epochs:
        batches += model.by_start(items=data, size=micro, rng=rng)
    total = max(1, len(batches) // accumulate)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 20) * max(0.1, 1 - step / total)
    )
    pilot.model.train()
    started, seen, correct = time.time(), 0, 0
    for step in range(total):
        optimizer.zero_grad(set_to_none=True)
        chunk = batches[step * accumulate : (step + 1) * accumulate]
        size = sum(map(len, chunk))
        for batch in chunk:
            loss, right = imitation_loss(pilot, [data[i] for i in batch])
            (loss * len(batch) / size).backward()
            seen, correct = seen + len(batch), correct + right
        torch.nn.utils.clip_grad_norm_(pilot.trainable(), 1.0)
        optimizer.step()
        pilot.updated()
        schedule.step()
        if step % 25 == 0 or step == total - 1:
            accuracy = round(correct / max(1, seen), 3)
            write_log(
                {
                    "stage": "imitation",
                    "step": step,
                    "of": total,
                    "loss": round(loss.item(), 4),
                    "decisions_all_right": accuracy,
                    "seconds": round(time.time() - started),
                }
            )
            seen = correct = 0


def dagger(
    out: Annotated[Path, typer.Option(help="Output folder (adapters and log.jsonl)")] = Path("output/dagger"),
    init: Annotated[Path | None, typer.Option(help="Start from this adapter instead of a fresh LoRA")] = None,
    resume: Annotated[
        bool, typer.Option(help="Skip the first round of training, since --init already learnt the expert flights")
    ] = False,
    expert_episodes: int = 128,
    iterations: int = 6,
    student_episodes: int = 64,
    epochs: float = 1.0,
    limit: Annotated[int, typer.Option(help="Max decisions trained on per round, picked at random")] = 32000,
    lr: float = 2e-4,
    micro: Annotated[int, typer.Option(help="Decisions per forward pass (8 fit a 12 GB GPU)")] = 8,
    accumulate: int = 4,
    workers: Annotated[int, typer.Option(help="Processes flying the expert flights")] = 8,
    batch: Annotated[int, typer.Option(help="Student flights flown at once")] = 96,
    tasks: Annotated[str, typer.Option(help="Objectives and weights, e.g. circuit=3,course=1 (default: all)")] = "",
    handovers: Annotated[bool, typer.Option(help="Have the scripted pilot fly the start of some flights")] = True,
    takeovers: Annotated[bool, typer.Option(help="Give some student flights a few seconds of random answers")] = True,
    wind: Annotated[float, typer.Option(help="Maximum wind in knots (0 for each objective's default)")] = 12.0,
    eval_seeds: Annotated[str, typer.Option(help="Development seeds the final adapter flies, e.g. 0-9")] = "0-9",
    delay: Annotated[str, typer.Option(help="Seconds each answer takes to apply, or a range like 0.1-2.5")] = "0.1-2.5",
    student_cap: Annotated[int, typer.Option(help="Decisions a student flight may take before it is cut")] = 900,
    seed: int = 0,
) -> None:
    rng, write_log, pilot = random.Random(seed), log.logger(out), model.Pilot(init)
    experts = curriculum.Curriculum(
        tasks=curriculum.task_weights(tasks), delay=curriculum.span(delay), wind=wind, takeovers=False
    )
    students = curriculum.Curriculum(
        tasks=curriculum.task_weights(tasks),
        delay=curriculum.span(delay),
        wind=wind,
        takeovers=takeovers,
        handovers=handovers,
    )
    with ProcessPoolExecutor(workers) as pool:
        drawn = [experts.draw(rng) for _ in range(expert_episodes)]
        demonstrations = list(pool.map(expert_flight, [job for job, _, _ in drawn], [delay for _, delay, _ in drawn]))
    data = [example(pilot, decision) for flight in demonstrations for decision in flight]
    write_log({"stage": "expert", "episodes": len(demonstrations), "decisions": len(data)})
    for round_ in range(iterations + 1):
        random.Random(round_).shuffle(data)
        if data and not (resume and round_ == 0):
            learn(pilot, data[:limit], epochs, lr, micro, accumulate, write_log)
        pilot.save(out / f"round-{round_}")
        if round_ == iterations:
            break
        drawn = [students.draw(rng) for _ in range(student_episodes)]
        timings = [(delay, interval) for _, delay, interval in drawn]
        decisions, results = evaluation.fly(
            pilot=pilot, jobs=[job for job, _, _ in drawn], timings=timings, batch=batch, cap=student_cap
        )
        write_log(
            {"stage": "student", "round": round_, "decisions": len(decisions), "results": evaluation.summarize(results)}
        )
        data += [example(pilot, decision) for decision in decisions]
    pilot.save(out / "final")
    middle = sum(curriculum.span(delay)) / 2
    results = evaluation.check(pilot=pilot, tasks=students.tasks, seeds=curriculum.seed_list(eval_seeds), delay=middle)
    write_log({"stage": "evaluate", "delay": middle, "results": results})
