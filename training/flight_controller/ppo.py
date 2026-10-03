import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import torch
import typer

from flight_controller import curriculum, episode, evaluation, log, model, reward

# imitation and entropy weights; imitation decays to a tenth over `decay` iterations
IMITATION, ENTROPY = 0.2, 0.005


def collect(
    pilot: model.Pilot,
    live: list[episode.Episode],
    horizon: int,
    fresh: Callable[[], episode.Episode],
    finished: list[dict],
) -> list[dict]:
    """Fly horizon decisions of every live flight with sampled answers, replacing each flight that ends"""
    steps = []
    for _ in range(horizon):
        prompts = [pilot.prompt(e.state, e.questions) for e in live]
        picks = pilot.act(prompts=prompts, greedy=False)
        for slot, (flight, prompt, chosen) in enumerate(zip(live, prompts, picks, strict=True)):
            record = {**pilot.encode(prompt, chosen), "env": slot, "chosen": chosen, "features": flight.features()}
            record["reward"], expert = flight.step(model.choice_names(prompt, chosen))
            record |= {"seconds": flight.seconds, "done": flight.done, "truncated": flight.truncated}
            record["expert"] = model.choice_indices(prompt=prompt, choices=expert)
            if flight.done:
                record["final"] = flight.features()
                finished.append({"task": flight.name, "seed": flight.seed, "score": flight.score()})
                live[slot] = fresh()
            steps.append(record)
    return steps


def advantages(
    steps: list[dict], values: list[float], tails: list[float], finals: dict[int, float], envs: int
) -> tuple[list[float], list[float]]:
    """Return GAE advantages and returns per flight slot, bootstrapping cut-short flights from the critic"""
    out, returns = [0.0] * len(steps), [0.0] * len(steps)
    for slot in range(envs):
        running, next_value = 0.0, tails[slot]
        for i in reversed([i for i, r in enumerate(steps) if r["env"] == slot]):
            if steps[i]["done"]:
                next_value, running = finals.get(i, 0.0), 0.0
            gamma = reward.REWARD.discount(steps[i]["seconds"])
            lam = reward.REWARD.discount(seconds=steps[i]["seconds"], per_half_second=reward.REWARD.lam)
            delta = steps[i]["reward"] + gamma * next_value - values[i]
            running = delta + gamma * lam * running
            out[i], returns[i] = running, running + values[i]
            next_value = values[i]
    return out, returns


def update(
    pilot: model.Pilot,
    optimizer: torch.optim.Optimizer,
    steps: list[dict],
    advantage: list[float],
    weight: float,
    epochs: int,
    clip: float,
    micro: int,
    target_kl: float,
    rng: random.Random,
) -> dict:
    """Run the PPO update with per-question ratio clipping and the imitation term, stopping early past target_kl"""
    mean = sum(advantage) / len(advantage)
    spread = (sum((a - mean) ** 2 for a in advantage) / len(advantage)) ** 0.5 + 1e-8
    normalized = [(a - mean) / spread for a in advantage]

    def picked(batch: list[dict], logs: list[torch.Tensor], key: str) -> list[torch.Tensor]:
        return [d.gather(1, torch.tensor(r[key], device=d.device)[:, None]) for d, r in zip(logs, batch, strict=True)]

    stats, batches = {"kl": 0.0, "clipped": 0.0, "entropy": 0.0, "imitation_loss": 0.0}, 0
    if not epochs:
        return {**stats, "early_stop": False}
    # recompute the old log-probabilities so the first ratios are exactly 1
    old: list = [None] * len(steps)
    pilot.model.eval()
    with torch.no_grad():
        for ids in model.by_start(items=steps, size=micro, rng=rng):
            batch = [steps[i] for i in ids]
            for i, value in zip(
                ids, picked(batch=batch, logs=pilot.log_probabilities(batch), key="chosen"), strict=True
            ):
                old[i] = value[:, 0]
    pilot.model.train()
    for _ in range(epochs):
        for ids in model.by_start(items=steps, size=micro, rng=rng):
            batch = [steps[i] for i in ids]
            logs = pilot.log_probabilities(batch)
            new = torch.cat([p[:, 0] for p in picked(batch=batch, logs=logs, key="chosen")])
            log_ratio = new - torch.cat([old[i] for i in ids])
            ratio = log_ratio.exp()
            adv = torch.tensor([normalized[i] for i in ids for _ in steps[i]["chosen"]], device=new.device)
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            entropy = torch.stack([-(d.exp() * d.clamp(min=-1e4)).sum(-1).mean() for d in logs]).mean()
            expert = torch.stack([-p.mean() for p in picked(batch=batch, logs=logs, key="expert")]).mean()
            loss = (policy_loss - ENTROPY * entropy + weight * expert) * len(ids) / micro
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pilot.trainable(), 1.0)
            optimizer.step()
            pilot.updated()
            batches += 1
            stats["kl"] += ((ratio - 1) - log_ratio).mean().item()
            stats["clipped"] += ((ratio - 1).abs() > clip).float().mean().item()
            stats["entropy"] += entropy.item()
            stats["imitation_loss"] += expert.item()
            if stats["kl"] / batches > target_kl:
                return {**{k: v / batches for k, v in stats.items()}, "early_stop": True}
    return {**{k: v / batches for k, v in stats.items()}, "early_stop": False}


def fit_critic(critic: model.Critic, steps: list[dict], returns: list[float], rng: random.Random) -> float:
    """Fit the critic to the returns in four passes of 256-decision batches"""
    for _ in range(4):
        order = rng.sample(range(len(steps)), len(steps))
        for at in range(0, len(order), 256):
            ids = order[at : at + 256]
            prediction = critic([steps[i]["features"] for i in ids])
            value_loss = torch.nn.functional.mse_loss(
                prediction, torch.tensor([returns[i] for i in ids], device=prediction.device)
            )
            critic.optimizer.zero_grad()
            value_loss.backward()
            critic.optimizer.step()
    return value_loss.item()


def ppo(
    init: Annotated[Path, typer.Option(help="Adapter to start from")] = Path("output/dagger/final"),
    out: Annotated[Path, typer.Option(help="Output folder (adapters and log.jsonl)")] = Path("output/ppo"),
    iterations: int = 20,
    decay: Annotated[int, typer.Option(help="Iterations over which the imitation term decays to a tenth")] = 100,
    envs: Annotated[int, typer.Option(help="Flights in progress at once")] = 32,
    horizon: Annotated[int, typer.Option(help="Decisions each flight makes per iteration")] = 64,
    epochs: int = 1,
    lr: float = 1e-5,
    clip: float = 0.2,
    micro: Annotated[int, typer.Option(help="Decisions per forward pass (8 fit a 12 GB GPU)")] = 8,
    target_kl: float = 0.03,
    critic_warmup: Annotated[int, typer.Option(help="Initial iterations that train only the critic")] = 5,
    evaluate_every: Annotated[int, typer.Option(help="Iterations between saving and checking the adapter")] = 20,
    eval_seeds: Annotated[str, typer.Option(help="Development seeds each checkpoint flies, e.g. 0-9")] = "0-4",
    tasks: Annotated[str, typer.Option(help="Objectives and weights, e.g. circuit=3,course=1 (default: all)")] = "",
    delay: Annotated[str, typer.Option(help="Seconds each answer takes to apply, or a range like 0.1-2.5")] = "0.1-2.5",
    interval: Annotated[str, typer.Option(help="Minimum seconds between decisions, or a range like 0.1-1.0")] = "0.5",
    wind: Annotated[float, typer.Option(help="Maximum wind in knots (0 for each objective's default)")] = 12.0,
    variety: Annotated[bool, typer.Option(help="Vary the airfield, aircraft, gusts, readings and start")] = False,
    takeovers: Annotated[bool, typer.Option(help="Give some flights a few seconds of random answers")] = True,
    cap: Annotated[int, typer.Option(help="Decisions a flight may take before it is cut")] = 900,
    device: Annotated[str, typer.Option(help="cuda, or cpu for a quick check of the loop")] = "cuda",
    seed: int = 0,
) -> None:
    mix = curriculum.Curriculum(
        tasks=curriculum.task_weights(tasks),
        delay=curriculum.span(delay),
        interval=curriculum.span(interval),
        wind=wind,
        variety=variety,
        takeovers=takeovers,
        handovers=True,
    )
    rng, write_log = random.Random(seed), log.logger(out)
    pilot, critic = model.Pilot(init=init, device=device), model.Critic(device)
    optimizer = torch.optim.AdamW(pilot.trainable(), lr=lr, weight_decay=0.0)

    def fresh() -> episode.Episode:
        """Draw a new training flight, redrawing any that ended before the handover"""
        while True:
            job, delay, interval = mix.draw(rng)
            flight = episode.Episode(job, delay, interval, cap)
            if not flight.done:
                return flight

    live, finished = [fresh() for _ in range(envs)], []
    variants = ["standard", *(["varied"] if variety else []), *(["takeover"] if takeovers else [])]
    for iteration in range(iterations):
        started = time.time()
        steps = collect(pilot, live, horizon, fresh, finished)
        with torch.no_grad():
            values = critic([r["features"] for r in steps]).tolist()
            tails = critic([e.features() for e in live]).tolist()
            finals = {i: critic([r["final"]]).item() for i, r in enumerate(steps) if r["truncated"]}
        advantage, returns = advantages(steps, values, tails, finals, envs)
        weight = IMITATION * max(0.1, 1 - iteration / decay)
        stats = update(
            pilot,
            optimizer,
            steps,
            advantage,
            weight,
            epochs if iteration >= critic_warmup else 0,
            clip,
            micro,
            target_kl,
            rng,
        )
        value_loss = fit_critic(critic, steps, returns, rng)
        recent = finished[-64:]
        write_log(
            {
                "stage": "ppo",
                "iteration": iteration,
                "decisions": len(steps),
                "reward_per_decision": round(sum(r["reward"] for r in steps) / len(steps), 5),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
                "value_loss": round(value_loss, 5),
                "episodes": len(finished),
                "recent": evaluation.summarize(recent) if recent else {},
                "seconds": round(time.time() - started),
            }
        )
        if (iteration + 1) % evaluate_every == 0 or iteration == iterations - 1:
            folder = out / f"iteration-{iteration + 1}"
            pilot.save(folder)
            torch.save(critic.net.state_dict(), folder / "critic.pt")
            middle = sum(mix.delay) / 2
            results = evaluation.check(
                pilot=pilot, tasks=mix.tasks, seeds=curriculum.seed_list(eval_seeds), delay=middle, variants=variants
            )
            write_log({"stage": "evaluate", "iteration": iteration + 1, "delay": middle, "results": results})
    pilot.save(out / "final")


def robust(
    init: Annotated[Path, typer.Option(help="Adapter to start from")] = Path("output/ppo/final"),
    out: Annotated[Path, typer.Option(help="Output folder (adapters and log.jsonl)")] = Path("output/robust"),
    device: Annotated[str, typer.Option(help="cuda, or cpu for a quick check of the loop")] = "cuda",
) -> None:
    ppo(
        init=init,
        out=out,
        iterations=20,
        decay=60,
        evaluate_every=10,
        delay="0.05-2.5",
        interval="0.1-1.0",
        variety=True,
        cap=2000,
        device=device,
    )
