import random

import pytest
import torch
import typer

from flight_controller import curriculum, episode, evaluation, ppo, reward
from jym import environments, runner
from jym.environments.flight import airfield, objectives


def test_progress_pays_nothing_over_a_flight(monkeypatch):
    only_progress = reward.Reward(crash=0.0, hard_touchdown=0.0, time=0.0, answer_change=0.0)
    monkeypatch.setattr("flight_controller.reward.REWARD", only_progress)
    flight = episode.Episode(job=episode.Job(task="drone-circuit", seed=10_003, handover=4), delay=1.5)
    start, discount, total, points = flight.progress, 1.0, 0.0, 0.0
    while not flight.done:
        before = set(flight.env.evaluator.milestones)
        reward_value, _ = flight.step(choices=None)
        points += discount * sum(flight.task.milestones[m] for m in flight.env.evaluator.milestones - before) / 100
        total += discount * reward_value
        discount *= only_progress.discount(flight.seconds)
    assert start > 0 and flight.score() == 100 and not flight.truncated
    assert total == pytest.approx(points - only_progress.progress_weight * start)  # it guides, but pays nothing


@pytest.mark.parametrize("task", ["circuit", "drone-circuit"])
def test_landing_progress_needs_the_landing_area(task):
    flight = episode.Episode(episode.Job(task=task, seed=10_061, settings={"start": "final"}))
    env, aim = flight.env, airfield.THRESHOLD_NORTH_M + airfield.AIM_PAST_THRESHOLD_M
    rolling = {**env.sample, "on_ground": True, "agl_m": airfield.PARKED_AGL_M, "ground_speed_kt": 30.0}
    env.sample = {**rolling, "north_m": aim, "east_m": 0.0}
    assert reward.progress(env, flight.start) == pytest.approx(0.85)  # on the runway, or the drone's spot
    env.sample = {**rolling, "north_m": 5000.0, "east_m": 3000.0}
    assert reward.progress(env, flight.start) == pytest.approx(0.5)  # touching down in a field earns no progress


def test_air_start_pays_only_its_flight():
    flight = episode.Episode(episode.Job(task="circuit", seed=10_004, settings={"start": "final"}))
    assert {"takeoff", "route"} <= flight.env.evaluator.milestones
    reward_value, _ = flight.step({qid: next(iter(spec["criteria"])) for qid, spec in flight.questions.items()})
    assert abs(reward_value) < 0.01  # nothing for the take-off and route it started with


def test_delayed_answers_are_labelled_ahead():
    flight = episode.Episode(job=episode.Job(task="takeoff", seed=10_003), delay=1.5)
    shown, label = {}, {}
    while label.get("lift_off") != "lift off":
        shown = flight.expert_choices()
        _, label = flight.step(choices=None)
    assert shown["lift_off"] == "not yet"  # the Autopilot's answer when asked; the label is its answer 1.5 s later
    assert flight.seconds == pytest.approx(1.5, abs=0.01)  # each answer applies after its delay


def test_curriculum_draws_are_deterministic():
    teaching, rng, again = curriculum.Curriculum(), random.Random(4), random.Random(4)
    draws = [teaching.draw(rng) for _ in range(60)]
    assert draws == [teaching.draw(again) for _ in range(60)]
    jobs = [job for job, _, _ in draws]
    assert all(job.seed >= curriculum.TRAIN_SEED for job in jobs)  # never a development or test seed
    assert {job.task for job in jobs} <= set(curriculum.TRAIN_TASKS) and any(job.takeover for job in jobs)
    assert all(0.1 <= delay <= 2.5 and interval == runner.DECISION_SECONDS for _, delay, interval in draws)


def test_robust_settings_are_drawn_per_flight():
    robust = curriculum.Curriculum(delay=(0.05, 2.5), interval=(0.1, 1.0), variety=True)
    draws = [robust.draw(random.Random(seed)) for seed in range(60)]
    jobs = [job for job, _, _ in draws]
    assert all(job.settings["airfield"] == "varied" and job.settings["dynamics"] == 0.2 for job in jobs)
    gusts, noise = [job.settings["gusts_kt"] for job in jobs], [job.settings["noise"] for job in jobs]
    assert 0 in gusts and 0 < max(gusts) <= 8 and 0 <= min(noise) < max(noise) <= 1
    circuits = [job for job in jobs if job.task.endswith("circuit")]
    assert {job.settings["start"] for job in circuits} == {"runway", "downwind", "final"}
    assert all("start" not in job.settings for job in jobs if job not in circuits)
    delays, intervals = [delay for _, delay, _ in draws], [interval for _, _, interval in draws]
    assert 0.05 <= min(delays) < max(delays) <= 2.5 and 0.1 <= min(intervals) < max(intervals) <= 1.0


@pytest.mark.parametrize("task", ["gauntlet", "drone-circuit"])
def test_critic_sees_26_features(task):
    assert episode.FEATURES == len(episode.Episode(episode.Job(task=task, seed=10_031)).features()) == 26


def test_evaluation_variants_are_fixed_by_seed():
    tasks, seeds = ["takeoff", "course", "drone-course"], range(4)
    jobs = {variant: evaluation.variant_jobs(variant, tasks, seeds) for variant in evaluation.VARIANTS}
    assert jobs == {variant: evaluation.variant_jobs(variant, tasks, seeds) for variant in evaluation.VARIANTS}
    assert jobs["standard"] == [episode.Job(task, seed) for task in tasks for seed in seeds]
    assert all(job.settings == {"wind_max_kt": 15.0} for job in jobs["windy"])
    assert {job.task for job in jobs["hard-course"]} == {"course", "drone-course"}
    rings = [ring for job in jobs["hard-course"] for ring in objectives.validate_rings(job.settings["rings"])]
    assert all(60 <= ring["height_m"] <= 500 for ring in rings)
    assert all(job.settings["airfield"] == "varied" and "start" not in job.settings for job in jobs["varied"])
    budget = {
        task: environments.get_task(env="flight", task=task).max_steps * runner.DECISION_SECONDS for task in tasks
    }
    assert all(2 <= job.takeover <= 8 and job.takeover_at <= 0.4 * budget[job.task] for job in jobs["takeover"])
    with pytest.raises(ValueError, match="Unknown variant"):
        evaluation.variant_jobs(variant="stormy", tasks=tasks, seeds=seeds)


def test_evaluate_rejects_ignored_model_options(tmp_path, monkeypatch):
    def fail_pilot(adapter):
        raise AssertionError("evaluate loaded a model before validating options")

    monkeypatch.setattr(evaluation.model, "Pilot", fail_pilot)
    with pytest.raises(typer.BadParameter, match="--scripted cannot be used with --adapter"):
        evaluation.evaluate(scripted=True, adapter=tmp_path / "adapter")
    with pytest.raises(typer.BadParameter, match="--scripted cannot be used with --gguf"):
        evaluation.evaluate(scripted=True, gguf=tmp_path / "model.gguf")
    with pytest.raises(typer.BadParameter, match="--gguf cannot be used with --adapter"):
        evaluation.evaluate(adapter=tmp_path / "adapter", gguf=tmp_path / "model.gguf")


def test_imitation_weight_follows_decay(tmp_path, monkeypatch):
    seen = []

    class Model:
        def trainable(self):
            return [torch.nn.Parameter(torch.zeros(1))]

        def save(self, folder):
            folder.mkdir(parents=True, exist_ok=True)

    def collect(model, live, horizon, fresh, finished):
        step = {"env": 0, "reward": 0.0, "seconds": 0.5, "done": False, "truncated": False}
        return [{**step, "features": live[0].features()}]

    def update(model, optimizer, steps, advantage, weight, *rest):
        seen.append(weight)
        return {}

    monkeypatch.setattr(ppo.model, "Pilot", lambda init, device: Model())
    monkeypatch.setattr(ppo, "collect", collect)
    monkeypatch.setattr(ppo, "update", update)
    monkeypatch.setattr(ppo.evaluation, "check", lambda *args, **kwargs: {})

    def weights(iterations: int) -> list[float]:
        seen.clear()
        flights = {"envs": 1, "tasks": "drone-takeoff", "device": "cpu"}
        ppo.ppo(init=tmp_path / "dagger", out=tmp_path / str(iterations), iterations=iterations, decay=100, **flights)
        return list(seen)

    first, whole = weights(iterations=20), weights(iterations=100)
    assert first == whole[:20]
    assert first[0] == ppo.IMITATION > first[-1] > whole[-1] == pytest.approx(ppo.IMITATION / 10)
