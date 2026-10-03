import math
import threading
import time
from collections.abc import Callable
from concurrent import futures

from jym import core, runs

DECISION_SECONDS = 0.5  # sim seconds per lockstep decision, and the min gap between real-time ones
WATCHDOG_SECONDS = 3.0  # hold steady after this many sim seconds without an answer
FRAME_SECONDS = 1 / 30  # real-time frame length when a model plays
HUMAN_FRAME_SECONDS = 1 / 60  # and when a person plays
SCENE_SECONDS = 0.1  # how often to log the scene
RELEASE_SECONDS = 1.0  # release a person's controls after this long untouched


class PolicyFailed(RuntimeError):
    pass


def run_lockstep(
    environment: core.Environment,
    policy: core.Policy,
    task: core.Task,
    seed: int,
    log: runs.RunLog,
    stop: threading.Event | None = None,
) -> core.Result:
    """Play one episode in lockstep: the game waits for each decision, then runs DECISION_SECONDS"""
    stop = stop or threading.Event()
    run = Run(
        environment=environment,
        task=task,
        seed=seed,
        log=log,
        player=policy.name,
        metadata=getattr(policy, "metadata", {}),
        mode="lockstep",
    )
    try:
        run.reset()
        while not run.over(
            stop=stop, budget_spent=f"step budget of {task.max_steps} reached" if run.steps >= task.max_steps else None
        ):
            state, questions = {"goal": task.goal, **environment.state()}, environment.questions()
            began = time.monotonic()
            try:
                reply = policy.decide(state, questions)
            except Exception as error:
                if stop.is_set():  # a person stopped the run
                    continue
                raise PolicyFailed(f"{policy.name} failed: {error_text(error)}") from error
            choices = chosen(questions, reply)
            seconds = time.monotonic() - began
            run.decided(reply, seconds)
            log.event(
                kind="decision",
                step=run.steps,
                state=state,
                questions=questions,
                answers=reply.answers,
                seconds=round(seconds, 4),
                **reply.info,
            )
            environment.apply(choices)
            done = environment.advance(DECISION_SECONDS)
            log.event(kind="step", step=run.steps, action=choices, scene=environment.scene())
            if done:
                break
    except Exception as error:  # noqa: BLE001
        run.fail(error)
    return run.finish()


def run_realtime(
    environment: core.Environment,
    policy: core.Policy,
    task: core.Task,
    seed: int,
    log: runs.RunLog,
    stop: threading.Event | None = None,
    start_at: float | None = None,
    speed: float = 1.0,
    cut_reason: str | None = None,
) -> core.Result:
    """Play one episode in real time: the game keeps running while the policy decides"""
    stop = stop or threading.Event()
    run = Run(
        environment=environment,
        task=task,
        seed=seed,
        log=log,
        player=policy.name,
        metadata=getattr(policy, "metadata", {}),
        mode="realtime",
    )
    worker = futures.ThreadPoolExecutor(1, thread_name_prefix=f"decide-{policy.name}")
    budget, stale = task.max_steps * DECISION_SECONDS, 0
    try:
        run.reset()
        if start_at is not None:  # players in a heat start together
            time.sleep(max(0.0, start_at - time.time()))
        sim, next_ask, logged, held = 0.0, 0.0, -1.0, False
        pending: futures.Future | None = None
        state, questions, asked_at, began = {}, {}, 0.0, 0.0
        last = time.monotonic()
        spent = f"time budget of {budget:g} s of simulated time reached"
        while not run.over(stop=stop, budget_spent=spent if sim >= budget else None, cut_reason=cut_reason):
            if pending is None and sim >= next_ask:
                state, questions = {"goal": task.goal, **environment.state()}, environment.questions()
                asked_at, began = sim, time.monotonic()
                pending = ask(worker, policy, state, questions)
            if pending is not None and pending.done():
                try:
                    reply = pending.result()
                except Exception as error:
                    raise PolicyFailed(f"{policy.name} failed: {error_text(error)}") from error
                pending = None
                choices = chosen(questions, reply)
                seconds = time.monotonic() - began
                run.decided(reply, seconds)
                try:
                    environment.apply(choices)
                    applied = True
                except ValueError:  # the questions changed while deciding, so ask again
                    applied, stale = False, stale + 1
                log.event(
                    kind="decision",
                    step=run.steps,
                    state=state,
                    questions=questions,
                    answers=reply.answers,
                    seconds=round(seconds, 4),
                    asked_at=round(asked_at, 3),
                    applied_at=round(sim, 3) if applied else None,
                    **reply.info,
                )
                if applied:  # the tick is needed for exact replays
                    action = {**choices, "tick": environment.ticks, "sim_time": round(sim, 3)}
                    log.event(kind="step", step=run.steps, action=action)
                held = False
                next_ask = max(asked_at + DECISION_SECONDS, sim) if applied else sim
            elif pending is not None and not held and sim - asked_at > WATCHDOG_SECONDS:
                environment.hold()
                log.event(kind="watchdog", sim_time=round(sim, 3), tick=environment.ticks)
                held = True
            now = time.monotonic()
            seconds = min(0.1, now - last) * speed
            last = now
            done = environment.advance(seconds)
            sim += seconds
            if sim - logged >= SCENE_SECONDS or done:
                log.event(kind="scene", scene=environment.scene())
                logged = sim
            if done:
                break
            time.sleep(max(0.0, FRAME_SECONDS - (time.monotonic() - now)))
    except Exception as error:  # noqa: BLE001
        run.fail(error)
    finally:
        worker.shutdown(wait=False, cancel_futures=True)
    return run.finish(stale_answers=stale, final_tick=environment.ticks if run.ready else None)


def run_human(
    environment: core.Environment,
    task: core.Task,
    seed: int,
    log: runs.RunLog,
    stop: threading.Event,
    inputs: Callable[[], tuple[dict, float]],
    on_frame: Callable[[dict], None],
) -> core.Result:
    """Play one episode as a person with the controls"""
    run = Run(
        environment=environment,
        task=task,
        seed=seed,
        log=log,
        player="human",
        metadata={"input": "controls"},
        mode="realtime",
    )
    released = False
    try:
        run.reset()
        on_frame(environment.scene())
        last = logged = time.monotonic()
        while not run.over(stop):
            values, idle = inputs()
            if values:
                environment.control(values)
                log.event(kind="controls", values=values)
                run.steps, released = run.steps + 1, False
            elif idle > RELEASE_SECONDS and not released:
                environment.control(environment.RELEASED)
                released = True
            now = time.monotonic()
            done = environment.advance(now - last)
            last = now
            scene = environment.scene()
            on_frame(scene)
            if now - logged >= SCENE_SECONDS or done:
                log.event(kind="scene", scene=scene)
                logged = now
            if done:
                break
            time.sleep(max(0.0, HUMAN_FRAME_SECONDS - (time.monotonic() - now)))
    except Exception as error:  # noqa: BLE001
        run.fail(error)
    return run.finish()


class Run:
    """The log and tally of one episode"""

    def __init__(
        self,
        environment: core.Environment,
        task: core.Task,
        seed: int,
        log: runs.RunLog,
        player: str,
        metadata: dict,
        mode: str,
    ):
        self.environment, self.task, self.seed, self.log, self.player = environment, task, seed, log, player
        self.started = time.monotonic()
        self.steps, self.latencies, self.waited = 0, [], 0.0
        self.ending: str | None = None
        self.reason: str | None = None
        self.ready = False  # reset, so it can be scored
        log.write(
            name="manifest.json",
            data=runs.manifest(
                run_id=log.run_id,
                environment=environment,
                policy=player,
                metadata=metadata,
                task=task,
                seed=seed,
                mode=mode,
            ),
        )

    def reset(self) -> None:
        self.environment.reset(self.task, self.seed)
        self.ready = True
        self.log.event(kind="reset", state=self.environment.state(), scene=self.environment.scene())

    def over(self, stop: threading.Event, budget_spent: str | None = None, cut_reason: str | None = None) -> bool:
        if stop.is_set():
            self.ending, self.reason = ("truncated", cut_reason) if cut_reason else ("stopped", "stop requested")
        elif budget_spent:
            self.ending, self.reason = "truncated", budget_spent
        elif time.monotonic() - self.started >= self.task.max_seconds:
            self.ending, self.reason = "truncated", f"time budget of {self.task.max_seconds:g} s reached"
        return self.ending is not None

    def decided(self, reply: core.Reply, seconds: float) -> None:
        self.steps += 1
        self.latencies.append(own_seconds(seconds, reply.info))
        self.waited += reply.info.get("waited_seconds", 0.0)

    def fail(self, error: Exception) -> None:
        if isinstance(error, PolicyFailed):
            self.ending, self.reason = "error", str(error)
            self.log.event(
                kind="error", stage="policy", error=self.reason, reply=getattr(error.__cause__, "reply", None)
            )
        else:
            self.ending, self.reason = "error", f"environment failed: {error_text(error)}"
            self.log.event(kind="error", stage="environment", error=self.reason)

    def finish(self, **metrics: int | None) -> core.Result:
        """Score the episode, close the environment and write result.json"""
        evaluation: core.Evaluation | None = None
        try:
            if self.ready:
                evaluation = self.environment.evaluate()
                self.log.event(kind="evaluation", evaluation=evaluation)
        except Exception as error:  # noqa: BLE001
            self.ending, self.reason = "error", f"evaluation failed: {error_text(error)}"
            self.log.event(kind="error", stage="evaluate", error=self.reason)
        finally:
            self.environment.close()
        status = core.status(evaluation, self.ending)
        result = core.Result(
            run_id=self.log.run_id,
            env=self.task.env,
            task=self.task.id,
            task_version=self.task.version,
            policy=self.player,
            seed=self.seed,
            status=status,
            score=core.score(self.task, evaluation) if evaluation else 0,
            steps=self.steps,
            seconds=round(time.monotonic() - self.started, 3),
            milestones=sorted(evaluation.milestones) if evaluation else [],
            reason=(evaluation.failure if evaluation and status == "failure" else None) or self.reason,
            metrics={
                "evaluation": evaluation.metrics if evaluation else {},
                "decision_seconds_p50": percentile(values=self.latencies, fraction=0.5),
                "decision_seconds_p95": percentile(values=self.latencies, fraction=0.95),
                "waited_seconds": round(self.waited, 3),
                **metrics,
            },
        )
        self.log.event(kind="end", status=result.status, score=result.score, reason=result.reason)
        self.log.write(name="result.json", data=result)
        return result


def ask(worker: futures.Executor, policy: core.Policy, state: dict, questions: dict) -> futures.Future:
    """Ask the policy on the worker thread, or directly if it's synchronous"""
    if not getattr(policy, "synchronous", False):
        return worker.submit(policy.decide, state, questions)
    answer: futures.Future = futures.Future()
    try:
        answer.set_result(policy.decide(state, questions))
    except Exception as error:  # noqa: BLE001
        answer.set_exception(error)
    return answer


def chosen(questions: dict[str, dict], reply: core.Reply) -> dict[str, str]:
    choices = {}
    for qid, question in questions.items():
        option = (reply.answers.get(qid) or {}).get("choice")
        if option not in question["criteria"]:
            raise PolicyFailed(f"answer to {qid!r} is {option!r}, not one of {list(question['criteria'])}")
        choices[qid] = option
    return choices


def own_seconds(seconds: float, info: dict) -> float:
    """Return a decision's seconds without the time it waited for a shared CPU"""
    return max(0.0, seconds - info.get("waited_seconds", 0.0))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)], 4)


def error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"
