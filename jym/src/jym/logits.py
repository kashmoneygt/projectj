import json
import math
import time
from collections.abc import Callable
from typing import Any

import requests

from jym import core, llama

LETTERS = "ABCDEFGH"
SYSTEM = (
    "You are the decision maker in an environment. Read the goal, the questions and the current state, then answer "
    "every question with the letter of the option that best achieves the goal."
)
# an empty think block puts Qwen3 in non-thinking mode
ANSWER_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def as_text(value: Any) -> str:
    if value is True or value is False:
        return "yes" if value else "no"
    if value is None:
        return "none"
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_state(state: dict) -> str:
    lines = []
    for key, value in state.items():
        if isinstance(value, dict):
            lines.append(f"{key}: " + ", ".join(f"{k}={as_text(v)}" for k, v in value.items()))
        else:
            lines.append(f"{key}: {as_text(value)}")
    return "\n".join(lines)


def prompt_start(goal: str, questions: dict) -> str:
    """Return the prompt up to the state (the goal and questions), which the server caches"""
    lines = [f"<|im_start|>system\n{SYSTEM}<|im_end|>", "<|im_start|>user", f"Goal: {goal}", "", "Questions:"]
    for qid, question in questions.items():
        lines.append(f"{qid}: {question['instructions']}")
        for letter, (option, about) in zip(LETTERS, question["criteria"].items(), strict=False):
            lines.append(f"  {letter}. {option}" + (f": {about}" if about else ""))
    return "\n".join(lines) + "\n\nState:\n"


def prompt_end(state: dict) -> str:
    return render_state({key: value for key, value in state.items() if key != "goal"}) + f"<|im_end|>\n{ANSWER_PREFIX}"


class Prompt:
    """One decision's prompt as token ids. Answers are lines like "turn: C", one per question"""

    def __init__(self, tokenize: Callable[[str], list[int]], letters: list[int], state: dict, questions: dict):
        self.questions = list(questions)
        self.options = [list(questions[qid]["criteria"]) for qid in self.questions]
        # tokenize the parts separately so the start's tokens stay the same and match training
        self.start = tokenize(prompt_start(state["goal"], questions))
        self.end = tokenize(prompt_end(state))
        self.ids = self.start + self.end
        self.answer_prefixes = [tokenize(("" if i == 0 else "\n") + f"{qid}:") for i, qid in enumerate(self.questions)]
        self.letters = [letters[: len(options)] for options in self.options]

    def up_to_answer(self, index: int, chosen: list[int]) -> list[int]:
        """Return the ids up to a question's answer letter"""
        ids = list(self.ids)
        for i in range(index):
            ids += self.answer_prefixes[i] + [self.letters[i][chosen[i]]]
        return ids + self.answer_prefixes[index]

    def sequence(self, chosen: list[int]) -> tuple[list[int], list[int]]:
        """Return the whole sequence with the chosen letters, and the position before each letter"""
        ids, positions = list(self.ids), []
        for i in range(len(self.questions)):
            ids += self.answer_prefixes[i]
            positions.append(len(ids) - 1)
            ids.append(self.letters[i][chosen[i]])
        return ids, positions

    def after_start(self, chosen: list[int]) -> tuple[list[int], list[int]]:
        """Return sequence() after the prompt's start, with positions counted from there"""
        ids, positions = self.sequence(chosen)
        return ids[len(self.start) :], [position - len(self.start) for position in positions]


def letter_ids(tokenize: Callable[[str], list[int]]) -> list[int]:
    """Return the token of each option letter after a question's colon and a space"""
    ids = []
    for letter in LETTERS:
        base, with_letter = tokenize("x\nq:"), tokenize(f"x\nq: {letter}")
        if with_letter[: len(base)] != base or len(with_letter) != len(base) + 1:
            raise core.Unavailable(f"Option letter {letter!r} is not one token after a colon for this tokenizer")
        ids.append(with_letter[-1])
    return ids


def softmax(logprobs: list[float]) -> list[float]:
    top = max(logprobs)
    weights = [math.exp(value - top) for value in logprobs]
    return [weight / sum(weights) for weight in weights]


class LogitsPolicy:
    """A GGUF model on llama.cpp that picks the most likely option letter for each question"""

    def __init__(self, name: str, model: llama.Gguf, device: str, session: requests.Session | None = None):
        self.name = name
        self.server = llama.LlamaServer(model, device, session)
        self.server.ensure()
        self.context = self.server.props["default_generation_settings"]["n_ctx"]
        self.tokens: dict[str, list[int]] = {}
        self.letters = letter_ids(self.tokenize)
        self.metadata = {"model": self.server.model_name, "device": device, "n_ctx": self.context}

    def tokenize(self, text: str) -> list[int]:
        # cached, since each tokenization is a round trip to the server
        ids = self.tokens.get(text)
        if ids is None:
            ids = self.server.tokenize(text)
            if len(self.tokens) >= 256:
                self.tokens.clear()
            self.tokens[text] = ids
        return list(ids)

    def warm(self, goal: str, questions: dict) -> None:
        """Cache the prompt start for this goal and questions"""
        prompt = Prompt(tokenize=self.tokenize, letters=self.letters, state={"goal": goal}, questions=questions)
        with llama.turn(self.server.device):
            self.server.next_token_logprobs(
                prompt=prompt.up_to_answer(index=0, chosen=[]), candidates=prompt.letters[0]
            )

    def decide(self, state: dict, questions: dict) -> core.Reply:
        asked = time.perf_counter()
        with llama.turn(self.server.device):
            started = time.perf_counter()
            answers, calls = self.answer(state, questions)
            seconds = time.perf_counter() - started
        return core.Reply(
            answers=answers,
            info={
                "model": self.server.model_name,
                "latency_seconds": round(seconds, 4),
                "waited_seconds": round(started - asked, 4),
                "calls": calls,
            },
        )

    def answer(self, state: dict, questions: dict) -> tuple[dict, list]:
        """Answer each question in turn, after the previous answers"""
        prompt = Prompt(self.tokenize, self.letters, state, questions)
        chosen: list[int] = []
        answers, calls = {}, []
        for index, qid in enumerate(prompt.questions):
            ids = prompt.up_to_answer(index, chosen)
            if len(ids) + 1 > self.context:
                raise ValueError(f"{qid}: the prompt needs {len(ids) + 1} tokens; the context is {self.context}")
            logprobs, timings = self.server.next_token_logprobs(prompt=ids, candidates=prompt.letters[index])
            probabilities = softmax(logprobs)
            best = max(range(len(probabilities)), key=probabilities.__getitem__)
            chosen.append(best)
            options = prompt.options[index]
            answers[qid] = {
                "type": "choice",
                "choice": options[best],
                "confidence": probabilities[best],
                "probabilities": dict(zip(options, probabilities, strict=True)),
            }
            calls.append(
                {
                    "question": qid,
                    "prompt_tokens": len(ids),
                    "processed_tokens": timings.get("prompt_n"),
                    "server_ms": timings.get("prompt_ms"),
                }
            )
        return answers, calls
