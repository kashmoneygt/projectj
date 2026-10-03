import math
import re
import types
import urllib.parse

import pytest

from jym import core, llama, logits

QUESTIONS = {
    "bank": core.choice(instructions="Which way?", options={"left": "turn left", "level": None, "right": "turn right"}),
    "throttle": core.choice(instructions="How much power?", options={"idle": None, "full": None}),
}
STATE = {
    "goal": "Fly.",
    "airspeed_kt": 60,
    "stall_warning": False,
    "target": {"direction": "left", "distance_m": 900, "height_band_m": [60, 400]},
}
VOCABULARY: dict[str, int] = {}


def tokenize(text: str) -> list[int]:
    # one token per character, except special tokens and a space before a capital
    pieces = re.findall(r"<\|[a-z_]+\|>| [A-Z]|.", text, re.DOTALL)
    return [VOCABULARY.setdefault(piece, len(VOCABULARY)) for piece in pieces]


def test_prompt_is_exact():
    start = (
        "<|im_start|>system\n"
        "You are the decision maker in an environment. Read the goal, the questions and the current state, then "
        "answer every question with the letter of the option that best achieves the goal.<|im_end|>\n"
        "<|im_start|>user\n"
        "Goal: Fly.\n"
        "\n"
        "Questions:\n"
        "bank: Which way?\n"
        "  A. left: turn left\n"
        "  B. level\n"
        "  C. right: turn right\n"
        "throttle: How much power?\n"
        "  A. idle\n"
        "  B. full\n"
        "\n"
        "State:\n"
    )
    end = (
        "airspeed_kt: 60\n"
        "stall_warning: no\n"
        "target: direction=left, distance_m=900, height_band_m=[60, 400]<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "\n"
        "</think>\n"
        "\n"
    )
    assert logits.prompt_start(STATE["goal"], QUESTIONS) == start
    assert logits.prompt_end(STATE) == end
    prompt = logits.Prompt(tokenize=tokenize, letters=logits.letter_ids(tokenize), state=STATE, questions=QUESTIONS)
    assert prompt.sequence(chosen=[2, 0])[0] == tokenize(start + end + "bank: C\nthrottle: A")


class LlamaServer:
    # a fake llama-server that always rates option B most likely
    def __init__(self):
        self.prompts = []

    def request(self, method, url, json=None, timeout=None):
        path = urllib.parse.urlsplit(url).path
        if path == "/props":
            body = {"default_generation_settings": {"n_ctx": 4096}}
        elif path == "/tokenize":
            body = {"tokens": tokenize(json["content"])}
        else:
            pieces = {i: piece for piece, i in VOCABULARY.items()}
            self.prompts.append("".join(pieces[i] for i in json["prompt"]))
            letters = logits.letter_ids(tokenize)
            top = [{"id": i, "logprob": -0.1 if i == letters[1] else -3.0} for i in letters]
            body = {"completion_probabilities": [{"top_logprobs": top}], "timings": {"prompt_n": 7}}
        return types.SimpleNamespace(json=lambda: body, raise_for_status=lambda: None)


def test_logits_answer_each_question_in_turn(monkeypatch):
    monkeypatch.setattr(llama, "serve", lambda model, device: "http://llama")
    server, model = (
        LlamaServer(),
        llama.Gguf(name="qwen", file="Qwen3-0.6B-Q8_0.gguf", repo="Qwen/Qwen3-0.6B-GGUF", revision="23749fef"),
    )
    reply = logits.LogitsPolicy(name="qwen@cpu", model=model, device="cpu", session=server).decide(STATE, QUESTIONS)
    assert [answer["choice"] for answer in reply.answers.values()] == ["level", "full"]
    prompt = logits.prompt_start(STATE["goal"], QUESTIONS) + logits.prompt_end(STATE)
    assert server.prompts == [prompt + "bank:", prompt + "bank: B\nthrottle:"]
    assert reply.answers["bank"]["confidence"] == pytest.approx(math.exp(-0.1) / (math.exp(-0.1) + 2 * math.exp(-3)))
    assert core.check_answers(QUESTIONS, reply.answers) == []
