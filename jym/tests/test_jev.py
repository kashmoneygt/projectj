import types

import pytest

from jym import core, jev

QUESTIONS = {"move": core.choice(instructions="Which way?", options={"left": None, "right": None})}


class Client:
    def __init__(self, model):
        self.model = model

    def system_one(self, state, questions, model):
        body = {"model": self.model, "answers": {"move": {"type": "choice", "choice": "right"}}}
        return types.SimpleNamespace(raw_http_response=types.SimpleNamespace(json=lambda: body))


def test_jev_answers_only_as_its_pinned_model():
    reply = jev.JevPolicy(Client(jev.JEV_MODEL)).decide(state={"goal": "win"}, questions=QUESTIONS)
    assert reply.answers["move"]["choice"] == "right" and reply.info["warnings"] == []
    with pytest.raises(core.InvalidAnswer, match="answered as 'jev-0.1'"):
        jev.JevPolicy(Client(model="jev-0.1")).decide(state={"goal": "win"}, questions=QUESTIONS)
