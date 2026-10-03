import importlib.metadata
import json
import os
import time

from typesafe_sdk import RetryPolicy, TypeSafeClient

from jym import core

JEV_MODEL = "jev-1.13.0"


def readiness() -> str:
    return "ready" if os.environ.get("JEV_API_KEY") else "set JEV_API_KEY"


class JevPolicy:
    name = "jev"

    def __init__(self, client: TypeSafeClient | None = None):
        if client is None:
            key = os.environ.get("JEV_API_KEY")
            if not key:
                raise core.Unavailable("Set JEV_API_KEY to use Jev")
            client = TypeSafeClient(api_key=key, model=JEV_MODEL, retry=RetryPolicy(max_retries=2), timeout=30)
        self.client = client
        self.metadata = {"model": JEV_MODEL, "sdk": importlib.metadata.version("typesafe-sdk")}

    def decide(self, state: dict, questions: dict) -> core.Reply:
        started = time.perf_counter()
        response = self.client.system_one(json.dumps(state, ensure_ascii=False), questions, model=JEV_MODEL)
        body = response.raw_http_response.json()
        if body.get("model") != JEV_MODEL:
            raise core.InvalidAnswer(
                message=f"Jev answered as {body.get('model')!r}, expected {JEV_MODEL!r}", reply=body
            )
        warnings = core.check_answers(questions, body.get("answers"))
        return core.Reply(
            answers=body["answers"],
            info={
                "model": JEV_MODEL,
                "latency_seconds": round(time.perf_counter() - started, 4),
                "usage": body.get("usage"),
                "warnings": warnings,
            },
        )
