import importlib.metadata
import itertools
import json
import os
import time
from pathlib import Path

import huggingface_hub

from jym import core, paths

try:
    import laya
except ImportError:
    laya = None

LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"


def checkpoint() -> Path:
    return paths.models() / LAYA_REPO


def download() -> Path:
    huggingface_hub.snapshot_download(
        LAYA_REPO,
        revision=LAYA_REVISION,
        local_dir=checkpoint(),
        allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
    )
    return checkpoint()


def readiness() -> str:
    if laya is None:
        return "needs uv sync --extra laya"
    return "ready" if (checkpoint() / "model.safetensors").is_file() else "downloads its checkpoint on first use"


class LayaPolicy:
    name = "laya"

    def __init__(self):
        if laya is None:
            raise core.Unavailable("Laya needs its extra: uv sync --extra laya")
        started = time.perf_counter()
        device = os.environ.get("LAYA_DEVICE", "cuda")
        self.agent = laya.load(str(download()), device=device)
        found = {str(self.agent.device), *(str(p.device) for p in self.agent.model.parameters())}
        if any(used.split(":")[0] != device.split(":")[0] for used in found):
            raise core.Unavailable(f"Laya was asked to run on {device} but runs on {sorted(found)}")
        self.metadata = {
            "laya": importlib.metadata.version("laya"),
            "revision": LAYA_REVISION,
            "checkpoints": ["english"],
            "device": device,
            "load_seconds": round(time.perf_counter() - started, 2),
        }

    def decide(self, state: dict, questions: dict) -> core.Reply:
        text = json.dumps(state, ensure_ascii=False)
        tokens = {
            qid: check_packing(agent=self.agent, state=text, question=question) for qid, question in questions.items()
        }
        started = time.perf_counter()
        raw = self.agent.system_one(text, questions)
        warnings = core.check_answers(questions, raw.get("answers"))
        return core.Reply(
            answers=raw["answers"],
            info={
                "model": f"laya/english@{LAYA_REVISION[:8]}",
                "latency_seconds": round(time.perf_counter() - started, 4),
                "usage": raw.get("usage"),
                "packed_tokens": tokens,
                "warnings": warnings,
            },
        )


def check_packing(agent: "laya.Agent", state: str, question: dict) -> int:
    """Return a request's token count, refusing any that would be truncated"""
    tok, config = agent.tok, agent.cfg
    internal = laya.Agent._to_internal(question)
    texts = [f"{internal['t']} question: {internal['ins']}", state, *laya.render_options(internal)]
    if any(special in text for special in tok.all_special_tokens if special for text in texts):
        raise ValueError("The request contains tokenizer special tokens")
    ids = [tok(text, add_special_tokens=False)["input_ids"] for text in texts[:2]]
    options = [[tok.mask_token_id, *tok(f" {text}", add_special_tokens=False)["input_ids"]] for text in texts[2:]]
    expected = [
        tok.cls_token_id,
        *ids[0],
        tok.sep_token_id,
        *itertools.chain(*options),
        tok.sep_token_id,
        *ids[1],
        tok.sep_token_id,
    ]
    limit = config.get("max_len", 512)
    sequence, markers = laya.common.build_sequence(tok, state, internal, limit, config.get("head_max_len", 192))
    if sequence != expected or len(markers) != len(options):
        raise ValueError(
            f"The checkpoint would truncate this request: it needs {len(expected)} tokens, the limit is {limit}"
        )
    return len(sequence)
