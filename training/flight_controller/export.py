import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import huggingface_hub
import torch
import typer
from peft import PeftModel
from transformers import AutoModelForCausalLM

from flight_controller import model

LLAMA_CPP_IMAGE = "ghcr.io/ggml-org/llama.cpp@sha256:0261bfbc7094add132aa3d1461f119ff04e6daedecc77efbf497e6199f413b70"
GGUF = Path("models", "flight-controller-Q8_0.gguf")  # what export writes
# the GGUF converter needs these next to the merged weights (fine-tuning doesn't change them)
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]


def export(
    adapter: Annotated[Path, typer.Option(help="Adapter folder")],
    out: Annotated[Path, typer.Option(help="Folder for the GGUF")] = GGUF.parent,
) -> None:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        merged = Path(temporary)
        base = AutoModelForCausalLM.from_pretrained(model.BASE_MODEL, dtype=torch.bfloat16)
        PeftModel.from_pretrained(base, str(adapter)).merge_and_unload().save_pretrained(str(merged))
        source = Path(huggingface_hub.snapshot_download(model.BASE_MODEL, allow_patterns=TOKENIZER_FILES))
        for name in TOKENIZER_FILES:
            shutil.copyfile(source / name, merged / name)
        user = f"{os.getuid()}:{os.getgid()}"
        docker = ["docker", "run", "--rm", "--user", user, "-v", f"{out}:/work", LLAMA_CPP_IMAGE]
        convert = ["--convert", "--outtype", "q8_0", "--outfile", f"/work/{GGUF.name}"]
        subprocess.run([*docker, *convert, f"/work/{merged.name}"], check=True)
    print(f"Wrote {out / GGUF.name}")
