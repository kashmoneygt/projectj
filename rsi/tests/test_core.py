from pathlib import Path

from rsi import core


def test_load_models_parses_registry() -> None:
    specs = core.load_models(path=Path(__file__).resolve().parents[1] / "models.toml")
    by_name = {spec.name: spec for spec in specs}
    gemma = by_name["gemma-4-12b"]
    qwen = by_name["qwen-3.8-27b"]
    assert gemma.source.startswith("hf:unsloth/gemma-4-12b-it-GGUF")
    assert gemma.model_id == "gemma-4-12b-it"
    assert gemma.backend == "llama.cpp"
    assert qwen.source.startswith("splash:unsloth/Qwen3.8-27B-GGUF")
    assert qwen.backend == "splash"
    assert gemma.thinking and qwen.thinking
    assert (gemma.parallel, qwen.parallel) == (1, 2)
    assert (gemma.port, qwen.port) == (11500, 11501)
    assert (gemma.memory_gb, qwen.memory_gb) == (0, 24)
    assert gemma.seed == 42
    assert (qwen.ctx, qwen.max_output_tokens, qwen.output_cap) == (131072, 0, 131072)
    assert (gemma.ctx, gemma.output_cap) == (131072, 16384)
    assert gemma.sampling == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert qwen.sampling == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert qwen.chat_template_kwargs == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
        "preserve_thinking": True,
    }
    assert gemma.chat_template_kwargs == {"enable_thinking": True}
