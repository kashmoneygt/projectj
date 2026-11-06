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


def test_budget_defaults() -> None:
    assert core.Budget().timeout_s == 3600


def test_model_endpoint_and_server_stats() -> None:
    ep = core.ModelEndpoint(model_id="x", base_url="http://h:1/v1")
    assert ep.root_url == ""
    stats = core.ServerStats(requests=2, cache_hits=[10.0, 50.0])
    assert stats.avg_cache_hit == 30.0
    assert core.ServerStats().avg_cache_hit is None


def test_node_roundtrip() -> None:
    node = core.Node(
        name="t-node",
        dataset=core.DatasetSpec(name="swebench-verified", dataset_id="princeton-nlp/SWE-bench_Verified"),
        harness="pi",
        harness_version="1.3.0",
        model=core.ModelSpec(name="gemma-4-12b", source="path:/tmp/x.gguf", model_id="gemma-4-12b-it"),
        budget=core.Budget(timeout_s=900),
        tasks=10,
        seed=7,
    )
    node.runs["a__b-1"] = core.TaskRun(instance_id="a__b-1", status="ok", patch="diff", seconds=12.5, resolved=True)
    restored = core.Node.model_validate_json(node.model_dump_json())
    assert restored.name == "t-node"
    assert restored.model.source == "path:/tmp/x.gguf"
    assert restored.runs["a__b-1"].resolved is True
    assert restored.budget.timeout_s == 900
    assert restored.parent is None


def test_task_metrics_defaults() -> None:
    m = core.TaskMetrics()
    assert m.requests == 0
    assert m.tok_s == 0.0
    assert m.system_tokens == 0
    assert m.ttft_first_ms == 0.0 and m.ttft_avg_ms == 0.0 and m.ttft_max_ms == 0.0
    assert m.model_ms == 0 and m.think_ms == 0 and m.stop_reason == "" and m.patch_lines == 0
