# rsi

rsi: Recursive self-improvement of a coding-agent harness.

We run a local model in the [pi](https://github.com/earendil-works/pi) coding-agent harness on a coding benchmark. A teacher model then looks at how those runs went and writes changes to the harness, called variants. We run each variant on the same tasks, keep the ones that do better, and have the teacher write new variants from those. The goal is an evolved version of pi that makes the model better at agentic coding.

Currently, we have the following onboarded:

- **Harness**: pi 1.4.0, changed only through pi's own packages (extensions, system prompt, settings, skills)
- **Models**: [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) served by splash and [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B-it) served by llama.cpp, each with its vendor's recommended settings (see [models.toml](models.toml))
- **Benchmark**: [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
- **Teacher**: Claude Opus 5.5 through GitHub Copilot CLI

## Run it

Needs [uv](https://docs.astral.sh/uv/), Docker, GitHub Copilot CLI logged in with `/login`, and splash (`brew install incoai/tap/splash`). splash serves models offline, so download Qwen once with `splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`.

```sh
uv sync
uv run rsi models up qwen-3.8-27b

# run pi with no changes (the baseline) on 100 tasks, about 6 hours
uv run rsi images build --tasks 100 --clone-tasks
uv run rsi run --model qwen-3.8-27b --benchmark swebench-verified --tasks 100

# pick the tasks variants are judged on, then run the baseline on them three times
uv run rsi probe --root qwen-3.8-27b--pi--swebench-verified--baseline--100t--s42
for seed in 43 44 45; do uv run rsi run --model qwen-3.8-27b --benchmark swebench-verified --probe --seed $seed; done

# have the teacher write variants, then run each one and see how it did against its parent
uv run rsi variants generate --parent qwen-3.8-27b--pi--swebench-verified--baseline--100t--s42
uv run rsi run --variant <name> --probe
uv run rsi variants list
```

To watch the runs and the teacher as they go, run `uv run rsi view --open`.

For help with any command, run `uv run rsi <command> --help`.

## Plug-in support

The harness, benchmark, model server and teacher are all plug-ins. Each one is a folder with a file that implements a protocol in [src/rsi/](src/rsi/):

- harness: `harnesses/<name>/harness.py` exposing `HARNESS`, like [pi's](harnesses/pi/harness.py)
- benchmark: `benchmarks/<name>/benchmark.py` exposing `BENCHMARK`, like [SWE-bench Verified's](benchmarks/swebench_verified/benchmark.py)
- model server: `backends/<name>/backend.py` exposing `BACKEND`, like [splash's](backends/splash/backend.py)
- teacher: `teachers/<name>/teacher.py` exposing `TEACHER`, like [Copilot CLI's](teachers/copilot/teacher.py), plus a `[teachers.<name>]` table in [models.toml](models.toml)

A new model is just a `[models.<name>]` table in [models.toml](models.toml).
