# rsi

rsi: Recursive self-improvement of a coding-agent harness.

We run a local model in the [pi](https://github.com/earendil-works/pi) coding-agent harness on a coding benchmark. A teacher model then looks at how those runs went and writes changes to the harness, called variants. We run each variant on the same tasks, keep the ones that do better, and have the teacher write new variants from those. The goal is an evolved version of pi that makes the model better at agentic coding.

Currently, we have the following onboarded:

- **Harness**: pi 1.4.0. The baseline is pi with no changes, and variants change it through pi's own package system (extensions, system prompt, settings, skills), so we never fork pi
- **Models**: [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B) served by splash (MLX with speculative decoding) and [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B-it) served by llama.cpp, both with the vendor's recommended settings (see [models.toml](models.toml))
- **Benchmark**: [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified), graded by the official swebench evaluator
- **Teacher**: Claude Opus 5.5 at max reasoning effort, through GitHub Copilot CLI

## Run it

Needs [uv](https://docs.astral.sh/uv/), Docker, GitHub Copilot CLI logged in with `/login` (for the teacher), and splash (`brew install incoai/tap/splash`) or llama.cpp (`brew install llama.cpp`) to serve the model. splash serves models offline, so download Qwen once first with `splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`.

```sh
uv sync
uv run rsi models up qwen-3.8-27b

# build the Docker images for 100 tasks (only needed once)
uv run rsi images build --tasks 100 --clone-tasks
# or choose the tasks yourself, like a set spread across the benchmark's repos, with --instances <file> here and in run

# run the default pi harness (the baseline) on those 100 tasks and score it. takes about 6 hours
uv run rsi run --model qwen-3.8-27b --benchmark swebench-verified --tasks 100

# pick the probe set from that run: the tasks it failed plus 30 it passed, to catch regressions. the teacher never sees half of them
uv run rsi probe --root qwen-3.8-27b--pi--swebench-verified--baseline--100t--s42
# or carry a probe set over from an earlier lineage, with the tasks it hides: run the baseline on those tasks first
# (--instances probe.txt --seed 41, so its name doesn't clash with the probe runs below)
uv run rsi probe --root <that run> --tasks probe.txt --hidden hidden.txt

# run the baseline on the probe set a couple of times, since results change from run to run
uv run rsi run --model qwen-3.8-27b --benchmark swebench-verified --probe
uv run rsi run --model qwen-3.8-27b --benchmark swebench-verified --probe --seed 43

# the teacher writes up to 4 variants from a run, one session after another (up to an hour each)
uv run rsi variants generate --parent qwen-3.8-27b--pi--swebench-verified--baseline--100t--s42

# run each variant on the probe set (3 to 4 hours each, two tasks at a time), then see how it did against its parent
uv run rsi run --variant <name> --probe
uv run rsi variants list
```

Each run is saved as a node in `runs/<model>--<harness>--<benchmark>--<variant>--<tasks>t--s<seed>/`, with everything needed to understand and reproduce it. Once a variant is better than its parent (see below), generate the next variants from one of its runs with `--parent <node>`.

To see the variant tree, every run and task trajectory, and the teacher sessions as they happen, run `uv run rsi view --open`. `uv run rsi view --save rsi.html` saves the same page as one HTML file you can share.

For help with any command, run `uv run rsi <command> --help`.

## Judging variants

The model samples at temperature 1.0, so two runs of the same harness can differ by 3 or 4 tasks on the probe set, and a single run proves nothing. `rsi variants list` compares all of a variant's runs with all of its parent's runs on the same tasks, task by task, and only calls a variant better if:

- it solves more tasks beyond noise (z ≥ 2), and also more of the tasks hidden from the teacher, or
- it solves as many tasks with clearly fewer tokens (z ≥ 2)

Before building on a variant, run it once or twice more (`--seed 43`), since with four variants a round one of them can look better by luck.

To compare an evolved harness with the baseline at the end, run both on new tasks that no run so far has seen (`--instances <file>`), then `uv run rsi compare --a <baseline node> --b <evolved node>` (repeat `--a` and `--b` for more runs).

## How it works

- **Variants**: each variant is a folder in `variants/<name>` with its pi package and a short card (`VARIANT.md`) saying what it changes and why. `variants/tree.json` lists every variant and the run it was made from. [CONTRACT.md](harnesses/CONTRACT.md) and [HARNESS.md](harnesses/pi/HARNESS.md) say what a variant can change. It can't change the model, its settings or how we measure runs
- **Teacher**: each session gets [variant-generator.md](prompts/variant-generator.md) and one area of the harness from [areas.md](prompts/areas.md), and writes at most one variant. The teacher runs in Copilot CLI on the host machine, but its tools only work inside a Docker container with the parent's runs, pi's source code, and commands to check and try out a variant. That container has no internet access, so the teacher can't look up the tasks' real fixes. Before a variant is kept, the loop checks its files and asks the agent a question that needs no change. A variant that keeps the agent from simply answering, by making it go on or change files, is turned down
- **Sandbox**: the agent runs in each task's Docker image and can only reach the model server, through a forwarder that logs every request. This stops the agent from downloading the real fix, and lets us check every request against the pinned model settings

## Plug-in support

The harness, benchmark, model server and teacher are all plug-ins. Each one is a folder with a file that implements a protocol in [src/rsi/](src/rsi/):

- harness: `harnesses/<name>/harness.py` exposing `HARNESS`, like [pi's](harnesses/pi/harness.py)
- benchmark: `benchmarks/<name>/benchmark.py` exposing `BENCHMARK`, like [SWE-bench Verified's](benchmarks/swebench_verified/benchmark.py)
- model server: `backends/<name>/backend.py` exposing `BACKEND`, like [splash's](backends/splash/backend.py)
- teacher: `teachers/<name>/teacher.py` exposing `TEACHER`, like [Copilot CLI's](teachers/copilot/teacher.py), plus a `[teachers.<name>]` table in [models.toml](models.toml)

A new model is just a `[models.<name>]` table in [models.toml](models.toml).
