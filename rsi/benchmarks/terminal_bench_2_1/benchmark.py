from rsi import folders, paths

harbor = folders.module(root=paths.benchmarks_dir(), name="harbor", file="harbor.py", kind="Harbor tasks")

# held out, i.e. the teacher never sees these tasks and they only measure a finished harness. Every task needs the
# internet because the graders install their test tools at grading time
BENCHMARK = harbor.HarborBenchmark(
    name="terminal-bench-2.1",
    sources=(
        harbor.Source(
            name="tb",
            kind="terminal tasks",
            host=harbor.HF,
            repo="harborframework/terminal-bench-2.1",
            revision="3e235dff6880252a587fa479c09bcb1e16edf2eb",
            folder="tasks",
        ),
    ),
    work="terminal tasks",
    terms=(
        "terminal-bench",
        "terminal bench",
        "tbench",
        "harbor",
        "benchmark",
        "/logs/verifier",
        "reward.txt",
    ),
)
