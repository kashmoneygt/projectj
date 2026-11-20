import functools
from pathlib import Path

from rsi import folders, paths

harbor = folders.module(root=paths.benchmarks_dir(), name="harbor", file="harbor.py", kind="Harbor tasks")

# held out, i.e. the teacher never sees these tasks and they only measure a finished harness. The agent works offline
# but grading gets the internet, since a few Go tests fetch modules
BENCHMARK = harbor.HarborBenchmark(
    name="swebench-pro",
    sources=(
        harbor.Source(
            name="pro",
            kind="bug fixes",
            host=harbor.GITHUB,
            repo="scaleapi/SWE-bench_Pro-os",
            revision="31a20c751b0bd56db6510ce7d4d3c269c5b7c72b",
            folder="v2/tasks",
            pick=functools.partial(harbor.listed, path=Path(__file__).with_name("hard51.txt")),
            repo_of=r"^instance_([^-]+)-",
        ),
    ),
    work="issues from public projects",
    terms=("swebench", "swe-bench", "harbor", "benchmark", "/logs/verifier", "reward.txt", "fail_to_pass"),
)
