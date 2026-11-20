import functools
import re
import zlib
from pathlib import Path

import datasets

from rsi import folders, paths

harbor = folders.module(root=paths.benchmarks_dir(), name="harbor", file="harbor.py", kind="Harbor tasks")

# SWE-rebench V2.1 - real GitHub issues graded by each project's own tests, with quality labels from an LLM review
REBENCH = "ibragim-bad/SWE-rebench-V2.1-harbor"
REBENCH_REVISION = "0e0ee91c0e8b09472167ed6edc36dd22dd8e4c09"
REBENCH_ROWS = "ibragim-bad/SWE-rebench-V2.1"  # same tasks as a table, which has the labels
REBENCH_ROWS_REVISION = "26c0145ded93168b0d5d095420155901ad2d5a2f"
HARBOR_DATASETS = "harborframework/harbor-datasets"
HARBOR_DATASETS_REVISION = "bae16d3bc8b10aab97018e8327cb4db4806d6033"

LANGUAGES = {"python", "go", "js", "ts"}  # tests in these languages build and run fine in a small Docker VM
LICENSES = {"MIT", "MIT-0", "Apache-2.0", "BSD", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Apache-2.0 OR MIT"}
BUGS = {"minor_bug", "edge_case_bug", "regression_bug", "major_bug", "critical_bug", "core_bug"}
FEATURES = {"core_feat", "integration_feat", "ui_ux_feat"}
# hold out SWE-bench Pro's repos, since we use SWE-bench Pro to test the finished harness
HELD_OUT = {
    "ansible/ansible",
    "element-hq/element-web",
    "flipt-io/flipt",
    "future-architect/vuls",
    "gravitational/teleport",
    "internetarchive/openlibrary",
    "navidrome/navidrome",
    "nodebb/nodebb",
    "protonmail/webclients",
    "qutebrowser/qutebrowser",
    "tutao/tutanota",
}
# regex for files in a fix that aren't code (docs, changelogs, CI config, lock files), same as in where/tests/grade.py
NOT_CODE = re.compile(
    r"(^|/)(docs?|changelog|changes|news|\.github)(/|\.|$)|\.(md|rst|txt|adoc)$"
    r"|(^|/)(package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|pnpm-lock\.yaml|Pipfile\.lock|poetry\.lock|uv\.lock)$"
    r"|(^|/)(go\.(mod|sum)|Cargo\.lock)$",
    re.IGNORECASE,
)
HARDWARE = r"cuda|gpu|optix|metal_|vulkan|opengl|jtag|fpga|tpu_"  # skip tasks that need a gpu, fpga, etc

ISSUE = "Please resolve this issue:\n\n{task}"
WHERE = (
    "Someone filed this issue:\n\n{task}\n\n"
    "Don't fix it. Find where in the code it comes from and tell me which files a fix would need to change, without "
    "editing anything."
)

# skip these repos because their tests need the network, run out of memory in our 6 GB Docker VM, or take too long
UNRUNNABLE = {
    "alteryx/woodwork": "tests fail even with the real fix",
    "aws-cloudformation/cfn-lint": "needs the network (DNS lookup)",
    "chainguard-dev/apko": "a test fails even with the real fix",
    "copier-org/copier": "a test fails even with the real fix",
    "google/jax": "out of memory",
    "jax-ml/jax": "out of memory",
    "jeremydaly/dynamodb-toolbox": "grading takes 20+ min",
    "kubernetes-sigs/cloud-provider-azure": "tests don't run offline",
    "microsoft/griffel": "a test fails even with the real fix",
    "modin-project/modin": "tests exit without running when offline",
    "obsidian-tasks-group/obsidian-tasks": "a test fails even with the real fix",
    "oclif/core": "out of memory",
    "ony3000/prettier-plugin-classnames": "tests fail even with the real fix",
    "peak/s5cmd": "tests fail even with the real fix",
    "projectevergreen/wcc": "needs the network",
    "pyccel/pyccel": "grading takes 10+ min",
    "pypa/pipx": "needs the network (installs packages)",
    "redocly/redocly-cli": "a test fails even with the real fix",
    "researchobject/ro-crate-py": "needs the network (downloads files)",
    "scikit-build/scikit-build-core": "needs the network (installs packages)",
    "tanstack/router": "out of memory",
    "unjs/h3": "grading hangs",
    "vuejs/eslint-plugin-vue": "grading hangs",
}


@functools.cache
def _issues() -> dict[str, list[str]]:
    """SWE-rebench task ids for each kind (fix, build, where), picked by quality labels and patch size"""
    rows = datasets.load_dataset(REBENCH_ROWS, split="train", revision=REBENCH_ROWS_REVISION)
    picked: dict[str, list[str]] = {"fix": [], "build": [], "where": []}
    for row in rows.select_columns(
        [
            "instance_id",
            "repo",
            "language",
            "license",
            "created_at",
            "meta",
            "patch",
            "problem_statement",
            "quality_flags",
        ]
    ):
        labels = (row["meta"] or {}).get("llm_metadata") or {}
        if (
            row["language"] not in LANGUAGES
            or row["license"] not in LICENSES
            or row["repo"].lower() in HELD_OUT
            or row["repo"].lower() in UNRUNNABLE
            # skip issues before 2022, since the model may have trained on them
            or row["created_at"] < "2022"
            or row["quality_flags"]
            or labels.get("code") != "A"
            or labels.get("intent_completeness") != "complete"
        ):
            continue
        patch = row["patch"]
        code = [f for f in re.findall(r"^diff --git a/(\S+) b/", patch, re.MULTILINE) if not NOT_CODE.search(f)]
        lines = sum(line[:1] in "+-" and line[:3] not in ("+++", "---") for line in patch.splitlines() if line)
        categories = set(labels.get("pr_categories") or ())
        # only multi-file changes, since the baseline solved ~90% of small ones and the difficulty labels don't predict
        # which tasks it fails
        if categories & FEATURES and not categories & BUGS and 3 <= len(code) <= 8 and 80 <= lines <= 400:
            picked["build"].append(row["instance_id"])
        elif categories & BUGS and not categories & FEATURES and 2 <= len(code) <= 5 and 30 <= lines <= 200:
            # split bugs 50/50 between fix and where tasks, so no issue is used for both
            if zlib.crc32(row["instance_id"].encode()) % 2 == 0:
                picked["fix"].append(row["instance_id"])
            elif _answerable(patch=patch, code=code, issue=row["problem_statement"]):
                picked["where"].append(row["instance_id"])
    return picked


def _answerable(patch: str, code: list[str], issue: str) -> bool:
    """True if every code file the fix touches already exists and the issue doesn't name any of them"""
    created = re.findall(r"^diff --git a/(\S+) b/\S+\nnew file mode", patch, re.MULTILINE)
    return not set(code) & set(created) and not any(Path(f).name in issue for f in code)


def _rebench(kind: str, label: str, request: str, grader: str = "") -> harbor.Source:
    return harbor.Source(
        name=f"rebench-{kind}",
        kind=label,
        host=harbor.HF,
        repo=REBENCH,
        revision=REBENCH_REVISION,
        folder="tasks",
        # offline since the images already have each project's dependencies
        internet=False,
        pick=lambda: _issues()[kind],
        repo_of=r"^[^_]+__(.+)-\d+$",
        request=request,
        grader=grader,
    )


# proxy for coding agent tasks, with an equal share of each: (1) bug fixing, (2) feature adding, (3) questions about
# where a bug is in the code, (4) terminal work, (5) data analysis
BENCHMARK = harbor.HarborBenchmark(
    name="coding-mix",
    sources=(
        _rebench(kind="fix", label="bug fixes", request=ISSUE),
        _rebench(kind="build", label="features", request=ISSUE),
        _rebench(kind="where", label="code questions", request=WHERE, grader=str(Path(__file__).with_name("where"))),
        harbor.Source(
            name="termigen",
            kind="terminal tasks",
            host=harbor.GITHUB,
            repo="ucsb-mlsec/terminal-bench-env",
            revision="dc329464161db64b0c670f46fa39b62e4719dddd",
            folder="environments_harbor",
            # offline since the images already have everything the tasks and graders need
            internet=False,
            leave_out=HARDWARE,
            # treat a topic's easy/medium/hard versions as one repo
            repo_of=r"^(.+?)_(?:easy|medium|hard)(?:_\d+)?$",
        ),
        harbor.Source(
            name="dacode",
            kind="data analysis",
            host=harbor.HF,
            repo=HARBOR_DATASETS,
            revision=HARBOR_DATASETS_REVISION,
            folder="datasets/dacode",
            internet=False,
            # skip ML training tasks, which need more compute than our containers have
            leave_out=r"^dacode-ml-",
            repo_of=r"^dacode-(.+)-\d+$",
        ),
        harbor.Source(
            name="bird",
            kind="data analysis",
            host=harbor.HF,
            repo=HARBOR_DATASETS,
            revision=HARBOR_DATASETS_REVISION,
            folder="datasets/bird-bench",
            internet=False,
            repo_of=r"^(.+)__\d+$",
        ),
    ),
    work=(
        "a mix of coding work: resolving bug reports and feature requests from the issue trackers of Python, Go, "
        "JavaScript and TypeScript projects, finding where in a project's code a reported problem comes from without "
        "changing anything, terminal tasks like debugging services and builds and setting up systems, and data "
        "analysis with Python and SQL"
    ),
    terms=(
        "swe-rebench",
        "rebench",
        "swe-bench",
        "swe_bench",
        "swebench",
        "termigen",
        "terminal-bench",
        "terminal_bench",
        "terminal bench",
        "tbench",
        "dacode",
        "da-code",
        "da_code",
        "bird-bench",
        "bird bench",
        "harbor",
        "benchmark",
        "hidden tests",
        "fail_to_pass",
        "pass_to_pass",
        "/logs/verifier",
        "reward.txt",
        "reward.json",
        "test_outputs",
    ),
)
