import base64
import gzip
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from rsi import cards, core, paths, runs, variants, viewer


def _assistant(stamp: int, thinking: str, calls: list[dict], text: str = "", stop: str = "toolUse") -> dict:
    content: list[dict] = [{"type": "thinking", "thinking": thinking}]
    if text:
        content.append({"type": "text", "text": text})
    content += [{"type": "toolCall", "id": c["id"], "name": c["name"], "arguments": c["args"]} for c in calls]
    message = {"role": "assistant", "content": content, "timestamp": stamp, "durationMs": 900, "stopReason": stop}
    message["usage"] = {"output": 50}
    return {"type": "message_end", "message": message}


def _result(call_id: str, text: str, error: bool = False) -> dict:
    message = {"role": "toolResult", "toolCallId": call_id, "isError": error, "durationMs": 300, "timestamp": 0}
    message["content"] = [{"type": "text", "text": text}]
    return {"type": "message_end", "message": message}


def _events(live_tail: bool = False) -> list[dict]:
    events = [
        {"type": "session", "id": "a__b-1"},
        {
            "type": "message_end",
            "message": {"role": "user", "content": [{"type": "text", "text": "Fix it"}], "timestamp": 1000},
        },
        _assistant(
            stamp=1010,
            thinking="Let me look at the code first. Then fix it.",
            calls=[{"id": "c1", "name": "bash", "args": {"command": "grep -rn bug src\nhead -5 x"}}],
        ),
        {"type": "message_update", "message": {"role": "assistant"}},
        _result(call_id="c1", text="src/a.py:3: bug"),
        {"type": "auto_retry_start", "attempt": 1, "errorMessage": "Request timed out."},
        _assistant(
            stamp=5000,
            thinking="Found it.",
            calls=[
                {
                    "id": "c2",
                    "name": "edit",
                    "args": {"path": "src/a.py", "edits": [{"oldText": "bug", "newText": "fix"}]},
                }
            ],
        ),
        _result(call_id="c2", text="edited", error=False),
        _assistant(stamp=8000, thinking="", calls=[], text="Fixed the bug.", stop="stop"),
    ]
    if live_tail:
        events.append({"type": "message_start", "message": {"role": "assistant", "timestamp": 9000}})
        events.append({"type": "message_update"})
    lines = [json.dumps(e) for e in events]
    return lines + (['{"type": "message_upd'] if live_tail else [])


def _node(
    name: str, parent: str | None = None, variant: str = "", resolved: dict[str, bool] | None = None
) -> core.Node:
    node = core.Node(
        name=name,
        parent=parent,
        dataset=core.DatasetSpec(name="swebench-verified", dataset_id="princeton-nlp/SWE-bench_Verified"),
        harness="pi",
        harness_version="1.4.0",
        variant=variant,
        model=core.ModelSpec(name="qwen-3.8-27b", source="splash:x/y", model_id="q"),
        budget=core.Budget(),
        tasks=2,
        seed=42,
        task_ids=["a__b-1", "c__d-2"],
    )
    for iid, ok in (resolved or {}).items():
        node.runs[iid] = core.TaskRun(
            instance_id=iid, status="ok", seconds=60.0, resolved=ok, patch="diff --git a/x b/x\n+y\n"
        )
        node.runs[iid].metrics = core.TaskMetrics(requests=3, output_tokens=150, patch_lines=1, stop_reason="stop")
    if resolved:
        node.resolved, node.total = sum(resolved.values()), len(resolved)
    runs.write_node(node=node)
    for iid in node.runs:
        folder = paths.instances_dir(node_name=name) / iid
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "problem.txt").write_text(f"```\nThe bug in {iid}\nmore\n")
        (folder / "events.jsonl").write_text("\n".join(_events()))
    return node


@pytest.fixture
def lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, core.Node]:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    root = _node(name="root", resolved={"a__b-1": False, "c__d-2": True})
    package = paths.variants_dir() / "cap"
    package.mkdir(parents=True)
    (package / "package.json").write_text("{}")
    (package / "settings.json").write_text('{"thinkingBudgets": {"medium": 100}}\n')
    card = cards.VariantCard(title="Cap thinking", hypothesis="h", evidence="e", predicts="p")
    variants.tree_register(name="cap", parent="baseline", parent_node="root", card=card, diff="+ settings.json")
    waiting = cards.VariantCard(title="Not run yet", hypothesis="h")
    variants.tree_register(name="later", parent="baseline", parent_node="root", card=waiting, diff="+ x")
    repeat = _node(name="repeat", parent="root", resolved={"a__b-1": False, "c__d-2": True})
    variant = _node(name="variant", parent="root", variant=str(package), resolved={"a__b-1": True, "c__d-2": True})
    return {"root": root, "repeat": repeat, "variant": variant}


def test_the_tree_is_harnesses_with_their_runs_and_how_each_did(lineage: dict[str, core.Node]) -> None:
    (found,) = viewer.tree_data()["lineages"]
    assert (found["model"], found["harness"], found["root"]["name"]) == ("qwen-3.8-27b", "pi 1.4.0", "root")
    assert (found["made"], found["ran"], found["better"], found["probe"], found["best"]) == (2, 1, 0, None, None)
    base = found["tree"]
    assert (base["name"], base["title"]) == ("baseline", "pi 1.4.0, no changes")
    assert [r["name"] for r in base["runs"]] == ["root", "repeat"]
    cap, later = base["children"]
    assert (cap["title"], cap["href"]) == ("Cap thinking", "#node/variant")
    assert [r["name"] for r in cap["runs"]] == ["variant"]
    assert (cap["solved"], cap["of"]) == (2.0, 2)
    assert cap["standing"]["solves"]["gain"] == 1.0 and cap["standing"]["reference_runs"] == 2
    assert cap["standing"]["better_by"] == ""
    assert (later["runs"], later["href"], later["standing"]) == ([], "#variant/later", None)


def test_the_tree_names_the_variant_that_solved_the_most_on_the_probe_set(lineage: dict[str, core.Node]) -> None:
    runs.probe_path(root=lineage["root"]).write_text("# probe set\na__b-1\nc__d-2\n")
    runs.hidden_path(root=lineage["root"]).write_text("a__b-1\n")
    (found,) = viewer.tree_data()["lineages"]
    assert found["probe"] == {"tasks": 2, "hidden": 1}
    assert (found["tree"]["solved"], found["tree"]["scored"]) == (1.0, 2)
    best = found["best"]
    assert (best["name"], best["solved"], best["baseline"], best["solves"]["gain"]) == ("cap", 2.0, 1.0, 1.0)


def test_a_run_shows_its_numbers_next_to_its_parents_runs(lineage: dict[str, core.Node]) -> None:
    data = viewer.node_data(name="variant")
    assert (data["label"], data["parent_label"], data["harness"]) == ("cap s42", "baseline s42", "pi 1.4.0")
    assert data["reference"]["runs"] == ["baseline s42", "baseline s42"] and data["reference"]["solved"] == 1.0
    rows = {r["iid"]: r for r in data["tasks"]}
    assert (rows["a__b-1"]["flip"], rows["a__b-1"]["parent"], rows["a__b-1"]["state"]) == ("fixed", [0, 2], "passed")
    assert (rows["c__d-2"]["flip"], rows["c__d-2"]["parent"]) == ("", [2, 2])
    assert rows["a__b-1"]["brief"] == "The bug in a__b-1"
    assert data["card"]["hypothesis"] == "h" and "thinkingBudgets" in data["diff"]
    assert data["card"]["result"][:3] == [
        "from root",
        "1 run, 2.0/2 solved per run",
        "solved +1.0 ± 0.7 tasks per run (z 1.41; 1 run vs 2 of the parent, 2 tasks): within noise",
    ]
    repeat = viewer.node_data(name="repeat")
    assert (repeat["card"], repeat["diff"]) == (None, "")
    assert [r["name"] for r in repeat["harness_runs"]] == ["root"]
    assert [c["name"] for c in viewer.node_data(name="root")["children"]] == ["cap", "later"]


def test_a_task_links_to_the_same_task_in_every_other_run(lineage: dict[str, core.Node]) -> None:
    others = viewer.task_data(node_name="repeat", iid="a__b-1")["others"]
    shown = [(o["name"], o["outcome"], o["same"]) for o in others]
    assert shown == [("root", "failed", True), ("variant", "passed", False)]


def test_a_task_that_is_still_running_shows_live(lineage: dict[str, core.Node]) -> None:
    folder = paths.instances_dir(node_name="repeat") / "c__d-2"
    node = runs.read_node(name="repeat")
    del node.runs["c__d-2"]
    node.resolved = node.total = 0
    runs.write_node(node=node)
    (folder / "events.jsonl").write_text("\n".join(_events(live_tail=True)))
    data = viewer.node_data(name="repeat")
    assert data["status"] == "running" and data["live"] == ["c__d-2"]
    assert {r["iid"]: r["state"] for r in data["tasks"]}["c__d-2"] == "running"
    task = viewer.task_data(node_name="repeat", iid="c__d-2")
    assert task["finished"] is False and task["inflight_at"] == 9000


def test_a_trajectory_pairs_each_call_with_its_result(lineage: dict[str, core.Node]) -> None:
    data = viewer.task_data(node_name="variant", iid="a__b-1")
    assert data["user"] == "Fix it"
    kinds = [item["kind"] for item in data["items"]]
    assert kinds == ["step", "note", "step", "step"]
    first, _, edit, last = data["items"]
    assert first["intent"] == "Let me look at the code first."
    assert first["t"] == 10 and first["calls"][0]["label"] == "grep -rn bug src"
    assert first["calls"][0]["out"] == "src/a.py:3: bug" and first["calls"][0]["ok"] is True
    assert edit["calls"][0]["args"] == {"edits": [{"old": "bug", "new": "fix"}]}
    assert last["text"] == "Fixed the bug." and last["calls"] == []
    assert data["finished"] is True and data["verdict"]["dot"] in ("ok", "pend")
    assert data["patch"]["files"][0]["path"] == "x"
    assert data["metrics"]["requests"] == 3 and data["timing"] == [] and data["told"] == ""


def test_a_task_shows_what_the_model_was_told_and_how_long_each_response_took(
    lineage: dict[str, core.Node],
) -> None:
    lines = [
        {"type": "context", "ts": 1000, "system": "You are pi.", "tools": [{"name": "bash"}]},
        {"ts": 1000, "system_tokens": 10, "tool_def_tokens": 20, "message_tokens": 30, "params": {}},
        {"type": "timing", "ts": 1900, "ttft_ms": 400, "gen_ms": 900, "think_ms": 300, "stop_reason": "toolUse"},
        {"type": "timing", "ts": 5900, "ttft_ms": 500, "gen_ms": 900, "think_ms": 0, "stop_reason": "stop"},
        {"ts": 5000, "system_tokens": 10, "tool_def_tokens": 20, "message_tokens": 90, "params": {}},
    ]
    folder = paths.instances_dir(node_name="variant") / "a__b-1"
    (folder / "metrics.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + '\n{"ts": 99')
    data = viewer.task_data(node_name="variant", iid="a__b-1")
    assert "You are pi." in data["told"] and '"bash"' in data["told"]
    assert data["timing"] == [
        {"ttft": 400, "total": 900, "think": 300, "context": 60, "stop": "toolUse"},
        {"ttft": 500, "total": 900, "think": 0, "context": 120, "stop": "stop"},
    ]


def test_tasks_across_runs_line_up_every_run_that_covers_a_shared_task_set(lineage: dict[str, core.Node]) -> None:
    wide = _node(name="wide", resolved={"a__b-1": True, "c__d-2": True, "e__f-3": False})
    wide.task_ids = ["a__b-1", "c__d-2", "e__f-3"]
    runs.write_node(node=wide)
    (group,) = viewer.flips_data()["groups"]
    names = [n["name"] for n in group["nodes"]]
    assert sorted(names) == ["repeat", "root", "variant", "wide"]
    flipped, steady = group["tasks"]
    assert (flipped["iid"], flipped["flips"], steady["flips"]) == ("a__b-1", True, False)
    cells = dict(zip(names, flipped["cells"], strict=True))
    assert cells == {"root": "failed", "repeat": "failed", "variant": "passed", "wide": "passed"}
    assert {n["name"]: n["resolved"] for n in group["nodes"]}["wide"] == 2


def test_a_run_lines_up_with_a_run_of_more_tasks_that_covers_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RSI_ROOT", str(tmp_path))
    _node(name="probe", resolved={"a__b-1": False, "c__d-2": True})
    wide = _node(name="wide", resolved={"a__b-1": True, "c__d-2": True, "e__f-3": False})
    wide.task_ids = ["a__b-1", "c__d-2", "e__f-3"]
    runs.write_node(node=wide)
    (group,) = viewer.flips_data()["groups"]
    assert [n["name"] for n in group["nodes"]] == ["probe", "wide"] and [t["iid"] for t in group["tasks"]] == [
        "a__b-1",
        "c__d-2",
    ]


def test_names_from_the_url_cannot_leave_runs(lineage: dict[str, core.Node]) -> None:
    for bad in ("../root", "..", ".hidden", "a/b"):
        with pytest.raises(KeyError):
            viewer.task_data(node_name=bad, iid="a__b-1")
    with pytest.raises(KeyError):
        viewer.task_data(node_name="variant", iid="../../etc")


def test_the_server_answers_json(lineage: dict[str, core.Node]) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), viewer._Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base}/api/tree", timeout=5) as resp:
            assert json.loads(resp.read())["lineages"][0]["root"]["name"] == "root"
        with urllib.request.urlopen(f"{base}/api/flips", timeout=5) as resp:
            assert len(json.loads(resp.read())["groups"]) == 1
        with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
            assert b"<title>rsi</title>" in resp.read()
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"{base}/api/node?name=nope", timeout=5)
        assert missing.value.code == 404
    finally:
        server.shutdown()


def test_a_session_shows_what_its_checks_made_of_it(lineage: dict[str, core.Node]) -> None:
    round_dir = paths.node_dir(name="root") / "generations" / "20261027-163053"
    session = round_dir / "session-1-tools"
    (session / "out" / "variants").mkdir(parents=True)
    (session / "prompt.md").write_text("You are improving pi.")
    (session / "events.jsonl").write_text("\n".join(_events()))
    (round_dir / "checks" / "session-1-tools" / "cap").mkdir(parents=True)
    data = viewer.session_data(node_name="root", round_id="20261027-163053", session=1)
    assert data["instructions"] == "You are improving pi."
    assert data["outcome"]["status"] == "registered" and data["outcome"]["registered"] == ["cap"]
    rounds = viewer.node_data(name="root")["rounds"]
    assert rounds[0]["registered"] == 1 and rounds[0]["live"] is False
    assert (rounds[0]["sessions"][0]["n"], rounds[0]["sessions"][0]["area"]) == (1, "tools")


def test_parse_patch_counts_and_lines() -> None:
    diff = (
        "diff --git a/src/a.py b/src/a.py\n"
        "index 111..222 100644\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1,3 +1,3 @@\n"
        " ctx\n"
        "-old line\n"
        "+new line\n"
        "+added line\n"
        "diff --git a/src/b.py b/src/b.py\n"
        "--- a/src/b.py\n"
        "+++ b/src/b.py\n"
        "@@ -1,1 +1,2 @@\n"
        " x\n"
        "+y\n"
    )
    patch = viewer._parse_patch(text=diff)
    assert [f["path"] for f in patch["files"]] == ["src/a.py", "src/b.py"]
    assert (patch["files"][0]["added"], patch["files"][0]["removed"]) == (2, 1)
    assert patch["files"][0]["lines"] == [
        ["hunk", "@@ -1,3 +1,3 @@"],
        ["ctx", "ctx"],
        ["del", "old line"],
        ["add", "new line"],
        ["add", "added line"],
    ]
    assert patch["stats"] == "2 files · +3 −1"


def test_the_eval_report_reads_as_expected_against_actual(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    tests = {
        "FAIL_TO_PASS": {"success": ["pkg/test_a.py::t0"], "failure": ["pkg/test_a.py::t1"]},
        "PASS_TO_PASS": {"success": ["pkg/test_b.py::t2"], "failure": []},
    }
    report = {"t__x-1": {"resolved": False, "patch_successfully_applied": True, "tests_status": tests}}
    (eval_dir / "report.json").write_text(json.dumps(report))
    (eval_dir / "test_output.txt").write_text("FAILED pkg/test_a.py::t1\n= 1 failed, 430 passed =\nTest Exit Code: 1\n")
    evald = viewer._read_eval(task_dir=tmp_path)
    assert evald is not None and (evald["resolved"], evald["applied"], evald["infra"]) == (False, True, False)
    assert evald["f2p"] == {"total": 2, "passed": 1, "failed": ["pkg/test_a.py::t1"]}
    assert evald["p2p"] == {"total": 1, "passed": 1, "failed": []}
    assert "FAILED pkg/test_a.py::t1" in evald["log"] and "Test Exit Code: 1" in evald["log"]
    assert viewer._read_eval(task_dir=tmp_path / "nowhere") is None


def test_classify_ordering() -> None:
    ok = core.TaskRun(instance_id="a", status="ok", seconds=1.0, resolved=True)
    ok.metrics = core.TaskMetrics(requests=1, patch_lines=1)
    assert viewer._classify(run=ok, evald={"resolved": True, "infra": False, "applied": True})[0] == "ok"
    err = core.TaskRun(instance_id="a", status="error", seconds=1.0, error="boom", resolved=False)
    assert viewer._classify(run=err, evald=None) == ("err", "error: boom")
    empty = core.TaskRun(instance_id="a", status="ok", seconds=1.0, resolved=False)
    assert viewer._classify(run=empty, evald=None)[0] == "warn"


def test_save_puts_the_viewer_and_every_answer_it_asks_for_in_one_file(
    lineage: dict[str, core.Node], tmp_path: Path
) -> None:
    path = tmp_path / "share" / "rsi.html"
    size = viewer.save(path=path, node_names=["variant"])
    page = path.read_text()
    assert size == path.stat().st_size and "<title>rsi</title>" in page
    found = re.search(r'<script type="application/json" id="saved">(.*?)</script>', page, re.DOTALL)
    assert found is not None
    packed = json.loads(found.group(1))["pages"]
    pages = {key: json.loads(gzip.decompress(base64.b64decode(value))) for key, value in packed.items()}
    assert pages["/api/tree"]["lineages"][0]["tree"]["name"] == "baseline"
    assert pages["/api/node?name=root"]["label"] == "baseline s42" and "/api/flips" in pages
    assert pages["/api/variant?name=later"]["card"]["title"] == "Not run yet"
    assert pages["/api/task?node=variant&iid=a__b-1"]["user"] == "Fix it"
    assert "/api/task?node=root&iid=a__b-1" not in pages
    with pytest.raises(KeyError):
        viewer.save(path=path, node_names=["nope"])
