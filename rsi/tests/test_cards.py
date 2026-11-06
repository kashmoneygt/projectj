import subprocess
import sys
from pathlib import Path

from rsi import cards

GOOD = (
    "bash tool: keep the end of long output\n"
    "\n"
    "What: long output keeps its last 40 lines\n"
    "Why: output is cut from the end, so the agent never sees the result line;\n"
    "  it runs the command again in a__b-1 and c__d-2\n"
    "- **Expect**: fewer reruns\n"
    "Reuses: settle-guard, `think-brake`\n"
)


def test_parse() -> None:
    card = cards.parse(text=GOOD)
    assert card.title == "bash tool: keep the end of long output"
    assert card.what == "long output keeps its last 40 lines"
    assert card.why == (
        "output is cut from the end, so the agent never sees the result line; it runs the command again in a__b-1 "
        "and c__d-2"
    )
    assert card.expect == "fewer reruns"
    assert card.reuses == ("settle-guard", "think-brake")


def test_parse_reads_reuses_and_values_on_their_own_lines() -> None:
    for prose in ("(none)", "None (new idea)", "-", "—", "n/a", "none.", "none — new; the loader stages it"):
        assert cards.parse(text=f"t\nWhat: w\nReuses: {prose}\n").reuses == ()
    block = cards.parse(text="t\n\nWhat:\nA weakness.\nA change.\n\nReuses:\nnone — new idea\n")
    assert (block.what, block.reuses) == ("A weakness. A change.", ())
    reused = cards.parse(text="t\nWhat: w\nReuses: `settle-guard` (its hook), think-brake.\n")
    assert reused.reuses == ("settle-guard", "think-brake")
    aside = cards.parse(text="t\nWhat: w\nReuses: cut-rescue (its trigger, not its message, and its cap)\n")
    assert aside.reuses == ("cut-rescue",)
    card = cards.parse(text="t\nWhat: w\nReuses: none\n\n## Files changed\n- extensions/verify.ts: new hook\n")
    assert (card.what, card.reuses) == ("w", ())


def test_parse_takes_a_card_in_any_shape() -> None:
    essay = cards.parse(
        text=(
            "# baseline-snapshot\n\n**Area:** tools\n\n## Why\n\nMany runs stash.\n\nOthers retype.\n\n"
            "## What\n\nA snapshot tool.\n\n## Expect\n\nFewer stash cycles.\n"
        )
    )
    assert essay.title == "baseline-snapshot"
    assert essay.why == "Many runs stash. Others retype."
    assert essay.expect == "Fewer stash cycles."
    assert essay.what == "A snapshot tool."
    loose = cards.parse(text="verify before finishing\n\nRuns settle unverified, so check first.\n")
    assert (loose.title, loose.what) == ("verify before finishing", "Runs settle unverified, so check first.")
    assert len(cards.parse(text=f"{'word ' * 40}\n").title) == cards.TITLE_MAX
    assert cards.parse(text=f"{'word ' * 40}\n", title_max=None).title == ("word " * 40).strip()


def test_check_and_read(tmp_path) -> None:
    assert cards.check(package=tmp_path) == ["VARIANT.md is missing: write the card before handing the variant in"]
    assert cards.read(package=tmp_path) is None
    (tmp_path / cards.FILE).write_text("\n\n")
    assert cards.check(package=tmp_path) == ["VARIANT.md is empty: write the card before handing the variant in"]
    (tmp_path / cards.FILE).write_text(GOOD)
    assert cards.check(package=tmp_path) == []
    assert cards.read(package=tmp_path).reuses == ("settle-guard", "think-brake")
    package = tmp_path / "act-sooner"
    package.mkdir()
    (package / cards.FILE).write_text("What: w\n")
    assert cards.read(package=package).title == "act sooner"


def test_runs_alone_as_the_card_check_in_the_teacher_container(tmp_path) -> None:
    # -I -S hides site-packages, like the teacher's container
    def run(package: Path) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-I", "-S", cards.__file__, str(package)]
        return subprocess.run(command, capture_output=True, text=True, check=False)

    missing = run(package=tmp_path)
    assert missing.returncode == 1
    assert missing.stdout == "problem: VARIANT.md is missing: write the card before handing the variant in\n"
    (tmp_path / cards.FILE).write_text("baseline-snapshot\n\nWhat: a pristine copy of the repository.\n")
    good = run(package=tmp_path)
    assert (good.returncode, good.stdout) == (0, "ok: VARIANT.md is there\n")
    assert run(package=tmp_path / "missing").returncode == 2
