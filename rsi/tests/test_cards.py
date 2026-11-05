import subprocess
import sys
from pathlib import Path

from rsi import cards

GOOD = (
    "# Force one action turn after a capped thinking-only turn\n"
    "\n"
    "hypothesis: Runs die when a turn is cut at the output cap with no tool call;\n"
    "  forcing one more request recovers them.\n"
    "evidence: a__b-1, c__d-2 end with a capped thinking-only turn\n"
    "- **predicts**: fewer runs end without a patch\n"
    "ports: settle-guard, `think-brake`\n"
)


def test_parse() -> None:
    card = cards.parse(text=GOOD)
    assert card.title == "Force one action turn after a capped thinking-only turn"
    assert card.hypothesis == (
        "Runs die when a turn is cut at the output cap with no tool call; forcing one more request recovers them."
    )
    assert card.evidence.startswith("a__b-1")
    assert card.predicts == "fewer runs end without a patch"
    assert card.ports == ("settle-guard", "think-brake")


def test_parse_reads_ports_and_values_on_their_own_lines() -> None:
    for prose in ("(none)", "None (new idea)", "-", "—", "n/a", "none.", "none — new; the loader stages it"):
        assert cards.parse(text=f"# T\nhypothesis: h\nports: {prose}\n").ports == ()
    block = cards.parse(text="# T\n\nhypothesis:\nA weakness.\nA change.\n\nports:\nnone — new idea\n")
    assert (block.hypothesis, block.ports) == ("A weakness. A change.", ())
    ported = cards.parse(text="# T\nhypothesis: h\nports: `settle-guard` (its hook), think-brake.\n")
    assert ported.ports == ("settle-guard", "think-brake")
    aside = cards.parse(text="# T\nhypothesis: h\nports: cut-rescue (its trigger, not its message, and its cap)\n")
    assert aside.ports == ("cut-rescue",)
    card = cards.parse(text="# T\nhypothesis: h\nports: none\n\n## Files changed\n- extensions/verify.ts: new hook\n")
    assert (card.hypothesis, card.ports) == ("h", ())


def test_parse_takes_a_card_in_any_shape() -> None:
    essay = cards.parse(
        text=(
            "# baseline-snapshot\n\n**Lane:** contrast\n\n## Evidence\n\nMany runs stash.\n\nOthers retype.\n\n"
            "## Change\n\nA snapshot tool.\n\n## Prediction\n\nFewer stash cycles.\n"
        )
    )
    assert essay.title == "baseline-snapshot"
    assert essay.evidence == "Many runs stash. Others retype."
    assert essay.predicts == "Fewer stash cycles."
    assert essay.hypothesis == "**Lane:** contrast A snapshot tool."
    loose = cards.parse(text="Verify before finishing\n\nRuns settle unverified, so check first.\n")
    assert (loose.title, loose.hypothesis) == ("Verify before finishing", "Runs settle unverified, so check first.")
    assert len(cards.parse(text=f"{'word ' * 40}\n").title) == cards.TITLE_MAX
    assert cards.parse(text=f"{'word ' * 40}\n", title_max=None).title == ("word " * 40).strip()


def test_check_and_read(tmp_path) -> None:
    assert cards.check(package=tmp_path) == ["VARIANT.md is missing: write the card before handing the variant in"]
    assert cards.read(package=tmp_path) is None
    (tmp_path / cards.FILE).write_text("\n\n")
    assert cards.check(package=tmp_path) == ["VARIANT.md is empty: write the card before handing the variant in"]
    (tmp_path / cards.FILE).write_text(GOOD)
    assert cards.check(package=tmp_path) == []
    assert cards.read(package=tmp_path).ports == ("settle-guard", "think-brake")
    package = tmp_path / "act-sooner"
    package.mkdir()
    (package / cards.FILE).write_text("hypothesis: h\n")
    assert cards.read(package=package).title == "act sooner"


def test_runs_alone_as_the_card_check_in_the_teacher_container(tmp_path) -> None:
    # -I -S hides site-packages, like the teacher's container
    def run(package: Path) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-I", "-S", cards.__file__, str(package)]
        return subprocess.run(command, capture_output=True, text=True, check=False)

    missing = run(package=tmp_path)
    assert missing.returncode == 1
    assert missing.stdout == "problem: VARIANT.md is missing: write the card before handing the variant in\n"
    (tmp_path / cards.FILE).write_text("# baseline-snapshot\n\nA pristine copy of the repository.\n")
    good = run(package=tmp_path)
    assert (good.returncode, good.stdout) == (0, "ok: VARIANT.md is there\n")
    assert run(package=tmp_path / "missing").returncode == 2
