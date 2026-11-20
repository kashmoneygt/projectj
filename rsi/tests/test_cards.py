import json
import subprocess
import sys
from pathlib import Path

from rsi import cards

GOOD = {
    "title": "keep the end of long output",
    "what": "long output keeps its last 40 lines",
    "why": "output is cut from the end, so the agent never sees the result line and runs the command again",
    "expect": "fewer reruns",
    "reuses": ["settle-guard", "think-brake", "settle-guard"],
}


def test_a_card_is_parsed_once_it_has_no_problems() -> None:
    text = json.dumps(GOOD)
    assert cards.problems(text=text) == []
    card = cards.parse(text=text)
    assert (card.title, card.what, card.expect) == (GOOD["title"], GOOD["what"], GOOD["expect"])
    assert card.reuses == ("settle-guard", "think-brake")
    assert cards.parse(text=json.dumps({**GOOD, "reuses": []})).reuses == ()


def test_problems_say_what_a_card_is_missing() -> None:
    assert cards.problems(text="title: t\nwhat: w\n")[0].startswith("variant.json isn't valid JSON")
    assert cards.problems(text="[]") == ["variant.json should hold one JSON object"]
    assert cards.problems(text="[" * 100_000)[0].startswith("variant.json isn't valid JSON")
    # a card left half-written at the time limit is incomplete, so the variant isn't run
    draft = {"title": "a time limit", "what": "", "why": 3}
    assert cards.problems(text=json.dumps(draft)) == [
        'variant.json needs "what", as text',
        'variant.json needs "why", as text',
        'variant.json needs "expect", as text',
    ]
    odd = {**GOOD, "title": "x" * 80, "reuses": "settle-guard", "notes": "n"}
    assert cards.problems(text=json.dumps(odd)) == [
        'variant.json has "notes", which a card doesn\'t have',
        "the title is 80 characters long; keep it under 72",
        '"reuses" in variant.json should be a list of variant names',
    ]


def test_check_and_read(tmp_path: Path) -> None:
    assert cards.check(package=tmp_path) == ["variant.json is missing: write the card before handing the variant in"]
    assert cards.read(package=tmp_path) is None
    (tmp_path / cards.FILE).write_text('{"title": "t"}')
    assert cards.check(package=tmp_path) != [] and cards.read(package=tmp_path) is None
    (tmp_path / cards.FILE).write_text(json.dumps(GOOD))
    assert cards.check(package=tmp_path) == []
    assert cards.read(package=tmp_path).title == GOOD["title"]


def test_runs_alone_as_the_card_check_in_the_teacher_container(tmp_path: Path) -> None:
    # -I -S hides site-packages, like in the teacher's container where rsi isn't installed
    def run(package: Path) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-I", "-S", cards.__file__, str(package)]
        return subprocess.run(command, capture_output=True, text=True, check=False)

    missing = run(package=tmp_path)
    assert missing.returncode == 1
    assert missing.stdout == "problem: variant.json is missing: write the card before handing the variant in\n"
    (tmp_path / cards.FILE).write_text(json.dumps(GOOD))
    good = run(package=tmp_path)
    assert (good.returncode, good.stdout) == (0, "ok: variant.json is a complete card\n")
    assert run(package=tmp_path / "missing").returncode == 2
