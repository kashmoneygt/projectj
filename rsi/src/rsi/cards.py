import dataclasses
import json
import re
import sys
from pathlib import Path

FILE = "variant.json"  # never reaches the agent
TITLE_MAX = 72
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")  # variant name
TEXT = ("title", "what", "why", "expect")


@dataclasses.dataclass(frozen=True)
class VariantCard:
    title: str
    what: str
    why: str = ""
    expect: str = ""
    reuses: tuple[str, ...] = ()  # variants whose ideas it builds on


def problems(text: str) -> list[str]:
    """What's wrong with a card's text, if anything"""
    try:
        card = json.loads(text)
    except (ValueError, RecursionError) as exc:
        return [f"{FILE} isn't valid JSON ({exc})"]
    if not isinstance(card, dict):
        return [f"{FILE} should hold one JSON object"]
    blank = [key for key in TEXT if not isinstance(card.get(key), str) or not card[key].strip()]
    found = [f'{FILE} needs "{key}", as text' for key in blank]
    found += [f'{FILE} has "{key}", which a card doesn\'t have' for key in card if key not in (*TEXT, "reuses")]
    title = card.get("title")
    if isinstance(title, str) and len(title.strip()) > TITLE_MAX:
        found.append(f"the title is {len(title.strip())} characters long; keep it under {TITLE_MAX}")
    reuses = card.get("reuses", [])
    if not isinstance(reuses, list) or not all(isinstance(name, str) for name in reuses):
        found.append(f'"reuses" in {FILE} should be a list of variant names')
    return found


def parse(text: str) -> VariantCard:
    """A card from text that has no problems"""
    card = json.loads(text)
    return VariantCard(
        title=card["title"].strip(),
        what=card["what"].strip(),
        why=card["why"].strip(),
        expect=card["expect"].strip(),
        reuses=tuple(dict.fromkeys(card.get("reuses", []))),
    )


def read(package: Path) -> VariantCard | None:
    """A package's card, or None if it has none or it has problems"""
    file = package / FILE
    if not file.is_file():
        return None
    text = file.read_text(errors="replace")
    return None if problems(text=text) else parse(text=text)


def check(package: Path) -> list[str]:
    """What's wrong with a package's card, if anything"""
    file = package / FILE
    if not file.is_file():
        return [f"{FILE} is missing: write the card before handing the variant in"]
    return problems(text=file.read_text(errors="replace"))


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not Path(argv[1]).is_dir():
        print(f"usage: {Path(argv[0]).name} <package dir>")
        return 2
    found = check(package=Path(argv[1]))
    for problem in found:
        print(f"problem: {problem}")
    if not found:
        print(f"ok: {FILE} is a complete card")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
