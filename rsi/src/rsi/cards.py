import dataclasses
import re
import sys
from pathlib import Path

FILE = "VARIANT.md"  # never reaches the agent
TITLE_MAX = 100

NAME = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")  # variant name
_KEY = r"(hypothesis|evidence|predicts?|predictions?|ports?)"
# a field line like "hypothesis: ...", with an optional bullet or bold
_FIELD_LINE = re.compile(rf"^[-*\s]*\**\s*{_KEY}\s*\**\s*:\s*\**\s*(.*)$", re.IGNORECASE)
# or a heading like "## Evidence"
_FIELD_HEADING = re.compile(rf"^#+\s*\**\s*{_KEY}\s*\**\s*:?\s*$", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class VariantCard:
    title: str
    hypothesis: str  # what changes and why
    evidence: str = ""  # what in the runs shows the cause
    predicts: str = ""  # expected effect on behavior, results and cost
    ports: tuple[str, ...] = ()  # variants whose ideas it reuses


def _field_name(key: str) -> str:
    key = key.lower()
    return "predicts" if key.startswith("predict") else "ports" if key.startswith("port") else key


def parse(text: str) -> VariantCard:
    """Parse a card in any format, using loose text as the hypothesis if it has none"""
    title = ""
    parts: dict[str, list[str]] = {}
    loose: list[str] = []
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line_match, heading_match = _FIELD_LINE.match(line), _FIELD_HEADING.match(line)
        if line_match or heading_match:
            current = _field_name(key=(line_match or heading_match).group(1))
            parts.setdefault(current, [])
            if line_match and line_match.group(2):
                parts[current].append(line_match.group(2))
        elif not title:
            title = line.lstrip("#").strip().strip("*").strip()
        elif line.startswith("#"):
            current = ""
        elif current:
            parts[current].append(line)
        else:
            loose.append(line)
    fields = {key: " ".join(values) for key, values in parts.items()}
    if len(title) > TITLE_MAX:
        title = title[: TITLE_MAX - 1].rstrip() + "…"
    return VariantCard(
        title=title,
        hypothesis=fields.get("hypothesis") or " ".join(loose),
        evidence=fields.get("evidence", ""),
        predicts=fields.get("predicts", ""),
        ports=_port_names(text=fields.get("ports", "")),
    )


def read(package: Path) -> VariantCard | None:
    file = package / FILE
    if not file.is_file():
        return None
    card = parse(text=file.read_text(errors="replace"))
    return card if card.title else dataclasses.replace(card, title=package.name.replace("-", " ").replace("_", " "))


def check(package: Path) -> list[str]:
    """Check that the card exists and isn't empty"""
    file = package / FILE
    if not file.is_file():
        return [f"{FILE} is missing: write the card before handing the variant in"]
    if not file.read_text(errors="replace").strip():
        return [f"{FILE} is empty: write the card before handing the variant in"]
    return []


def _port_names(text: str) -> tuple[str, ...]:
    """Variant names in a ports field, ignoring asides and 'none'"""

    def first_word(part: str) -> str:
        words = part.split()
        return words[0].strip("`'\"()[].:;—-").lower() if words else ""

    if first_word(part=text) in ("", "none", "n/a", "na", "no"):
        return ()
    names: list[str] = []
    for part in re.sub(r"\([^)]*\)", "", text).split(","):
        word = first_word(part=part)
        if NAME.match(word) and "--" not in word and word not in names:
            names.append(word)
    return tuple(names)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not Path(argv[1]).is_dir():
        print(f"usage: {Path(argv[0]).name} <package dir>")
        return 2
    found = check(package=Path(argv[1]))
    for problem in found:
        print(f"problem: {problem}")
    if not found:
        print(f"ok: {FILE} is there")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
