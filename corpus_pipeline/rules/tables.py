"""Load reviewed TSV corrections and apply whole-phrase replacements."""

import csv
import re
from pathlib import Path


def load_table(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    rows = csv.DictReader(lines, delimiter="\t")
    return {row["wrong"]: row["correct"] for row in rows if row.get("wrong") and row.get("correct")}


def replace_phrases(text: str, mappings: dict[str, str]) -> tuple[str, int]:
    if not mappings:
        return text, 0
    # Reject tables that can create more corrections on a second run.
    sources = sorted(mappings, key=lambda word: (-len(word), word))
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sources)) + r")(?!\w)")
    if any(pattern.search(value) for value in mappings.values()):
        raise ValueError("Correction mappings must be terminal (no chains or cycles).")
    return pattern.subn(lambda match: mappings[match.group()], text)
