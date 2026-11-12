"""Deterministic semantic corrections, enabled explicitly by the caller."""

from pathlib import Path

from ..settings import load_json
from . import dates, digits, footers, scripts, tables


def apply_rules(text: str, data_path: Path, words_path: Path) -> tuple[str, dict[str, int]]:
    data = load_json(data_path)
    hits: dict[str, int] = {}
    text, hits["article_digit_confusion"] = digits.apply(text)
    text, hits["foreign_script_corrections"] = scripts.apply(text, data.get("foreign_glyphs", {}))
    text, hits["explicit_date_corrections"] = dates.apply(text, data.get("dates", {}))
    text, hits["embedded_page_numbers"] = footers.apply(text, data.get("footer_markers", []))
    text, hits["curated_wrong_words"] = tables.replace_phrases(text, tables.load_table(words_path))
    return text, hits
