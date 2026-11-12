"""Exact deduplication after configurable-domain-independent Arabic normalization."""

import re
import unicodedata


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ـ": ""}))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()
