"""Existing adjacent-word collapse, shared by both cleaning modes."""

import re

_WORDS = re.compile(r"(?iu)(?<!\w)([\w\u0600-\u06ff]+)(?:\s+\1)+(?!\w)")


def apply(text: str) -> tuple[str, int]:
    return _WORDS.subn(r"\1", text)
