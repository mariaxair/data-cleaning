"""Remove only explicitly configured footer tokens surrounded by body text."""

import re


def apply(text: str, markers: list[str]) -> tuple[str, int]:
    count = 0
    for marker in markers:
        if marker.strip():
            # No blanket removal of legal numbers or standalone numeric paragraphs.
            text, hits = re.subn(r"(?<=\S)[ \t]*" + re.escape(marker) + r"[ \t]*(?=\S)", " ", text)
            count += hits
    return text, count
