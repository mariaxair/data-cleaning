"""Repair I/l only inside explicit article-number labels; keep digit script."""

import re

_ARTICLE = re.compile(
    r"(?P<label>(?i:article|art\.|المادة)\s+)(?P<number>[0-9٠-٩Il]+)(?!\w)")


def apply(text: str) -> tuple[str, int]:
    count = 0

    def fix(match: re.Match) -> str:
        nonlocal count
        number = match["number"]
        replacement = "١" if any(c in "٠١٢٣٤٥٦٧٨٩" for c in number) else "1"
        fixed = number.replace("I", replacement).replace("l", replacement)
        count += fixed != number
        return match["label"] + fixed

    return _ARTICLE.sub(fix, text), count
