"""Apply reviewed foreign-glyph mappings only between Arabic letters."""

import re

_ARABIC = r"[\u0621-\u063a\u0641-\u064a]"


def apply(text: str, mappings: dict[str, str]) -> tuple[str, int]:
    count = 0
    for wrong, right in mappings.items():
        if len(wrong) != 1 or not re.fullmatch(r"[\u0400-\u052f\u0530-\u058f\u0590-\u05ff]", wrong):
            raise ValueError("Foreign-glyph mappings must have a single Cyrillic, Armenian, or Hebrew source.")
        if not re.fullmatch(_ARABIC + "+", right):
            raise ValueError("Foreign-glyph mappings must target Arabic letters.")
        text, hits = re.subn(f"(?<={_ARABIC}){re.escape(wrong)}(?={_ARABIC})", lambda _: right, text)
        count += hits
    return text, count
