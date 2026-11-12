"""Share the existing HTML removal behavior with the format cleaner."""

import re


def apply(text: str) -> tuple[str, int]:
    text, breaks = re.subn(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text, tags = re.subn(r"<\s*/?\s*[a-zA-Z][^>]*>", "", text)
    return text, breaks + tags
