"""Bounded general-document passages with original character/page provenance."""

import re

PAGE_PATTERN = re.compile(r"(?m)^--- Page (?P<page>\d+) ---[ \t]*$")


def passage_spans(text: str, max_chars: int) -> list[tuple[int, int, list[int]]]:
    """Prefer paragraph/line/word boundaries; even long paragraphs remain bounded."""
    markers = list(PAGE_PATTERN.finditer(text))
    spans = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            for separator in ("\n\n", "\n", " "):
                boundary = text.rfind(separator, start + max_chars // 2, end)
                if boundary > start:
                    end = boundary
                    break
        if end <= start:
            end = min(start + max_chars, len(text))
        preceding = [marker for marker in markers if marker.start() <= start]
        pages = [int(preceding[-1]["page"])] if preceding else []
        pages += [int(marker["page"]) for marker in markers if start < marker.start() < end]
        if PAGE_PATTERN.sub("", text[start:end]).strip():
            spans.append((start, end, sorted(set(pages))))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return spans
