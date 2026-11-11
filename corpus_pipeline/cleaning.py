"""Conservative, deterministic post-processing for extracted OCR text."""

import re
from dataclasses import dataclass

import ftfy


@dataclass
class CleaningResult:
    text: str
    hit_counts: dict[str, int]


_HEADER_LINE = re.compile(
    r"(?i)^\s*(?:ocr\s+)?(?:transcription|conversion|extraction)\s+"
    r"(?:result|notes?|output|of\s+the\s+document)\s*[:\-].*$"
)
_BR_TAG = re.compile(r"(?i)<\s*br\s*/?\s*>")
_HTML_TAG = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")
_REPEATED_WORD = re.compile(
    r"(?iu)(?<!\w)([\w\u0600-\u06ff]+)(?:\s+\1)+(?!\w)"
)


def clean_text(text: str, languages: list[str] | None = None) -> CleaningResult:
    """Repair mojibake and apply traceable, conservative format cleanups."""
    del languages  # Language hints are retained in the API for future rules.
    hits = {"encoding_repairs": 0, "metadata_headers": 0, "html_residue": 0,
            "repeated_lines": 0, "adjacent_word_duplicates": 0}

    repaired = ftfy.fix_text(text)
    if repaired != text:
        hits["encoding_repairs"] = 1
    text = repaired.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and _HEADER_LINE.match(lines[0]):
        lines.pop(0)
        hits["metadata_headers"] += 1
        while lines and not lines[0].strip():
            lines.pop(0)
    text = "\n".join(lines)

    text, br_count = _BR_TAG.subn("\n", text)
    text, tag_count = _HTML_TAG.subn("", text)
    hits["html_residue"] = br_count + tag_count

    # Collapse only runs of three or more identical consecutive lines.
    collapsed: list[str] = []
    index = 0
    lines = text.split("\n")
    while index < len(lines):
        end = index + 1
        while end < len(lines) and lines[end] == lines[index]:
            end += 1
        run = end - index
        collapsed.append(lines[index])
        if lines[index].strip() and run >= 3:
            hits["repeated_lines"] += run - 1
        else:
            collapsed.extend(lines[index + 1:end])
        index = end
    text = "\n".join(collapsed)

    text, hits["adjacent_word_duplicates"] = _REPEATED_WORD.subn(r"\1", text)
    return CleaningResult(text=text.strip(), hit_counts=hits)
