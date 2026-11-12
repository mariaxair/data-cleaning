"""Conservative, deterministic post-processing for extracted OCR text."""

import re
from dataclasses import dataclass

import ftfy
from config import PipelineConfig
from .rules import apply_rules
from .rules.html import apply as clean_html
from .rules.duplicates import apply as clean_duplicates


@dataclass
class CleaningResult:
    text: str
    hit_counts: dict[str, int]


_HEADER_LINE = re.compile(
    r"(?i)^\s*(?:ocr\s+)?(?:transcription|conversion|extraction)\s+"
    r"(?:result|notes?|output|of\s+the\s+document)\s*[:\-].*$"
)
def clean_text(text: str, languages: list[str] | None = None, *,
               config: PipelineConfig | None = None) -> CleaningResult:
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

    text, hits["html_residue"] = clean_html(text)

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

    text, hits["adjacent_word_duplicates"] = clean_duplicates(text)
    if config is not None and config.semantic_rules_enabled:
        text, semantic_hits = apply_rules(text, config.rules_data_path, config.wrong_words_path)
        hits.update(semantic_hits)
        # Reviewed mappings must also be terminal across rule families and formatting.
        candidate = text.strip()
        repeated_format = clean_text(candidate, config=None).text
        repeated_rules, _ = apply_rules(repeated_format, config.rules_data_path, config.wrong_words_path)
        if repeated_rules.strip() != candidate:
            raise ValueError("Semantic rules are not idempotent together; review the configured mappings.")
    return CleaningResult(text=text.strip(), hit_counts=hits)
