"""Consecutive article groups within explicit chapter/law boundaries."""

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..artifacts import fingerprint
from .screening import ScreeningResult
from .passages import PAGE_PATTERN, passage_spans

ARTICLE_PATTERN = re.compile(
    r"(?im)^[ \t#]*(?:Article|Art\.|المادة)[ \t]+(?P<number>[0-9٠-٩]+(?:[-–][0-9٠-٩]+)*)[^\n]*")
_BOUNDARY = re.compile(r"(?im)^[ \t#]*(?:chapitre|chapter|section|الفصل|الباب|القسم|loi|law|قانون)\b[^\n]*")
_PAGE = re.compile(r"(?m)^--- Page (?P<page>\d+) ---\s*$")


@dataclass
class SynthesisUnit:
    source_file: str
    source_sha256: str
    text: str
    tier: str
    law_title: str = ""
    law_number: str = ""
    article_numbers: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    category: str = ""
    sensitive_values: list[str] = field(default_factory=list)
    source_document: str = ""
    source_document_sha256: str = ""
    synthesis_mode: str = "legal"
    topic: str = ""
    topic_group_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None

    @property
    def id(self) -> str:
        return fingerprint(asdict(self))


def segment_document(text: str, source: Path, screening: ScreeningResult, *,
                     group_size: int = 3, min_chars: int = 40, policy: dict | None = None,
                     source_hash: str = "", general_chunk_chars: int = 4000) -> list[SynthesisUnit]:
    """Keep rulings whole; drop singleton article tails rather than fabricate context."""
    if group_size not in {2, 3, 4}:
        raise ValueError("Article group size must be 2–4.")
    if screening.tier not in {"statutory_law", "case_law", "general"}:
        return []
    policy = policy or {}
    metadata = screening.metadata
    page_marks = list(_PAGE.finditer(text))

    def pages_at(start: int, end: int) -> list[int]:
        preceding = [m for m in page_marks if m.start() <= start]
        pages = [int(preceding[-1]["page"])] if preceding else []
        pages.extend(int(m["page"]) for m in page_marks if start < m.start() < end)
        return sorted(set(pages))

    def unit(body: str, numbers: list[str], start: int, end: int, category: str = "") -> SynthesisUnit:
        body = _PAGE.sub("", body).strip()
        return SynthesisUnit(str(source), source_hash, body, screening.tier,
                             metadata.get("law_title", ""), metadata.get("law_number", ""),
                             numbers, pages_at(start, end), category,
                             metadata.get("sensitive_values", []))

    if "\ufffd" in text or any("\ue000" <= char <= "\uf8ff" for char in text):
        return []  # Known garbled residue; semantic scrambling still needs review.
    if screening.tier == "general":
        result = []
        for start, end, pages in passage_spans(text, general_chunk_chars):
            body = PAGE_PATTERN.sub("", text[start:end]).strip()
            if len(body) < min_chars:
                continue
            entry = unit(body, [match["number"] for match in ARTICLE_PATTERN.finditer(body)],
                         start, end)
            entry.pages = pages
            entry.synthesis_mode = "automatic"
            entry.topic = metadata.get("topic", "")
            entry.topic_group_id = metadata.get("topic_group_id")
            entry.char_start, entry.char_end = start, end
            result.append(entry)
        return result
    if screening.tier == "case_law":
        return [unit(text, [m["number"] for m in ARTICLE_PATTERN.finditer(text)], 0, len(text))]
    # Preamble/masthead is outside the article spans. Heading groups constrain relatedness.
    boundaries = list(_BOUNDARY.finditer(text))
    groups: list[list[tuple]] = []
    current: list[tuple] = []
    previous_category = None
    articles = list(ARTICLE_PATTERN.finditer(text))
    for index, article in enumerate(articles):
        end = articles[index + 1].start() if index + 1 < len(articles) else len(text)
        next_boundary = next((b for b in boundaries if article.end() <= b.start() < end), None)
        if next_boundary:
            end = next_boundary.start()
        preceding = [b for b in boundaries if b.start() < article.start()]
        category = preceding[-1].group().strip() if preceding else ""
        body = text[article.start():end].strip()
        excluded = any(term.casefold() in body.casefold()
                       for term in policy.get("exclude_article_terms", []) if term)
        if category != previous_category or excluded or len(body) < min_chars:
            if current:
                groups.append(current)
                current = []
        previous_category = category
        if excluded or len(body) < min_chars or re.search(r"\.{4,}\s*[0-9٠-٩]+\s*$", body):
            continue
        current.append((body, article["number"], article.start(), end, category))
    if current:
        groups.append(current)
    result = []
    for group in groups:
        offset = 0
        while offset < len(group):
            count = min(group_size, len(group) - offset)
            # Avoid a final singleton when the previous group can leave two articles.
            if len(group) - offset - count == 1 and count > 2:
                count -= 1
            chunk = group[offset:offset + count]
            offset += count
            if len(chunk) < 2:
                continue
            result.append(unit("\n\n".join(a[0] for a in chunk), [a[1] for a in chunk],
                               chunk[0][2], chunk[-1][3], chunk[0][4]))
    return result
