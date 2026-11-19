"""Slice a whole gazette OCR Markdown issue into one record per legal instrument.

The OCR stage emits an entire issue in a single Markdown file: masthead, index,
then dozens of instruments (decrees, orders, decisions, agreements, communiqués).
This stage reconstructs instrument boundaries, extracts metadata, and reports
structural defects (missing articles, reversed numbers, index mismatches) instead
of silently emitting a truncated law.

Every record keeps the original Markdown text so the slice stays auditable; the
only edits are dropping the index, the interleaved annex/preamble (optional) and
page furniture. Nothing is reordered: if the OCR transcribed pages out of order,
the record carries an ``article_order`` warning instead of a guessed reconstruction.

Fixes applied to the source before slicing are listed in the per-file summary
under ``repairs``; they never invent words, only split headings the OCR glued to
the end of a previous paragraph.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Text primitives
# --------------------------------------------------------------------------- #

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_TYPE = (
    r"(?:مرسوم(?:ان|ين|تان|تين)?"
    r"|قرار(?:ان|ين|ات)?"
    r"|مقرر(?:ان|ين|ات)?"
    r"|أمر|قانون|اتفاق(?:ية|يات)?"
    r"|بلاغ(?:ان|ات)?|إعلان(?:ان|ات)?|منشور(?:ان|ات)?"
    r"|نظام|تعليمات|لائحة|رأي)"
)
_INSTRUMENT = re.compile(
    rf"^(?P<type>{_TYPE})(?:\s+(?P<qualifiers>[^\n]{{0,60}}?))?\s*(?:رقم\b|مؤرخ|مؤرّخ|$)"
)
_EMBEDDED_HEADING = re.compile(rf"\*\*[ \t]*(?={_TYPE}\b)")
_NUMBER = re.compile(r"رقم\s*([0-9٠-٩]{1,4})\s*[-–]\s*([0-9٠-٩]{1,4})")
_SINGLE_NUMBER = re.compile(r"رقم\s*([0-9٠-٩]{1,4})(?!\s*[-–])")
_DATE = re.compile(r"مؤرخ(?:ة|ان|تين|تان)?\s+في")
_YEAR = re.compile(r"سنة\s*([0-9]{4})")
_GREGORIAN = re.compile(r"الموافق\s+[^\n]*?سنة\s*([0-9]{4})")
_PREAMBLE = re.compile(
    r"^(?:إن|ان|انّ|إنّ)\s+(?:رئيس الجمهورية|الوزير الأول|الوزيرالاول|وزير الدولة|رئيس الحكومة)"
)
_ENACT = re.compile(
    r"^(?:يرسم|يقرر|تقرر|أمر|يأمر|يرسمان|يقرران|اتفق(?:ا|ت|وا)?|اتفقت)"
    r"\s*(?:على\s+)?ما\s+يأتي"
)
_CONNECTOR = re.compile(
    r"^(?:وبمقتضى|وبناء|بناء|طبقا|تطبيقا|ويقتضي|بموجب|عملا|وعملا|وفقا|بعد الاطلاع|"
    r"إن|ان|يرسم|تقرر|يقرر|أمر|قرار\s+بأن)"
)
_SECTION = re.compile(
    r"^(?:اتفاقيات واتفاقات دولية|اتفاقيات دولية|مراسيم تنظيمية|مراسيم فردية|مراسيم|"
    r"قوانين|أوامر|قرارات[،,]?\s*مقرّرات[،,]?\s*آراء|قرارات وآراء|"
    r"قرارات[،,].*مناشير.*)$"
)
_TREATY = re.compile(r"^(?:اتفاق|اتفاقية)\b")
_ARTICLE = re.compile(
    r"^[ \t#>*]*"
    r"(?P<quote1>[\"“])?\s*\*{0,2}\s*(?P<quote2>[\"“])?\s*"
    r"(?:المادة|Article|Art\.)[ \t]+"
    r"(?P<num>[0-9٠-٩]{1,4}|الأولى|الاولى|الأول|الاول|أولا|اولا|أوّلا)"
    r"[ \t]*(?P<repeat>مكرر(?:[ \t]*[0-9٠-٩]{1,2})?)?"
    r"[ \t]*\*{0,2}[ \t]*(?::|：|\*{2}|$)",
    re.M,
)
_FURNITURE = re.compile(
    r"^(?:"
    r"[-–—_*]{3,}"
    r"|[★☆•·]+"
    r"|[0-9٠-٩]{1,4}"
    r"|الصفحة\s*[0-9٠-٩]{1,4}"
    r"|الجريدة الرسمية[^\n]{0,60}العدد\s*[0-9٠-٩]{1,4}"
    r"|#*\s*[0-9٠-٩]{1,2}\s+\S+(?:\s+عام|\s+سنة)?\s+[0-9]{4}\s*(?:هـ|ه|م)?"
    r"|[^|\n]{0,40}\|\s*[0-9٠-٩]{1,2}\s+\S+\s+سنة\s+[0-9]{4}\s*(?:هـ|ه|م)"
    r")$"
)
_WORD_ONE = {"الأولى", "الاولى", "الأول", "الاول", "أولا", "اولا", "أوّلا"}
_PLURAL_TYPE = {
    "مرسومان": "مرسوم", "مرسومين": "مرسوم", "مرسومتان": "مرسوم", "مرسومتين": "مرسوم",
    "قراران": "قرار", "قرارين": "قرار", "قرارات": "قرار",
    "مقرران": "مقرر", "مقررين": "مقرر", "مقررات": "مقرر",
    "بلاغان": "بلاغ", "بلاغات": "بلاغ",
    "إعلانان": "إعلان", "إعلانات": "إعلان",
    "منشوران": "منشور", "منشورات": "منشور",
}
_AMENDING = re.compile(r"(?:يعدل|تعدل|يتمم|تتمم|يعدّل|تعدّل|يتمّم|تتمّم)")


def _plain(line: str) -> str:
    """Strip Markdown heading, emphasis and bullet decoration from a line."""
    value = line.strip().replace("**", "").replace("__", "")
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = re.sub(r"^[>*\-+]+[ \t]*", "", value)
    return value.strip()


def _digits(value: str) -> str:
    return value.translate(_ARABIC_DIGITS)


def _compact(numbers: list[int]) -> str:
    """Render 1,2,3,7 as '1-3,7'."""
    if not numbers:
        return ""
    spans: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        spans.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = number
    spans.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(spans)


# --------------------------------------------------------------------------- #
# Instrument detection
# --------------------------------------------------------------------------- #

@dataclass
class InstrumentStart:
    line: int
    law_type: str
    law_number: str | None
    number_parts: tuple[int, int] | None
    plural: bool
    has_date: bool
    title: str
    parent_line: int | None = None

    @property
    def is_annex(self) -> bool:
        return self.parent_line is not None


def _normalise_type(match: re.Match[str]) -> tuple[str, bool]:
    head = match.group("type")
    words = _plain(match.group("qualifiers") or "").split()
    base = _PLURAL_TYPE.get(head, head)
    plural = head in _PLURAL_TYPE
    if words:
        first = words[0]
        if first in {"تنفيذي", "تنفيذية", "تنفيذيان", "تنفيذيين", "تنفيذيتان", "تنفيذيتين"}:
            base += " تنفيذي"
        elif first in {"رئاسي", "رئاسية", "رئاسيان", "رئاسيين", "رئاسيتان", "رئاسيتين"}:
            base += " رئاسي"
        elif first == "وزاري":
            base += " وزاري"
            if len(words) > 1 and words[1] in {"مشترك", "مشتركة", "مشتركان", "مشتركون"}:
                base += " مشترك"
        elif first in {"حكومي", "حكومية"}:
            base += " حكومي"
    return base, plural


def _title_at(lines: list[str], index: int) -> str:
    """Rejoin a heading the OCR wrapped over two or three bolded lines."""
    parts = [_plain(lines[index])]
    marks = lines[index].count("**")
    cursor = index + 1
    while marks % 2 == 1 and cursor < len(lines) and cursor <= index + 3:
        if not lines[cursor].strip():
            break
        parts.append(_plain(lines[cursor]))
        marks += lines[cursor].count("**")
        cursor += 1
    return " ".join(part for part in parts if part)


def _instrument_start(lines: list[str], index: int) -> InstrumentStart | None:
    """Recognise a line that opens a new legal instrument (not a cross-reference)."""
    plain = _plain(lines[index])
    if not plain or _CONNECTOR.match(plain):
        return None
    match = _INSTRUMENT.match(plain)
    if not match:
        return None
    if "المادة" in plain[:30]:
        return None
    has_date = bool(_DATE.search(plain))
    number = _NUMBER.search(plain)
    single = _SINGLE_NUMBER.search(plain)
    if not has_date and not number and not single:
        return None
    law_type, plural = _normalise_type(match)
    parts: tuple[int, int] | None = None
    value: str | None = None
    if number:
        first, second = int(_digits(number.group(1))), int(_digits(number.group(2)))
        parts = (first, second)
        value = f"{first}-{second}"
    elif single:
        value = _digits(single.group(1))
    return InstrumentStart(index, law_type, value, parts, plural, has_date,
                           _title_at(lines, index))


# --------------------------------------------------------------------------- #
# Index (TOC) parsing
# --------------------------------------------------------------------------- #

@dataclass
class TocEntry:
    law_number: str
    page: int | None
    law_type: str
    title: str


def _parse_toc(lines: list[str]) -> dict[str, TocEntry]:
    entries: dict[str, TocEntry] = {}
    for index, line in enumerate(lines):
        start = _instrument_start(lines, index)
        if start is None or start.law_number is None:
            continue
        page_match = re.search(r"([0-9٠-٩]{1,3})\s*$", _plain(line))
        page = int(_digits(page_match.group(1))) if page_match else None
        entries[start.law_number] = TocEntry(start.law_number, page, start.law_type, start.title)
    return entries


def _toc_bounds(lines: list[str]) -> tuple[int, int]:
    """Return (index_start, body_start) for the catalogue block.

    Index rows look exactly like body headings, and some issues do not print page
    numbers next to them. The reliable difference is content: an index row is
    followed by another index row, while the first body instrument is followed by
    its articles. Fall back to the last page-numbered row when the issue has no
    body article at all.
    """
    toc_start = next((i for i, line in enumerate(lines) if re.match(r"^#{0,6}\s*فهرس", line.strip())),
                     None)
    if toc_start is None:
        return 0, 0
    candidates = [index for index in range(toc_start, len(lines))
                  if _instrument_start(lines, index) is not None]
    for position, index in enumerate(candidates):
        end = candidates[position + 1] if position + 1 < len(candidates) else len(lines)
        if any(_ARTICLE.match(line) for line in lines[index:end]):
            return toc_start, index
    numbered = [index for index in range(toc_start, len(lines))
                if re.search(r"[0-9٠-٩]{1,3}\s*$", _plain(lines[index]))
                and (_SECTION.match(_plain(lines[index])) or _instrument_start(lines, index))]
    return toc_start, min((max(numbered) + 1) if numbered else toc_start, len(lines))


# --------------------------------------------------------------------------- #
# Article analysis
# --------------------------------------------------------------------------- #

def _article_label(match: re.Match[str]) -> str:
    raw = match.group("num")
    value = "1" if raw in _WORD_ONE else _digits(raw)
    if match.group("repeat"):
        value += " " + _plain(match.group("repeat"))
    return value


def _article_hits(body: str, *, amending: bool) -> list[tuple[str, bool]]:
    """Article headings in order, flagged when they quote an amended text.

    An amending instrument quotes the rewritten articles verbatim, so its own
    numbering restarts at 1 and then jumps to the quoted numbers. A heading is
    treated as quoted when it opens with a quotation mark, or (for amending
    instruments only) when its number breaks the sequence the instrument itself
    had established.
    """
    hits: list[tuple[str, bool]] = []
    expected = 1
    for match in _ARTICLE.finditer(body):
        label = _article_label(match)
        base = label.split()[0]
        quoted = bool(match.group("quote1") or match.group("quote2"))
        if amending and not quoted and base.isdigit() and int(base) != expected:
            quoted = True
        hits.append((label, quoted))
        if not quoted and base.isdigit():
            expected = int(base) + 1
    return hits


def _sequence_warnings(labels: list[str]) -> list[str]:
    bases = [int(label.split()[0]) for label in labels if label.split()[0].isdigit()]
    warnings: list[str] = []
    if not bases:
        return warnings
    seen = set(bases)
    missing = [value for value in range(1, max(bases) + 1) if value not in seen]
    if missing:
        warnings.append(f"missing_articles:{_compact(missing)}")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        warnings.append("duplicate_articles:" + ",".join(duplicates))
    if bases != sorted(bases):
        warnings.append("article_order")
    return warnings


# --------------------------------------------------------------------------- #
# Slicing
# --------------------------------------------------------------------------- #

@dataclass
class Slice:
    text: str
    law_number: str | None
    law_type: str
    year: str | None
    lang: str
    source: str
    index: int
    category: str
    eligible: bool
    title: str
    parent: str | None = None
    article_numbers: list[str] = field(default_factory=list)
    quoted_articles: list[str] = field(default_factory=list)
    page: int | None = None
    line_start: int = 0
    line_end: int = 0
    char_start: int = 0
    char_end: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def slice_id(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if key not in {"text", "index"}}
        payload["text_sha256"] = hashlib.sha256(self.text.encode()).hexdigest()
        return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()

    def record(self) -> dict:
        return {**asdict(self), "slice_id": self.slice_id,
                "article_count": len(self.article_numbers)}


def _language(text: str) -> str:
    arabic = sum(1 for char in text if "\u0600" <= char <= "\u06ff")
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    if not arabic and not latin:
        return "ar"
    return "ar" if arabic >= latin else "fr"


def _normalise_lines(text: str) -> tuple[list[str], list[str]]:
    """Split headings the OCR glued to the previous paragraph, keep everything else."""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    repairs: list[str] = []
    for raw in normalised.split("\n"):
        stripped = raw.lstrip()
        first = len(raw) - len(stripped)
        cuts = [match.start() for match in _EMBEDDED_HEADING.finditer(raw)
                if match.start() > first and raw[match.start():].count("**") % 2 == 1]
        if not cuts or not re.search(r"[.،;:!?]\s*$", raw[:cuts[0]].rstrip()):
            lines.append(raw)
            continue
        bounds = [0, *cuts, len(raw)]
        pieces = [raw[a:b].strip() for a, b in zip(bounds, bounds[1:]) if raw[a:b].strip()]
        lines.extend(pieces)
        repairs.append("split_joined_heading")
    return lines, repairs


_JOURNAL = re.compile(r"الجريدة الرسمية[^\n]{0,60}العدد\s*[0-9٠-٩]{1,4}")
_PAGE = re.compile(r"^الصفحة\s*[0-9٠-٩]{1,4}$")


def _is_furniture(line: str) -> bool:
    stripped = line.strip()
    return bool(_FURNITURE.match(stripped) or _JOURNAL.search(stripped) or _PAGE.match(stripped))


def _body_anchors(lines: list[str], start: int, end: int) -> tuple[int | None, int | None]:
    """Locate the preamble opening and the enacting formula after a title."""
    preamble = next((index for index in range(start + 1, end) if _PREAMBLE.match(_plain(lines[index]))),
                    None)
    search_from = preamble if preamble is not None else start + 1
    enact = next((index for index in range(search_from, end) if _ENACT.match(_plain(lines[index]))),
                 None)
    return preamble, enact


def _instrument_text(lines: list[str], start: int, end: int, *, include_preamble: bool,
                     first_article: int | None) -> str:
    kept: list[str] = []
    for index in range(start, end):
        line = lines[index]
        if _is_furniture(line):
            continue
        if not include_preamble and first_article is not None and start < index < first_article:
            continue
        kept.append(line.rstrip())
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept).strip()


def _year_of(title: str) -> str | None:
    match = _GREGORIAN.search(title)
    if match:
        return match.group(1)
    for candidate in _YEAR.findall(title):
        if int(candidate) >= 1900:
            return candidate
    return None


def slice_markdown(text: str, source: str, *, include_preamble: bool = False) -> tuple[list[Slice], dict]:
    """Return one record per instrument plus a per-file structural summary."""
    lines, repairs = _normalise_lines(text)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)

    toc_start, body_start = _toc_bounds(lines)
    toc = _parse_toc(lines[toc_start:body_start])

    starts: list[InstrumentStart] = []
    sections: list[int] = []
    for index in range(body_start, len(lines)):
        plain = _plain(lines[index])
        if not plain:
            continue
        if _SECTION.match(plain):
            sections.append(index)
            continue
        found = _instrument_start(lines, index)
        if found is not None:
            starts.append(found)
            continue
        if _TREATY.match(plain) and not _DATE.search(plain):
            # A ratified treaty printed as an annex: its own record, linked to the
            # decree that ratifies it when the signature closes the decree first.
            previous = starts[-1] if starts else None
            signed = previous is not None and any(
                re.match(r"^(?:حرر بالجزائر|عبد المجيد تبون)", _plain(lines[line]))
                for line in range(previous.line + 1, index)
            )
            headed = lines[index].lstrip().startswith("#") or "بين حكوم" in plain
            if signed or headed:
                starts.append(InstrumentStart(index, plain.split()[0], None, None, False, False, plain,
                                              parent_line=previous.line if previous else None))

    slices: list[Slice] = []
    line_to_number: dict[int, str | None] = {}
    file_warnings: list[str] = []
    for position, start in enumerate(starts):
        end = starts[position + 1].line if position + 1 < len(starts) else len(lines)
        end = min([section for section in sections if start.line < section < end] + [end])
        block = "\n".join(lines[start.line:end])

        preamble, enact = _body_anchors(lines, start.line, end)
        search_from = enact if enact is not None else (preamble if preamble is not None else start.line + 1)
        first_article = next((index for index in range(search_from, end) if _ARTICLE.match(lines[index])),
                             None)
        instrument_text = _instrument_text(lines, start.line, end, include_preamble=include_preamble,
                                           first_article=first_article)
        amending = bool(_AMENDING.search(start.title))
        hits = _article_hits(instrument_text, amending=amending)
        numbers = [label for label, quoted in hits if not quoted]
        quoted = [label for label, quoted in hits if quoted]
        warnings = _sequence_warnings(numbers)
        if amending:
            warnings.append("amending_instrument")
        if start.plural:
            warnings.append("plural_instrument")
        if not hits and instrument_text.count("\nبموجب ") > 1:
            warnings.append("multiple_decrees")
        if start.law_number is None and not start.is_annex:
            warnings.append("unnumbered")
        if not hits:
            warnings.append("no_articles")
        if start.is_annex:
            warnings.append("annex")
        if preamble is not None and first_article is not None:
            skipped = sum(1 for index in range(start.line + 1, preamble)
                          if lines[index].strip() and not _is_furniture(lines[index]))
            if skipped > 3:
                warnings.append("page_order_suspected")

        title = start.title
        number = start.law_number
        if start.number_parts:
            first, second = start.number_parts
            candidate = f"{second}-{first}"
            if candidate in toc and number not in toc:
                number = candidate
                warnings.append("number_reversed_in_ocr")
        if number is not None and toc and number not in toc:
            warnings.append("number_not_in_index")

        category = _category(start, number)
        eligible = (category != "individual" and bool(numbers)
                    and not any(warning.startswith(("missing_articles", "duplicate_articles"))
                                or warning == "article_order" for warning in warnings))
        entry = toc.get(number or "")
        line_to_number[start.line] = number
        slices.append(Slice(
            text=instrument_text,
            law_number=number, law_type=start.law_type, year=_year_of(title),
            lang=_language(block), source=source, index=len(slices), category=category,
            eligible=eligible, title=title,
            parent=line_to_number.get(start.parent_line) if start.is_annex else None,
            article_numbers=numbers, quoted_articles=quoted,
            page=entry.page if entry else None,
            line_start=start.line, line_end=end - 1,
            char_start=offsets[start.line], char_end=offsets[end - 1],
            warnings=warnings,
        ))

    # Index reconciliation: numbers promised by the TOC must exist in the body.
    body_numbers = {item.law_number for item in slices if item.law_number}
    toc_numbers = set(toc)
    if toc_numbers - body_numbers:
        file_warnings.append("index_missing:" + _compact_sorted(toc_numbers - body_numbers))
    if body_numbers - toc_numbers:
        file_warnings.append("index_extra:" + _compact_sorted(body_numbers - toc_numbers))
    if not slices:
        file_warnings.append("no_instruments")

    summary = {
        "source": source,
        "instruments": len(slices),
        "eligible": sum(item.eligible for item in slices),
        "by_category": {name: sum(item.category == name for item in slices)
                        for name in sorted({item.category for item in slices})},
        "with_warnings": sum(bool(item.warnings) for item in slices),
        "articles": sum(len(item.article_numbers) for item in slices),
        "index_entries": len(toc),
        "repairs": sorted(set(repairs)),
        "warnings": file_warnings,
    }
    return slices, summary


def _category(start: InstrumentStart, number: str | None) -> str:
    if start.is_annex:
        return "agreement"
    if start.law_type.startswith("مرسوم") or start.law_type in {"قانون", "أمر"}:
        return "individual" if number is None else "statutory"
    if start.law_type.startswith("قرار") or start.law_type.startswith("مقرر"):
        return "decision"
    if start.law_type.startswith("اتفاق"):
        return "agreement"
    return "notice"


def _compact_sorted(values: set[str]) -> str:
    numeric = sorted((value for value in values if value and value.replace("-", "").isdigit()),
                     key=lambda value: tuple(int(part) for part in value.split("-")))
    other = sorted(value for value in values if value and not value.replace("-", "").isdigit())
    return ",".join(numeric + other) or "none"


# --------------------------------------------------------------------------- #
# Batch entry point
# --------------------------------------------------------------------------- #

def slice_folder(paths: list[Path], output: Path, *, include_preamble: bool = False) -> dict:
    from .artifacts import write_json, write_jsonl

    summary = {"files": 0, "instruments": 0, "eligible": 0, "warnings": 0, "sources": []}
    for path in paths:
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        slices, file_summary = slice_markdown(text, path.name, include_preamble=include_preamble)
        write_jsonl(output / f"{path.stem}.jsonl", [item.record() for item in slices])
        summary["files"] += 1
        summary["instruments"] += file_summary["instruments"]
        summary["eligible"] += file_summary["eligible"]
        summary["warnings"] += file_summary["with_warnings"]
        summary["sources"].append(file_summary)
    write_json(output / "summary.json", summary)
    return summary
