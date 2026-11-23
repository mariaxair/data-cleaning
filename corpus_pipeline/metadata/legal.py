"""Conservative legal syntax extraction, independent of synthesis segmentation."""

import re
from datetime import date

from ..synthesis.passages import PAGE_PATTERN
_PREFIX = r"^[ \t]*(?:\#{1,6}[ \t]*)?(?:\*\*)?"
ARTICLE = re.compile(_PREFIX + r"(?:المادة|Article|Art\.)[ \t]+"
                     r"(?P<number>[\d]+(?:[-–][\d]+)*|الأولى|الاولى|الأول|الثانية|الثالثة|الرابعة|الخامسة)"
                     r"(?=[\s:*\.،\-–]|$)[^\n]*", re.M | re.I)
SECTION = re.compile(_PREFIX + r"(?:الفصل|الباب|القسم|Chapter|Chapitre|Section)\b[^\n]*", re.M | re.I)
INSTRUMENT = re.compile(_PREFIX + r"(?P<kind>قانون(?: عضوي)?|مرسوم(?: رئاسي| تنفيذي)?|أمر|قرار|"
                        r"loi|décret|ordonnance|arrêté|law|decree)[ \t]+"
                        r"(?:رقم|n[°ºo.]*)[ \t]*(?P<number>\d+(?:[\-–/]\d+)+)[^\n]*", re.M | re.I)
_LEGAL_SIGNAL = re.compile(r"الجريدة الرسمية|الجمهورية|gazette|journal officiel|république|republic", re.I)
_ORDINALS = {"الأولى": "1", "الاولى": "1", "الأول": "1", "الثانية": "2", "الثالثة": "3",
             "الرابعة": "4", "الخامسة": "5"}
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NAMED_DATE = re.compile(r"\d{1,2}\s+(?:جانفي|يناير|فيفري|فبراير|مارس|أفريل|ابريل|ماي|يونيو|جوان|"
                         r"يوليو|جويلية|أوت|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر|janvier|février|mars|avril|"
                         r"mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(?:سنة\s+)?\d{4}", re.I)
_MONTHS = {word: month for month, words in enumerate((
    "جانفي يناير janvier", "فيفري فبراير février", "مارس mars", "أفريل ابريل avril", "ماي mai",
    "يونيو جوان juin", "يوليو جويلية juillet", "أوت أغسطس août", "سبتمبر septembre", "أكتوبر octobre",
    "نوفمبر novembre", "ديسمبر décembre"), 1) for word in words.split()}


def pages_for(text: str, start: int, end: int) -> list[int]:
    markers = list(PAGE_PATTERN.finditer(text))
    before = [marker for marker in markers if marker.start() <= start]
    pages = [int(before[-1]["page"])] if before else []
    pages += [int(marker["page"]) for marker in markers if start < marker.start() < end]
    return sorted(set(pages))


def normalize_article_number(value: str) -> str:
    return value.translate(_DIGITS)


def normalize_date(value: str) -> str | None:
    """Only unambiguous Gregorian dates become ISO; ambiguous d/m is left raw."""
    value = value.translate(_DIGITS).strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value).isoformat()
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", value):
            day, month, year = map(int, value.split("/"))
            return date(year, month, day).isoformat() if day > 12 else None
        match = re.fullmatch(r"(\d{1,2})\s+(\S+)\s+(?:سنة\s+)?(\d{4})", value)
        if match and match[2].casefold() in _MONTHS:
            return date(int(match[3]), _MONTHS[match[2].casefold()], int(match[1])).isoformat()
    except ValueError:
        pass
    return None


def invalid_calendar_date(value: str) -> bool:
    """Validate extracted date values without extracting them from the source."""
    numeric = value.translate(_DIGITS)
    parts = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", numeric)
    named = re.fullmatch(r"\d{1,2}\s+(\S+)\s+(?:سنة\s+)?\d{4}", numeric)
    recognized = (bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", numeric)) or
                  bool(named and named[1].casefold() in _MONTHS) or
                  bool(parts and (int(parts[1]) > 12 or int(parts[2]) > 12)))
    return recognized and normalize_date(value) is None


def legal_extract(text: str) -> dict:
    """Recognize legal layout for chunk boundaries only; metadata is extracted separately."""
    articles, instruments = list(ARTICLE.finditer(text)), list(INSTRUMENT.finditer(text))
    likely = bool(instruments or (articles and _LEGAL_SIGNAL.search(text)))
    # A gazette can contain several instruments: retain each title/number and local membership.
    headings = sorted([*articles, *SECTION.finditer(text)], key=lambda match: match.start())
    spans = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        next_instrument = next((m for m in instruments if heading.end() <= m.start() < end), None)
        if next_instrument:
            end = next_instrument.start()
        is_article = heading.re is ARTICLE
        preceding = [m for m in instruments if m.start() <= heading.start()]
        sections = [m.group() for m in SECTION.finditer(text) if
                    (preceding[-1].start() if preceding else 0) <= m.start() <= heading.start()]
        number = heading["number"] if is_article else None
        spans.append({"start": heading.start(), "end": end, "text": text[heading.start():end],
                      "heading": heading.group(), "level": "article" if is_article else "section",
                      "article_numbers": [number] if number else [],
                       "normalized_article_numbers": [_ORDINALS.get(number, normalize_article_number(number))] if number else [],
                      "structural_path": sections, "pages": pages_for(text, heading.start(), end),
                      "instrument_number": preceding[-1]["number"] if preceding else None})
    article_lines = list(re.finditer(_PREFIX + r"(?:المادة|Article|Art\.)(?=[\s*:：]|$)[^\n]*", text, re.M | re.I))
    broken_article = len(article_lines) != len(articles)
    complete = likely and bool(articles and instruments) and not broken_article
    # Include every source character; preambles/gaps are explicit structural context chunks.
    covered = []
    cursor = 0
    for span in spans:
        if cursor < span["start"]:
            covered.append({"start": cursor, "end": span["start"], "level": "context", "article_numbers": [],
                            "structural_path": [], "pages": pages_for(text, cursor, span["start"])})
        covered.append(span)
        cursor = span["end"]
    if cursor < len(text):
        covered.append({"start": cursor, "end": len(text), "level": "context", "article_numbers": [],
                        "structural_path": [], "pages": pages_for(text, cursor, len(text))})
    return {"likely": likely, "status": "complete" if complete else "needs_fallback",
            "spans": covered if likely else [],
            "warnings": [] if complete else ["Legal markers/required identifiers or article layout are incomplete."]}
