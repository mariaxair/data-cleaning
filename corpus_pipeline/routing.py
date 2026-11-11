"""Measurable per-page route selection."""

import unicodedata

from config import PipelineConfig
from .models import PageRoute, PageSignals

_EXPECTED_PUNCTUATION = set(".,;:!?()[]{}<>/\\-–—_'\"%°§«»؟،؛ـ")


def measure_page(text: str, image_ratio: float) -> PageSignals:
    """Measure text quantity, character quality, and image coverage."""
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        valid_ratio = 1.0
    else:
        valid = 0
        for char in nonspace:
            category = unicodedata.category(char)
            if category[0] in {"L", "N", "M"} or char in _EXPECTED_PUNCTUATION:
                valid += 1
        valid_ratio = valid / len(nonspace)
    return PageSignals(
        extracted_characters=len(text),
        valid_character_ratio=round(valid_ratio, 4),
        full_page_image_ratio=round(min(max(image_ratio, 0.0), 1.0), 4),
    )


def classify_page(text: str, signals: PageSignals, config: PipelineConfig) -> tuple[PageRoute, list[str]]:
    """Choose DIGITAL, SCANNED, or SUSPECT from configured signals."""
    warnings: list[str] = []
    has_text_layer = bool(text.strip())
    if has_text_layer and signals.valid_character_ratio < config.min_valid_char_ratio:
        warnings.append("Text layer contains many invalid or suspicious characters; routed to OCR.")
        return PageRoute.SUSPECT, warnings
    if signals.full_page_image_ratio >= config.full_page_image_ratio:
        if has_text_layer:
            warnings.append("A page-sized image covers most of the page; routed to OCR.")
        return PageRoute.SCANNED, warnings
    if signals.extracted_characters < config.min_extracted_chars:
        return PageRoute.SCANNED, warnings
    return PageRoute.DIGITAL, warnings
