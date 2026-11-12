"""Observable checks, with semantic grounding/privacy explicitly unproven."""

import re
import unicodedata
from dataclasses import dataclass, field

from .segmentation import SynthesisUnit

_ARTICLE_REF = re.compile(r"(?i)(?:المادة|المواد|article[s]?|art\.)\s+([0-9٠-٩]+(?:[-–][0-9٠-٩]+)*)")
_NUMBERS = re.compile(r"[0-9٠-٩]+(?:[-–/.,][0-9٠-٩]+)*")
_PII = re.compile(r"[\w.+-]+@[\w.-]+\.\w+|(?<!\d)(?:\+?\d[\s-]?){9,15}(?!\d)")
_IDENTIFIERS = re.compile(
    r"(?i)(?:رقم\s+(?:القضية|الهوية)|تاريخ\s+الميلاد|عنوان\s+السكن|case\s+(?:no|number)|date\s+of\s+birth)\s*[:：]?\s*\S+")


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=lambda: ["semantic_grounding", "personal_names_review"])

    @property
    def passed(self) -> bool:
        return not self.errors


def language_pure(text: str, language: str) -> bool:
    """Script purity, not linguistic or dialect identification."""
    for char in text:
        if not unicodedata.category(char).startswith("L"):
            continue
        name = unicodedata.name(char, "")
        if language == "ar" and "ARABIC" not in name:
            return False
        if language in {"fr", "en"} and "LATIN" not in name:
            return False
    return bool(text.strip())


def article_consistent(text: str, unit: SynthesisUnit) -> bool:
    allowed = (set(match[1] for match in _ARTICLE_REF.finditer(unit.text)) if unit.tier == "general"
               else set(unit.article_numbers))
    return all(match[1] in allowed and match[1] in unit.text for match in _ARTICLE_REF.finditer(text))


def privacy_clear(text: str, sensitive_values: list[str]) -> bool:
    return not (_PII.search(text) or _IDENTIFIERS.search(text)
                or any(value and value.casefold() in text.casefold() for value in sensitive_values))


def question_unnamed(text: str, unit: SynthesisUnit) -> bool:
    if unit.tier == "general":
        return True  # General learning questions need not hide legitimate document/section names.
    if re.search(r"(?i)(?:article|art\.|المادة|المواد)\s+[0-9٠-٩]+", text):
        return False
    return not any(value and value.casefold() in text.casefold()
                   for value in (unit.law_title, unit.law_number))


def validate_generation(phase: str, response: dict, unit: SynthesisUnit, language: str,
                        *, question: str = "", answer: str = "") -> ValidationResult:
    result = ValidationResult()
    text = response.get("text", "")
    if not isinstance(text, str) or not text.strip():
        result.errors.append("empty_text")
        return result
    if not language_pure(text, language):
        result.errors.append("language_purity")
    if not privacy_clear(text, unit.sensitive_values):
        result.errors.append("privacy")
    if not article_consistent(text, unit):
        result.errors.append("article_consistency")
    numeric_source = question + "\n" + answer if phase == "explanation" else unit.text
    if not set(_NUMBERS.findall(text)) <= set(_NUMBERS.findall(numeric_source)):
        result.errors.append("unsupported_number")
    if phase == "question" and not question_unnamed(text, unit):
        result.errors.append("named_law_or_article")
    if phase != "explanation":
        evidence = response.get("evidence", [])
        if not isinstance(evidence, list) or not evidence:
            result.errors.append("missing_evidence")
        elif any(not isinstance(item, dict) or not isinstance(item.get("quote"), str)
                 or not item["quote"] or item["quote"] not in unit.text
                 or not isinstance(item.get("claim"), str) or not item["claim"]
                 or item["claim"] not in text for item in evidence):
            result.errors.append("invalid_evidence")
    elif len(text) <= len(answer):
        result.errors.append("explanation_too_short")
    return result
