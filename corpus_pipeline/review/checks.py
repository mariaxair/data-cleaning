"""Deterministic quality checks with explicit limits and missing-source handling."""

import re
import unicodedata
from dataclasses import dataclass, field

from config import PipelineConfig
from ..governance.dialect import DialectClassifier
from ..governance.runner import FIELDS, govern_record
from ..rules.duplicates import apply as remove_duplicates
from ..synthesis.segmentation import SynthesisUnit
from ..synthesis.validation import article_consistent, language_pure, privacy_clear, question_unnamed

LLM_DIMENSIONS = ("question_accuracy", "reasoning_completeness", "question_self_containedness", "privacy_compliance")
RULE_DIMENSIONS = ("language_purity", "markdown_compliance", "fluency_no_repetition", "corpus_accuracy", "dialect_compliance")


@dataclass
class ReviewResult:
    scores: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, list[str]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    failed_fields: dict[str, list[str]] = field(default_factory=dict)
    dialect_veto: bool = False
    degraded: bool = False


def unit_from_record(record: dict) -> SynthesisUnit:
    provenance = record.get("provenance", {})
    return SynthesisUnit(
        provenance.get("source_file", ""), provenance.get("source_sha256", ""),
        record.get("source_text", ""), provenance.get("tier", "manual"),
        provenance.get("law_title", ""), provenance.get("law_number", ""),
        provenance.get("article_numbers", []), provenance.get("pages", []),
        provenance.get("category", ""), record.get("sensitive_values", []))


def markdown_valid(text: str) -> bool:
    """Check code fences, raw HTML, and simple Markdown link delimiter balance."""
    if text.count(chr(96) * 3) % 2 or re.search(r"</?[A-Za-z][^>]*>", text):
        return False
    return text.count("](") <= text.count(")") and text.count("](") <= text.count("[")


def fluent(text: str) -> bool:
    if remove_duplicates(text)[1]:
        return False
    lines = text.splitlines()
    return not any(lines[i].strip() and lines[i] == lines[i + 1] == lines[i + 2]
                   for i in range(max(0, len(lines) - 2)))


def deterministic_review(record: dict, config: PipelineConfig,
                         classifier: DialectClassifier) -> ReviewResult:
    result = ReviewResult()
    unit = unit_from_record(record)
    language = record.get("language", config.synthesis_language)

    def evaluate(dimension, predicate):
        failed = [field for field in FIELDS if not predicate(str(record.get(field, "")))]
        result.scores[dimension] = 0 if failed else 2
        result.failed_fields[dimension] = failed
        result.reasons[dimension] = [f"Failed deterministic check in {field}." for field in failed]

    evaluate("language_purity", lambda text: language_pure(text, language))
    evaluate("markdown_compliance", markdown_valid)
    evaluate("fluency_no_repetition", fluent)
    evaluate("corpus_accuracy", lambda text: bool(text.strip()) and article_consistent(text, unit)
             and not any(c == "\ufffd" or unicodedata.category(c) == "Co" for c in text))
    numbers = re.compile(r"[0-9٠-٩]+(?:[-–/.,][0-9٠-٩]+)*")
    if unit.text:
        for field in FIELDS:
            text = str(record.get(field, ""))
            if not set(numbers.findall(text)) <= set(numbers.findall(unit.text)):
                result.scores["corpus_accuracy"] = 0
                result.failed_fields["corpus_accuracy"].append(field)
                result.reasons["corpus_accuracy"].append(f"Unsupported numeric facts in {field}.")
        for field in ("question", "answer"):
            for item in record.get("evidence", {}).get(field, []):
                if not isinstance(item, dict) or not item.get("quote") or item["quote"] not in unit.text \
                        or not item.get("claim") or item["claim"] not in str(record.get(field, "")):
                    result.scores["corpus_accuracy"] = 0
                    result.failed_fields["corpus_accuracy"].append(field)
                    result.reasons["corpus_accuracy"].append(f"Invalid evidence in {field}.")
    elif result.scores["corpus_accuracy"] == 2:
        result.scores["corpus_accuracy"] = 1
        result.skipped.append("source_grounding_missing")
    governed = govern_record(record, config, classifier)["governance"]
    old_dialects = record.get("governance", {}).get("dialect", {})
    result.dialect_veto = (
        any(item.get("decision") == "intercept" for item in governed["dialect"].values())
        or any(item.get("decision") == "intercept" for item in old_dialects.values()))
    uncertain = any(item.get("decision") == "manual" for item in governed["dialect"].values())
    result.degraded = governed["degraded"]
    result.scores["dialect_compliance"] = 0 if result.dialect_veto else 1 if uncertain else 2
    result.reasons["dialect_compliance"] = (
        ["Moroccan dialect veto; no repair."] if result.dialect_veto
        else ["Dialect classification requires manual adjudication."] if uncertain else [])
    if any(value["decision"] == "manual" for value in governed["language"].values()):
        result.skipped.append("language_identity_review")
    if not question_unnamed(str(record.get("question", "")), unit):
        result.scores["question_accuracy"] = 0
        result.reasons["question_accuracy"] = ["Question names a law or article."]
    if not privacy_clear("\n".join(str(record.get(field, "")) for field in FIELDS), unit.sensitive_values):
        result.scores["privacy_compliance"] = 0
        result.reasons["privacy_compliance"] = ["Known sensitive value or identifier detected."]
    return result
