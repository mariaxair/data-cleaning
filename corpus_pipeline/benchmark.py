"""Per-document benchmark metrics for OCR and downstream pipeline stages.

The module is intentionally dependency-free: edit distances come from
:mod:`difflib`, so no new packages are required. Metrics are split into three
groups:

* reference-free quality proxies, which are always available and only indicate
  suspicious output (they do not measure accuracy);
* reference-based character/word error rates, which need an operator-supplied
  reference transcription;
* operational metrics (OCR latency, cache reuse, failures) and stage summaries.

Results are written as one ``<stem>.metrics.json`` per document, appended to an
append-only JSONL history, and summarised in a provider/model comparison report.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .artifacts import atomic_text, read_jsonl, write_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config import PipelineConfig
    from .models import PageReport

SCHEMA_VERSION = 1

# Page markers written by _join_pages, e.g. "\n\n--- Page 12 ---\n\n".
PAGE_MARKER = re.compile(r"^\s*-{2,}\s*Page\s+(\d+)\s*-{2,}\s*$", re.MULTILINE)
_HARAKAT = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TATWEEL = "\u0640"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U0010ffff]")
_ARABIC = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
_LATIN = re.compile(r"[A-Za-z\u00c0-\u024f]")
_DIGIT = re.compile(r"[0-9\u0660-\u0669\u06f0-\u06f9]")
_FOREIGN_SCRIPT = re.compile(
    r"[\u0400-\u04ff\u0590-\u05ff\u0900-\u097f\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_REPLACEMENT = "\ufffd"

STAGES = ("synthesis", "governance", "review")


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_text(
    text: str,
    *,
    strip_diacritics: bool = True,
    unify_alef: bool = False,
    unify_ya: bool = False,
    drop_page_markers: bool = True,
) -> str:
    """Normalise text for comparison without changing the exported document.

    Arabic diacritics and tatweel are removed because OCR commonly drops or
    invents them, which would otherwise dominate an error rate. Alef and ya
    unification is optional because it hides real orthographic differences.
    """
    if drop_page_markers:
        text = PAGE_MARKER.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_TATWEEL, "")
    if strip_diacritics:
        text = _HARAKAT.sub("", text)
    if unify_alef:
        text = text.replace("\u0623", "\u0627").replace("\u0625", "\u0627")
        text = text.replace("\u0622", "\u0627").replace("\u0671", "\u0627")
    if unify_ya:
        text = text.replace("\u0649", "\u064a")
    text = _CONTROL.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_config(text: str, config: "PipelineConfig", *, drop_page_markers: bool = True) -> str:
    return normalize_text(
        text,
        strip_diacritics=config.benchmark_strip_diacritics,
        unify_alef=config.benchmark_unify_alef,
        unify_ya=config.benchmark_unify_ya,
        drop_page_markers=drop_page_markers,
    )


def split_pages(text: str) -> dict[int, str]:
    """Split text that still contains page markers into {page number: text}."""
    matches = list(PAGE_MARKER.finditer(text))
    if not matches:
        return {}
    pages: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages[int(match.group(1))] = text[match.end():end].lstrip("\r\n")
    return pages


# ---------------------------------------------------------------------------
# Edit-based metrics
# ---------------------------------------------------------------------------

def _alignment_counts(reference: list[Any], hypothesis: list[Any]) -> Counter:
    """Approximate Levenshtein counts from difflib's matching blocks.

    ``difflib`` collapses a replacement into one opcode even when the two
    spans have different lengths, so the shorter length is counted as
    substitutions and the remainder as insertions or deletions.
    """
    matcher = SequenceMatcher(None, reference, hypothesis, autojunk=False)
    counts: Counter = Counter()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left, right = i2 - i1, j2 - j1
        if tag == "equal":
            counts["matches"] += left
        elif tag == "replace":
            counts["substitutions"] += min(left, right)
            counts["deletions"] += max(0, left - right)
            counts["insertions"] += max(0, right - left)
        elif tag == "delete":
            counts["deletions"] += left
        elif tag == "insert":
            counts["insertions"] += right
    return counts


def _word_alignment(reference: list[str], hypothesis: list[str]) -> tuple[Counter, Counter]:
    """Align words first, then compare characters only inside changed words.

    Character-level diffing of two 100k-character documents is quadratic and
    takes minutes on ordinary OCR noise. Words are a much smaller alphabet, so
    the same alignment is fast; replaced word pairs are then compared character
    by character. This keeps CER meaningful for intra-word OCR errors while
    avoiding a global character matrix.
    """
    matcher = SequenceMatcher(None, reference, hypothesis, autojunk=False)
    characters: Counter = Counter()
    words: Counter = Counter()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left, right = reference[i1:i2], hypothesis[j1:j2]
        if tag == "equal":
            words["matches"] += len(left)
            characters["matches"] += sum(len(word) for word in left)
        elif tag == "replace":
            words["substitutions"] += min(len(left), len(right))
            words["deletions"] += max(0, len(left) - len(right))
            words["insertions"] += max(0, len(right) - len(left))
            left_text, right_text = "".join(left), "".join(right)
            if len(left_text) + len(right_text) <= 1000:
                # Small blocks can be compared exactly without a large matrix.
                characters.update(_alignment_counts(list(left_text), list(right_text)))
            else:
                for reference_word, hypothesis_word in zip(left, right):
                    characters.update(
                        _alignment_counts(list(reference_word), list(hypothesis_word)))
                characters["deletions"] += sum(len(word) for word in left[len(right):])
                characters["insertions"] += sum(len(word) for word in right[len(left):])
        elif tag == "delete":
            words["deletions"] += len(left)
            characters["deletions"] += sum(len(word) for word in left)
        elif tag == "insert":
            words["insertions"] += len(right)
            characters["insertions"] += sum(len(word) for word in right)
    return characters, words


def _rate(value: float, denominator: int) -> float | None:
    return round(value / denominator, 6) if denominator else None


def _score(counts: Counter, reference_units: int, hypothesis_units: int) -> dict[str, Any]:
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    matches = counts["matches"]
    precision = _rate(matches, hypothesis_units)
    recall = _rate(matches, reference_units)
    f1 = None
    if precision is not None and recall is not None and precision + recall:
        f1 = round(2 * precision * recall / (precision + recall), 6)
    return {
        "reference_units": reference_units,
        "hypothesis_units": hypothesis_units,
        "matches": matches,
        "substitutions": counts["substitutions"],
        "deletions": counts["deletions"],
        "insertions": counts["insertions"],
        "errors": errors,
        "error_rate": _rate(errors, reference_units),
        "accuracy": round(max(0.0, 1.0 - errors / reference_units), 6) if reference_units else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare_texts(reference: str, hypothesis: str, max_eval_chars: int) -> dict[str, Any]:
    """Character (CER) and word (WER) metrics for one already-normalised pair.

    Character units exclude whitespace because normalisation collapses word
    spacing; words remain the alignment anchors. The cap is applied before
    tokenisation and applies equally to both sides.
    """
    truncated = len(reference) > max_eval_chars or len(hypothesis) > max_eval_chars
    reference = reference[:max_eval_chars]
    hypothesis = hypothesis[:max_eval_chars]
    reference_words, hypothesis_words = reference.split(), hypothesis.split()
    character_counts, word_counts = _word_alignment(reference_words, hypothesis_words)
    reference_characters = sum(len(word) for word in reference_words)
    hypothesis_characters = sum(len(word) for word in hypothesis_words)
    character = _score(character_counts, reference_characters, hypothesis_characters)
    word = _score(word_counts, len(reference_words), len(hypothesis_words))
    return {
        "evaluated_reference_chars": len(reference),
        "evaluated_hypothesis_chars": len(hypothesis),
        "truncated": truncated,
        "character": character,
        "word": word,
    }


# ---------------------------------------------------------------------------
# Reference-free quality proxies
# ---------------------------------------------------------------------------

def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def text_quality(
    text: str,
    page_texts: list[str] | None = None,
    *,
    max_ngram_chars: int = 100_000,
) -> dict[str, Any]:
    """Heuristic indicators of OCR damage; these are not accuracy measures."""
    body = PAGE_MARKER.sub("", text)
    content = "".join(body.split())
    size = len(content)
    words = body.split()
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    ngram_content = content[:max_ngram_chars]
    four_grams = [ngram_content[index:index + 4] for index in range(max(0, len(ngram_content) - 3))]

    repeated_run = 0
    current_run = 0
    previous = None
    for line in lines:
        if line == previous:
            current_run += 1
        else:
            current_run = 1
            previous = line
        repeated_run = max(repeated_run, current_run)

    pages = [page for page in (page_texts or [])]
    return {
        "characters": len(body),
        "non_space_characters": size,
        "words": len(words),
        "lines": len(lines),
        "unique_character_ratio": _ratio(len(set(content)), size),
        "valid_character_ratio": _ratio(
            sum(1 for char in content
                if char.isprintable() and char != _REPLACEMENT and not _PRIVATE_USE.match(char)),
            size,
        ),
        "replacement_character_ratio": _ratio(content.count(_REPLACEMENT), size),
        "private_use_ratio": _ratio(len(_PRIVATE_USE.findall(content)), size),
        "control_character_ratio": _ratio(len(_CONTROL.findall(body)), len(body)),
        "arabic_ratio": _ratio(len(_ARABIC.findall(content)), size),
        "latin_ratio": _ratio(len(_LATIN.findall(content)), size),
        "digit_ratio": _ratio(len(_DIGIT.findall(content)), size),
        "foreign_script_ratio": _ratio(len(_FOREIGN_SCRIPT.findall(content)), size),
        "repeated_4gram_ratio": _ratio(len(four_grams) - len(set(four_grams)), len(four_grams)),
        "four_gram_sample_chars": len(ngram_content),
        "max_repeated_line_run": repeated_run,
        "single_char_token_ratio": _ratio(sum(1 for word in words if len(word) == 1), len(words)),
        "empty_page_ratio": _ratio(sum(1 for page in pages if not page.strip()), len(pages)),
    }


# ---------------------------------------------------------------------------
# Operational metrics
# ---------------------------------------------------------------------------

def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def latency_stats(values: Iterable[float]) -> dict[str, Any]:
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min_s": round(min(values), 4),
        "mean_s": round(statistics.fmean(values), 4),
        "median_s": round(statistics.median(values), 4),
        "p95_s": round(_percentile(values, 0.95), 4),
        "max_s": round(max(values), 4),
        "total_s": round(sum(values), 4),
    }


def _ocr_metrics(page_reports: list["PageReport"], wall_s: float) -> dict[str, Any]:
    routed = [report for report in page_reports
              if report.route.value in {"SCANNED", "SUSPECT"}]
    failed = [report for report in routed if report.ocr_failed]
    cached = [report for report in routed if report.ocr_from_cache]
    latencies = [report.ocr_latency_s for report in routed
                 if report.ocr_latency_s is not None and not report.ocr_from_cache]
    confidences = [report.ocr_confidence for report in routed if report.ocr_confidence is not None]
    stats = latency_stats(latencies)
    attempted = len(routed) - len(cached)
    return {
        "pages_routed": len(routed),
        "pages_attempted": attempted,
        "pages_succeeded": len(routed) - len(failed),
        "pages_failed": len(failed),
        "failed_page_numbers": [report.page_number for report in failed],
        "pages_from_cache": len(cached),
        "failure_rate": _ratio(len(failed), attempted),
        "cache_hit_rate": _ratio(len(cached), len(routed)),
        "latency": stats,
        "pages_per_minute": (
            round(attempted / wall_s * 60, 2) if wall_s > 0 and attempted else None),
        "document_wall_s": round(wall_s, 3),
        "confidence": {
            "count": len(confidences),
            "mean": round(statistics.fmean(confidences), 6) if confidences else None,
            "min": round(min(confidences), 6) if confidences else None,
            "max": round(max(confidences), 6) if confidences else None,
        },
    }


def _cleaning_metrics(page_reports: list["PageReport"], raw_text: str, cleaned_text: str) -> dict[str, Any]:
    hits: Counter = Counter()
    for report in page_reports:
        hits.update(report.cleaning_hits or {})
    return {
        "before_chars": len(raw_text),
        "after_chars": len(cleaned_text),
        "delta_chars": len(cleaned_text) - len(raw_text),
        "hit_counts": dict(sorted(hits.items())),
        "total_hits": sum(hits.values()),
    }


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def find_reference(source: Path, config: "PipelineConfig") -> Path | None:
    """Locate the first existing reference transcription for a source file."""
    stem = source.stem
    candidates = [
        config.benchmark_reference_dir / f"{stem}.txt",
        config.benchmark_reference_dir / f"{stem}.ref.txt",
        source.with_name(f"{stem}.ref.txt"),
        source.parent / "reference" / f"{stem}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _per_page_metrics(
    reference_text: str,
    hypothesis_text: str,
    config: "PipelineConfig",
) -> list[dict[str, Any]]:
    reference_pages = split_pages(reference_text)
    hypothesis_pages = split_pages(hypothesis_text)
    common = sorted(set(reference_pages) & set(hypothesis_pages))
    if not common:
        return []
    results = []
    for number in common:
        reference = normalize_for_config(reference_pages[number], config)
        hypothesis = normalize_for_config(hypothesis_pages[number], config)
        comparison = compare_texts(reference, hypothesis, config.benchmark_max_eval_chars)
        results.append({"page": number, **comparison})
    return results


def _reference_metrics(
    raw_text: str,
    cleaned_text: str,
    reference_text: str,
    reference_path: Path,
    config: "PipelineConfig",
) -> dict[str, Any]:
    reference = normalize_for_config(reference_text, config)
    raw = normalize_for_config(raw_text, config)
    cleaned = normalize_for_config(cleaned_text, config)
    max_chars = config.benchmark_max_eval_chars
    cleaned_comparison = compare_texts(reference, cleaned, max_chars)
    raw_comparison = compare_texts(reference, raw, max_chars)
    per_page = _per_page_metrics(reference_text, cleaned_text, config)
    raw_cer = raw_comparison["character"]["error_rate"]
    cleaned_cer = cleaned_comparison["character"]["error_rate"]
    raw_wer = raw_comparison["word"]["error_rate"]
    cleaned_wer = cleaned_comparison["word"]["error_rate"]
    return {
        "matched": True,
        "path": str(reference_path),
        "reference_chars": len(reference_text),
        "page_aligned": bool(per_page),
        "cleaned": cleaned_comparison,
        "raw": raw_comparison,
        "raw_vs_cleaned": {
            "cer_before": raw_cer,
            "cer_after": cleaned_cer,
            "cer_improvement": (
                round(raw_cer - cleaned_cer, 6)
                if raw_cer is not None and cleaned_cer is not None else None),
            "wer_before": raw_wer,
            "wer_after": cleaned_wer,
            "wer_improvement": (
                round(raw_wer - cleaned_wer, 6)
                if raw_wer is not None and cleaned_wer is not None else None),
        },
        "per_page": per_page,
    }


# ---------------------------------------------------------------------------
# Document and batch records
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stage_snapshot(config: "PipelineConfig") -> dict[str, Any]:
    """Read the latest batch summary of each optional stage, when present."""
    snapshot: dict[str, Any] = {"scope": "latest batch snapshot under stage_output_dir"}
    for name in STAGES:
        path = config.stage_output_dir / name / "summary.json"
        if path.is_file():
            try:
                snapshot[name] = {"available": True, "path": str(path),
                                  "summary": json.loads(path.read_text(encoding="utf-8"))}
            except (OSError, json.JSONDecodeError) as exc:
                snapshot[name] = {"available": False, "path": str(path), "error": str(exc)}
        else:
            snapshot[name] = {"available": False, "path": str(path)}
    return snapshot


def build_document_metrics(
    source: Path,
    config: "PipelineConfig",
    *,
    page_reports: list["PageReport"],
    cleaned_pages: list[str],
    raw_text: str,
    cleaned_text: str,
    source_hash: str,
    status: str,
    wall_s: float,
    reference_path: Path | None,
) -> dict[str, Any]:
    """Assemble the complete per-document benchmark record."""
    routes = Counter(report.route.value for report in page_reports)
    ocr = _ocr_metrics(page_reports, wall_s)
    reference: dict[str, Any] = {"matched": False}
    if reference_path is not None:
        try:
            reference = _reference_metrics(
                raw_text, cleaned_text,
                reference_path.read_text(encoding="utf-8-sig", errors="replace"),
                reference_path, config,
            )
        except OSError as exc:
            reference = {"matched": False, "error": str(exc), "path": str(reference_path)}
    else:
        reference["reason"] = "No reference transcription found for this source."
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "source": str(source),
        "source_name": source.name,
        "source_sha256": source_hash,
        "status": status,
        "provider": config.ocr_provider,
        "model": config.ocr_model,
        "base_url": config.ocr_base_url,
        "languages": list(config.languages),
        "render_dpi": config.render_dpi,
        "page_count": len(page_reports),
        "routes": {
            "digital": routes.get("DIGITAL", 0),
            "scanned": routes.get("SCANNED", 0),
            "suspect": routes.get("SUSPECT", 0),
        },
        "timing": {
            "document_wall_s": round(wall_s, 3),
        },
        "ocr": ocr,
        "text_quality": text_quality(
            cleaned_text, cleaned_pages,
            max_ngram_chars=config.benchmark_max_eval_chars),
        "cleaning": _cleaning_metrics(page_reports, raw_text, cleaned_text),
        "reference": reference,
        "stages": stage_snapshot(config),
    }


def metrics_path(output_dir: Path, stem: str) -> Path:
    return output_dir / f"{stem}.metrics.json"


def history_path(config: "PipelineConfig") -> Path:
    return config.benchmark_dir / "history.jsonl"


def history_row(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten one document record into a comparable history row."""
    reference = metrics.get("reference", {})
    cleaned = reference.get("cleaned", {}) if reference.get("matched") else {}
    character = cleaned.get("character", {})
    word = cleaned.get("word", {})
    delta = reference.get("raw_vs_cleaned", {}) if reference.get("matched") else {}
    ocr = metrics.get("ocr", {})
    quality = metrics.get("text_quality", {})
    latency = ocr.get("latency", {})
    return {
        "generated_at": metrics.get("generated_at"),
        "source": metrics.get("source"),
        "source_name": metrics.get("source_name"),
        "source_sha256": metrics.get("source_sha256"),
        "status": metrics.get("status"),
        "provider": metrics.get("provider"),
        "model": metrics.get("model"),
        "base_url": metrics.get("base_url"),
        "page_count": metrics.get("page_count", 0),
        "ocr_pages": ocr.get("pages_routed", 0),
        "ocr_attempted": ocr.get("pages_attempted", 0),
        "ocr_failed": ocr.get("pages_failed", 0),
        "ocr_from_cache": ocr.get("pages_from_cache", 0),
        "characters": metrics.get("cleaning", {}).get("after_chars", 0),
        "document_wall_s": ocr.get("document_wall_s"),
        "ocr_latency_total_s": latency.get("total_s"),
        "ocr_latency_mean_s": latency.get("mean_s"),
        "ocr_pages_per_minute": ocr.get("pages_per_minute"),
        "valid_character_ratio": quality.get("valid_character_ratio"),
        "reference_matched": bool(reference.get("matched")),
        "page_aligned": bool(reference.get("page_aligned")),
        "evaluated_chars": cleaned.get("evaluated_reference_chars"),
        "truncated": cleaned.get("truncated"),
        "cer": character.get("error_rate"),
        "cer_accuracy": character.get("accuracy"),
        "cer_precision": character.get("precision"),
        "cer_recall": character.get("recall"),
        "cer_f1": character.get("f1"),
        "wer": word.get("error_rate"),
        "wer_precision": word.get("precision"),
        "wer_recall": word.get("recall"),
        "wer_f1": word.get("f1"),
        "raw_cer": delta.get("cer_before"),
        "cer_improvement": delta.get("cer_improvement"),
        "raw_wer": delta.get("wer_before"),
        "wer_improvement": delta.get("wer_improvement"),
    }


def record_batch(config: "PipelineConfig", results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Append completed per-document rows to history and refresh the comparison."""
    if not config.benchmark_enabled:
        return None
    rows = [result["benchmark_row"] for result in results if result.get("benchmark_row")]
    if not rows:
        return None
    path = history_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    comparison = compare_records(read_jsonl(path), history_path=str(path))
    write_comparison(config, comparison)
    return {"history_path": str(path), "documents": len(rows),
            "comparison_path": str(config.benchmark_dir / "comparison.json")}


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return round(statistics.fmean(clean), 6) if clean else None


def compare_records(records: list[dict[str, Any]], *, history_path: str = "") -> dict[str, Any]:
    """Group history rows by provider/model and aggregate comparable metrics."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("provider", "")), str(record.get("model", "")))
        groups.setdefault(key, []).append(record)

    comparisons = []
    for (provider, model), rows in sorted(groups.items()):
        with_reference = [row for row in rows if row.get("reference_matched")]
        ocr_pages = sum(row.get("ocr_pages") or 0 for row in rows)
        ocr_attempted = sum(row.get("ocr_attempted") or 0 for row in rows)
        ocr_failed = sum(row.get("ocr_failed") or 0 for row in rows)
        latency_total = sum(row.get("ocr_latency_total_s") or 0 for row in rows)
        comparisons.append({
            "provider": provider,
            "model": model,
            "base_urls": sorted({str(row.get("base_url")) for row in rows if row.get("base_url")}),
            "documents": len(rows),
            "documents_with_reference": len(with_reference),
            "pages": sum(row.get("page_count") or 0 for row in rows),
            "ocr_pages": ocr_pages,
            "ocr_attempted": ocr_attempted,
            "ocr_failed": ocr_failed,
            "ocr_failure_rate": _ratio(ocr_failed, ocr_attempted),
            "ocr_cache_hits": sum(row.get("ocr_from_cache") or 0 for row in rows),
            "characters": sum(row.get("characters") or 0 for row in rows),
            "ocr_latency_total_s": round(latency_total, 4),
            "ocr_pages_per_minute": (
                round(ocr_attempted / latency_total * 60, 2) if latency_total else None),
            "mean_document_wall_s": _mean(row.get("document_wall_s") for row in rows),
            "mean_valid_character_ratio": _mean(row.get("valid_character_ratio") for row in rows),
            "mean_cer": _mean(row.get("cer") for row in with_reference),
            "mean_cer_accuracy": _mean(row.get("cer_accuracy") for row in with_reference),
            "mean_cer_precision": _mean(row.get("cer_precision") for row in with_reference),
            "mean_cer_recall": _mean(row.get("cer_recall") for row in with_reference),
            "mean_cer_f1": _mean(row.get("cer_f1") for row in with_reference),
            "mean_wer": _mean(row.get("wer") for row in with_reference),
            "mean_wer_precision": _mean(row.get("wer_precision") for row in with_reference),
            "mean_wer_recall": _mean(row.get("wer_recall") for row in with_reference),
            "mean_wer_f1": _mean(row.get("wer_f1") for row in with_reference),
            "mean_raw_cer": _mean(row.get("raw_cer") for row in with_reference),
            "mean_cer_improvement": _mean(row.get("cer_improvement") for row in with_reference),
            "runs": [{
                "generated_at": row.get("generated_at"),
                "source_name": row.get("source_name"),
                "cer": row.get("cer"),
                "wer": row.get("wer"),
                "raw_cer": row.get("raw_cer"),
                "ocr_pages_per_minute": row.get("ocr_pages_per_minute"),
            } for row in with_reference],
        })

    # Best model per metric is the lowest CER/WER; useful when picking a provider.
    ranked = [item for item in comparisons if item["mean_cer"] is not None]
    best = min(ranked, key=lambda item: item["mean_cer"]) if ranked else None
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "history_path": history_path,
        "documents": len(records),
        "groups": comparisons,
        "lowest_mean_cer": (
            {"provider": best["provider"], "model": best["model"],
             "mean_cer": best["mean_cer"]} if best else None),
    }


_CSV_FIELDS = (
    "provider", "model", "documents", "documents_with_reference", "pages", "ocr_pages",
    "ocr_attempted", "ocr_failure_rate", "ocr_pages_per_minute", "mean_document_wall_s",
    "mean_valid_character_ratio", "mean_cer", "mean_cer_accuracy", "mean_cer_precision",
    "mean_cer_recall", "mean_cer_f1", "mean_wer", "mean_wer_precision", "mean_wer_recall",
    "mean_wer_f1", "mean_raw_cer", "mean_cer_improvement",
)


def comparison_csv(comparison: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for group in comparison["groups"]:
        writer.writerow(group)
    return buffer.getvalue()


def write_comparison(config: "PipelineConfig", comparison: dict[str, Any]) -> None:
    write_json(config.benchmark_dir / "comparison.json", comparison)
    atomic_text(config.benchmark_dir / "comparison.csv", comparison_csv(comparison))


def load_comparison(config: "PipelineConfig") -> dict[str, Any]:
    """Read history and build a comparison without processing new files."""
    path = history_path(config)
    records = read_jsonl(path) if path.is_file() else []
    return compare_records(records, history_path=str(path))
