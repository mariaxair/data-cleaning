"""Conservative, deterministic post-processing for extracted OCR text."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
               config: PipelineConfig | None = None, format_cleaning: bool = True) -> CleaningResult:
    """Repair mojibake and apply traceable, conservative format cleanups."""
    del languages  # Language hints are retained in the API for future rules.
    if not format_cleaning:
        if config is None or not config.semantic_rules_enabled:
            return CleaningResult(text, {})
        corrected, counts = apply_rules(text, config.rules_data_path, config.wrong_words_path)
        repeated, _ = apply_rules(corrected, config.rules_data_path, config.wrong_words_path)
        if repeated != corrected:
            raise ValueError("Semantic rules are not idempotent; review the configured mappings.")
        return CleaningResult(corrected, counts)
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


def clean_files(paths: list[Path], config: PipelineConfig, *, format_cleaning: bool = True,
                force: bool = False, progress: Callable[[str], None] | None = None) -> dict:
    """Clean exported text page by page, retaining partial-source provenance."""
    from collections import Counter
    from dataclasses import replace
    import hashlib
    from .artifacts import atomic_text, fingerprint, write_json
    from .settings import load_json
    from .processor import _join_pages

    output = config.stage_output_dir / "cleaning"
    counts, results = Counter(), []
    for path in paths:
        path = Path(path)
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8-sig")
            previous = load_json(path.with_suffix(".report.json"))
            tables = ([p.read_text(encoding="utf-8") if p.is_file() else ""
                       for p in (config.rules_data_path, config.wrong_words_path)]
                      if config.semantic_rules_enabled else [])
            identity = fingerprint([str(path.resolve()), hashlib.sha256(raw).hexdigest(), previous, format_cleaning,
                                    config.semantic_rules_enabled, config.languages,
                                    config.include_page_markers, config.page_marker_template,
                                    tables])
            # A path component prevents collisions between equal basenames from different folders.
            stem = path.stem + "-" + fingerprint(str(path.resolve()))[:8]
            target, report = output / f"{stem}.txt", output / f"{stem}.report.json"
            saved = load_json(report)
            if not force and target.is_file() and saved.get("cleaning_identity") == identity:
                counts.update(saved.get("cleaning_hits", {}))
                results.append({**saved, "status": "skipped", "output_path": str(target), "report_path": str(report)})
                continue
            pages = previous.get("page_texts") if previous.get("stage") == "ocr" else None
            markers = previous.get("page_markers", [config.include_page_markers, config.page_marker_template])
            source_config = replace(config, include_page_markers=markers[0], page_marker_template=markers[1])
            per_page = isinstance(pages, list) and _join_pages(pages, source_config) == text.replace("\r\n", "\n")
            if not per_page:
                pages = [text]
            cleaned = [clean_text(page, config.languages, config=config, format_cleaning=format_cleaning)
                       for page in pages]
            hits = Counter()
            for page in cleaned:
                hits.update(page.hit_counts)
            body = _join_pages([page.text for page in cleaned], config) if per_page else cleaned[0].text + "\n"
            atomic_text(target, body)
            record = {k: v for k, v in previous.items() if k != "page_texts"}
            if per_page:
                record["pages"] = [{**page, "cleaning_hits": result.hit_counts}
                                   for page, result in zip(previous.get("pages", []), cleaned)]
            record.update(stage="cleaning", source=previous.get("source", str(path)),
                          cleaning_input=str(path), cleaning_identity=identity,
                          status="partial" if previous.get("failed_pages") or previous.get("status") == "partial" else "complete",
                          format_cleaning=format_cleaning, semantic_rules=config.semantic_rules_enabled,
                          cleaning_hits=dict(hits), total_characters=len(body))
            write_json(report, record)
            counts.update(hits)
            results.append({**record, "output_path": str(target), "report_path": str(report),
                            "original_text": text, "cleaned_text": body})
            if progress:
                progress(f"Cleaned {path.name}")
        except (OSError, ValueError, TypeError):
            results.append({"source": str(path), "status": "failed",
                            "error": "Cleaning failed; check the source, report and correction rules."})
    summary = {"results": results, "files_processed": sum(r["status"] in {"complete", "partial"} for r in results),
               "files_skipped": sum(r["status"] == "skipped" for r in results),
               "files_failed": sum(r["status"] == "failed" for r in results), "hit_counts": dict(counts)}
    write_json(output / "rules-report.json", dict(counts))
    write_json(output / "summary.json", {k: v for k, v in summary.items() if k != "results"})
    return summary
