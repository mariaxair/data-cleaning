"""Resumable orchestration for page routing, OCR, and cleaning."""

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from config import PipelineConfig
from .cache import PipelineCache
from .cleaning import clean_text
from .extraction import extract_docx, extract_pdf_pages, extract_text_file
from .models import PageReport, PageRoute
from .ocr.registry import create_adapter
from .routing import classify_page, measure_page


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _config_hash(config: PipelineConfig) -> str:
    """Hash output-affecting settings, excluding API credentials."""
    values = {
        "languages": config.languages,
        "markers": [config.include_page_markers, config.page_marker_template],
        "routing": [config.min_extracted_chars, config.min_valid_char_ratio, config.full_page_image_ratio],
        "dpi": config.render_dpi,
        "provider": config.ocr_provider,
        "base_url": config.ocr_base_url,
        "model": config.ocr_model,
    }
    return _sha256(json.dumps(values, sort_keys=True).encode())


def _safe_stem(path: Path, source_hash: str) -> str:
    return f"{path.stem}-{source_hash[:10]}"


def _join_pages(pages: list[str], config: PipelineConfig) -> str:
    chunks: list[str] = []
    for number, page_text in enumerate(pages, start=1):
        if config.include_page_markers:
            chunks.append(config.page_marker_template.format(page=number))
        chunks.append(page_text)
    return "\n".join(chunks).strip() + ("\n" if chunks else "")


def process_file(
    path: str | Path,
    config: PipelineConfig,
    *,
    ocr_adapter: Any | None = None,
    progress: Callable[[str], None] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Process one supported file and emit TXT plus a JSON route report."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(source.read_bytes())
    source_key = str(source.resolve())
    config_hash = _config_hash(config)
    cache = PipelineCache(config.cache_db)
    adapter = ocr_adapter or create_adapter(config)
    stem = _safe_stem(source, source_hash)
    output_path = config.output_dir / f"{stem}.txt"
    report_path = config.output_dir / f"{stem}.report.json"

    if not force and cache.is_complete(source_key, source_hash, config_hash):
        return {
            "source": str(source), "status": "skipped", "output_path": str(output_path),
            "report_path": str(report_path), "message": "Already processed with the current settings.",
        }

    page_reports: list[PageReport] = []
    original_pages: list[str] = []
    cleaned_pages: list[str] = []
    extension = source.suffix.lower()

    if extension == ".pdf":
        extracted_pages = extract_pdf_pages(source, config.render_dpi)
        for page in extracted_pages:
            signals = measure_page(page.text, page.image_ratio)
            route, warnings = classify_page(page.text, signals, config)
            report = PageReport(page.number, route, signals, warnings)
            if route in (PageRoute.SCANNED, PageRoute.SUSPECT):
                key_material = "|".join([
                    _sha256(page.image_bytes), config.ocr_provider.lower(), config.ocr_model,
                    config.ocr_base_url.strip(), ",".join(config.languages),
                ])
                ocr_key = _sha256(key_material.encode())
                cached = cache.get_ocr(ocr_key)
                if cached is not None:
                    page_text = cached.text
                    report.ocr_provider = cached.provider
                    report.warnings.append("OCR result loaded from page cache.")
                else:
                    try:
                        if progress:
                            progress(f"OCR page {page.number} of {len(extracted_pages)}: {source.name}")
                        result = adapter.ocr_page(page.image_bytes, config.languages)
                        page_text = result.text
                        report.ocr_provider = result.provider
                        cache.put_ocr(ocr_key, result)
                    except Exception as exc:  # A failed page must not abort the PDF.
                        page_text = ""
                        report.ocr_failed = True
                        report.warnings.append(str(exc))
                        logging.getLogger("corpus_pipeline").warning(
                            f"OCR failed for {source.name}, page {page.number}: {exc}"
                        )
            else:
                page_text = page.text
            original_pages.append(page_text)
            cleaned = clean_text(page_text, config.languages)
            cleaned_pages.append(cleaned.text)
            report.cleaning_hits = cleaned.hit_counts
            page_reports.append(report)
    elif extension in {".txt", ".md"}:
        text = extract_text_file(source)
        original_pages = [text]
        cleaned = clean_text(text, config.languages)
        cleaned_pages = [cleaned.text]
        signals = measure_page(text, 0.0)
        page_reports.append(PageReport(1, PageRoute.DIGITAL, signals, cleaning_hits=cleaned.hit_counts))
    elif extension == ".docx":
        text = extract_docx(source)
        original_pages = [text]
        cleaned = clean_text(text, config.languages)
        cleaned_pages = [cleaned.text]
        signals = measure_page(text, 0.0)
        page_reports.append(PageReport(1, PageRoute.DIGITAL, signals, cleaning_hits=cleaned.hit_counts))
    else:
        raise ValueError(f"Unsupported file type '{extension}'. Use PDF, TXT, MD, or DOCX.")

    output_text = _join_pages(cleaned_pages, config)
    output_path.write_text(output_text, encoding="utf-8")
    counts = Counter(report.route.value for report in page_reports)
    failed_pages = [r.page_number for r in page_reports if r.ocr_failed]
    scanned_count = counts.get(PageRoute.SCANNED.value, 0)
    suspect_count = counts.get(PageRoute.SUSPECT.value, 0)
    digital_count = counts.get(PageRoute.DIGITAL.value, 0)
    summary = f"{digital_count} digital / {scanned_count} OCR / {suspect_count} suspect"
    status = "partial" if failed_pages else "complete"
    report_data = {
        "source": str(source),
        "source_sha256": source_hash,
        "status": status,
        "detected_type": (
            "mixed" if digital_count and scanned_count + suspect_count
            else "scanned" if scanned_count + suspect_count
            else "digital"
        ),
        "route_summary": summary,
        "page_count": len(page_reports),
        "pages_routed_to_ocr": scanned_count + suspect_count,
        "failed_pages": failed_pages,
        "provider": config.ocr_provider,
        "total_characters": sum(len(text) for text in cleaned_pages),
        "warnings": [warning for item in page_reports for warning in item.warnings],
        "pages": [item.to_dict() for item in page_reports],
    }
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    cache.mark_processed(source_key, source_hash, config_hash, output_path, report_path, status)
    logging.getLogger("corpus_pipeline").info(
        f"processed file={source.name} status={status} pages={len(page_reports)} "
        f"ocr_pages={scanned_count + suspect_count} characters={report_data['total_characters']}"
    )
    return {
        **report_data,
        "output_path": str(output_path),
        "report_path": str(report_path),
        "original_text": _join_pages(original_pages, config),
        "cleaned_text": output_text,
    }


def process_batch(
    paths: list[str | Path],
    config: PipelineConfig,
    *,
    progress: Callable[[str], None] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Process a group of files independently and summarize outcomes."""
    results: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        if progress:
            progress(f"Processing file {index} of {len(paths)}: {Path(path).name}")
        try:
            results.append(process_file(path, config, progress=progress, force=force))
        except Exception as exc:
            logging.getLogger("corpus_pipeline").error(
                f"file failed name={Path(path).name}: {exc}"
            )
            results.append({"source": str(path), "status": "failed", "error": str(exc)})
    total_chars = sum(result.get("total_characters", 0) for result in results)
    ocr_pages = sum(result.get("pages_routed_to_ocr", 0) for result in results)
    return {
        "files_processed": sum(r.get("status") in {"complete", "partial"} for r in results),
        "files_skipped": sum(r.get("status") == "skipped" for r in results),
        "files_failed": sum(r.get("status") == "failed" for r in results),
        "pages_routed_to_ocr": ocr_pages,
        "provider": config.ocr_provider,
        "total_characters": total_chars,
        "results": results,
    }
