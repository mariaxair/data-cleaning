"""Resumable orchestration for page routing, OCR, and cleaning."""

import hashlib
import json
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from config import PipelineConfig
from .cache import PipelineCache
from .cleaning import clean_text
from .extraction import PdfPageSource, extract_docx, extract_text_file
from .models import PageReport, PageRoute
from .ocr.registry import create_adapter
from .routing import classify_page, measure_page

ProgressCallback = Callable[[str], None]


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
    if config.semantic_rules_enabled:
        values["semantic_rules"] = [
            "v1",
            _sha256(config.rules_data_path.read_bytes()) if config.rules_data_path.is_file() else "",
            _sha256(config.wrong_words_path.read_bytes()) if config.wrong_words_path.is_file() else "",
        ]
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


def _ocr_cache_key(image: bytes, config: PipelineConfig) -> str:
    """Identify one rendered page under the current provider settings."""
    material = "|".join([
        _sha256(image), config.ocr_provider.lower(), config.ocr_model,
        config.ocr_base_url.strip(), ",".join(config.languages),
    ])
    return _sha256(material.encode())


def _ocr_page(
    image: bytes,
    config: PipelineConfig,
    adapter: Any,
    cache: PipelineCache,
    semaphore: threading.BoundedSemaphore | None,
    progress: ProgressCallback | None,
    page_number: int,
    total_pages: int,
    source_name: str,
) -> tuple[str, str | None, list[str], float | None, float | None, bool]:
    """Return (text, provider, warnings, latency, confidence, from_cache)."""
    cache_key = _ocr_cache_key(image, config)
    cached = cache.get_ocr(cache_key)
    if cached is not None:
        if progress:
            progress(f"OCR page {page_number}/{total_pages} cache hit: {source_name}")
        return (cached.text, cached.provider,
                ["OCR result loaded from page cache."], None, cached.confidence, True)
    if progress:
        progress(f"OCR page {page_number}/{total_pages} queued: {source_name}")
    # The semaphore caps simultaneous provider calls across every file. A None
    # semaphore means OCR_CONCURRENCY=0: every page may call the provider at once.
    # Only the provider call is timed so the metric reflects model latency, not queueing.
    with (semaphore if semaphore is not None else nullcontext()):
        if progress:
            progress(f"OCR page {page_number}/{total_pages} sending: {source_name}")
        started = time.perf_counter()
        result = adapter.ocr_page(image, config.languages)
        latency = time.perf_counter() - started
    cache.put_ocr(cache_key, result)
    if progress:
        progress(
            f"OCR page {page_number}/{total_pages} done in {latency:.1f}s "
            f"({len(result.text)} chars): {source_name}")
    return result.text, result.provider, [], latency, result.confidence, False


def _process_pdf(
    source: Path,
    config: PipelineConfig,
    adapter: Any,
    cache: PipelineCache,
    semaphore: threading.BoundedSemaphore | None,
    progress: ProgressCallback | None,
) -> tuple[list[str], list[str], list[PageReport]]:
    """Route every page, OCR the scanned ones in parallel, and keep page order."""
    with PdfPageSource(source, config.render_dpi) as pdf:
        total_pages = pdf.page_count
        texts: list[str] = [""] * total_pages
        page_reports: list[PageReport] = []
        ocr_pages: list[int] = []

        for number, text, image_ratio in pdf.pages():
            signals = measure_page(text, image_ratio)
            route, warnings = classify_page(text, signals, config)
            page_reports.append(PageReport(number, route, signals, warnings))
            if route in (PageRoute.SCANNED, PageRoute.SUSPECT):
                ocr_pages.append(number)
            else:
                texts[number - 1] = text

        def process_page(number: int) -> tuple[str, str | None, list[str], float | None, float | None, bool]:
            # Rendering happens inside the worker so only in-flight pages are
            # held in memory; PdfPageSource serialises the PyMuPDF call.
            image = pdf.render(number)
            return _ocr_page(
                image, config, adapter, cache, semaphore, progress,
                number, total_pages, source.name,
            )

        if progress:
            progress(f"{source.name}: {total_pages} pages, {len(ocr_pages)} routed to OCR")
        if ocr_pages:
            # PAGE_CONCURRENCY=0 submits every OCR page at once; the shared
            # semaphore (or its absence) still decides how many calls run.
            workers = (len(ocr_pages) if config.page_concurrency <= 0
                       else max(1, min(config.page_concurrency, len(ocr_pages))))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr-page") as pool:
                futures = {pool.submit(process_page, number): number for number in ocr_pages}
                for future in as_completed(futures):
                    number = futures[future]
                    report = page_reports[number - 1]
                    try:
                        text, provider, warnings, latency, confidence, from_cache = future.result()
                    except Exception as exc:  # A failed page must not abort the PDF.
                        text, provider, warnings = "", None, [str(exc)]
                        latency, confidence, from_cache = None, None, False
                        report.ocr_failed = True
                        if progress:
                            progress(f"OCR page {number}/{total_pages} failed: {source.name}")
                        logging.getLogger("corpus_pipeline").warning(
                            f"OCR failed for {source.name}, page {number}: {exc}"
                        )
                    texts[number - 1] = text
                    report.ocr_provider = provider
                    report.ocr_latency_s = latency
                    report.ocr_confidence = confidence
                    report.ocr_from_cache = from_cache
                    report.warnings.extend(warnings)

        if progress:
            progress(f"Cleaning {source.name}: {len(page_reports)} pages")
        cleaned_pages: list[str] = []
        for index, page_text in enumerate(texts):
            cleaned = clean_text(page_text, config.languages, config=config)
            page_reports[index].cleaning_hits = cleaned.hit_counts
            cleaned_pages.append(cleaned.text)
        return texts, cleaned_pages, page_reports


def process_file(
    path: str | Path,
    config: PipelineConfig,
    *,
    ocr_adapter: Any | None = None,
    progress: ProgressCallback | None = None,
    force: bool = False,
    ocr_semaphore: threading.BoundedSemaphore | None = None,
) -> dict[str, Any]:
    """Process one supported file and emit TXT plus a JSON route report."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"Reading {source.name}")
    source_hash = _sha256(source.read_bytes())
    source_key = str(source.resolve())
    config_hash = _config_hash(config)
    cache = PipelineCache(config.cache_db)
    adapter = ocr_adapter or create_adapter(config)
    semaphore = ocr_semaphore
    if semaphore is None and config.ocr_concurrency > 0:
        semaphore = threading.BoundedSemaphore(max(1, config.ocr_concurrency))
    stem = _safe_stem(source, source_hash)
    output_path = config.output_dir / f"{stem}.txt"
    report_path = config.output_dir / f"{stem}.report.json"

    if not force and cache.is_complete(source_key, source_hash, config_hash):
        existing_metrics = config.output_dir / f"{stem}.metrics.json"
        if progress:
            progress(f"Skipped (already processed): {source.name}")
        return {
            "source": str(source), "status": "skipped", "output_path": str(output_path),
            "report_path": str(report_path),
            "metrics_path": str(existing_metrics) if existing_metrics.is_file() else None,
            "message": "Already processed with the current settings.",
        }

    page_reports: list[PageReport] = []
    original_pages: list[str] = []
    cleaned_pages: list[str] = []
    extension = source.suffix.lower()

    if extension == ".pdf":
        original_pages, cleaned_pages, page_reports = _process_pdf(
            source, config, adapter, cache, semaphore, progress,
        )
    elif extension in {".txt", ".md"}:
        text = extract_text_file(source)
        original_pages = [text]
        cleaned = clean_text(text, config.languages, config=config)
        cleaned_pages = [cleaned.text]
        signals = measure_page(text, 0.0)
        page_reports.append(PageReport(1, PageRoute.DIGITAL, signals, cleaning_hits=cleaned.hit_counts))
    elif extension == ".docx":
        text = extract_docx(source)
        original_pages = [text]
        cleaned = clean_text(text, config.languages, config=config)
        cleaned_pages = [cleaned.text]
        signals = measure_page(text, 0.0)
        page_reports.append(PageReport(1, PageRoute.DIGITAL, signals, cleaning_hits=cleaned.hit_counts))
    else:
        raise ValueError(f"Unsupported file type '{extension}'. Use PDF, TXT, MD, or DOCX.")

    output_text = _join_pages(cleaned_pages, config)
    if progress:
        progress(f"Writing {output_path.name}")
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
    if config.semantic_rules_enabled:
        from .rules.tables import load_table
        from .settings import load_json
        domain = load_json(config.rules_data_path)
        report_data["semantic_rules"] = {
            "enabled": True,
            "degraded": not all((domain.get("foreign_glyphs"), domain.get("dates"),
                                 domain.get("footer_markers"), load_table(config.wrong_words_path))),
            "note": "Only explicit reviewed mappings are applied; empty or missing families are no-ops.",
        }
    raw_text = _join_pages(original_pages, config)
    metrics_file: Path | None = None
    benchmark_row: dict[str, Any] | None = None
    if config.benchmark_enabled:
        from .artifacts import write_json
        from .benchmark import (build_document_metrics, find_reference, history_row,
                                metrics_path as document_metrics_path)
        try:
            benchmark = build_document_metrics(
                source, config,
                page_reports=page_reports,
                cleaned_pages=cleaned_pages,
                raw_text=raw_text,
                cleaned_text=output_text,
                source_hash=source_hash,
                status=status,
                wall_s=time.perf_counter() - started,
                reference_path=find_reference(source, config),
            )
            metrics_file = document_metrics_path(config.output_dir, stem)
            write_json(metrics_file, benchmark)
            benchmark_row = history_row(benchmark)
        except OSError as exc:
            # Metrics are diagnostics; never fail extraction because they cannot be written.
            metrics_file = None
            benchmark_row = None
            logging.getLogger("corpus_pipeline").warning(
                f"benchmark metrics not written for {source.name}: {exc}")
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    cache.mark_processed(source_key, source_hash, config_hash, output_path, report_path, status)
    if progress:
        progress(
            f"Finished {source.name}: {status}, {summary}, "
            f"{report_data['total_characters']} chars")
    logging.getLogger("corpus_pipeline").info(
        f"processed file={source.name} status={status} pages={len(page_reports)} "
        f"ocr_pages={scanned_count + suspect_count} characters={report_data['total_characters']}"
    )
    return {
        **report_data,
        "output_path": str(output_path),
        "report_path": str(report_path),
        "metrics_path": str(metrics_file) if metrics_file else None,
        "benchmark_row": benchmark_row,
        "original_text": raw_text,
        "cleaned_text": output_text,
    }


def process_batch(
    paths: list[str | Path],
    config: PipelineConfig,
    *,
    progress: ProgressCallback | None = None,
    force: bool = False,
    ocr_adapter: Any | None = None,
) -> dict[str, Any]:
    """Process a group of files concurrently and summarize outcomes in input order."""
    targets = [Path(path) for path in paths]
    total = len(targets)
    # OCR_CONCURRENCY=0 means no batch-wide cap; each page call starts immediately.
    semaphore = (threading.BoundedSemaphore(max(1, config.ocr_concurrency))
                 if config.ocr_concurrency > 0 else None)
    adapter = ocr_adapter or create_adapter(config)
    progress_lock = threading.Lock()

    def emit(message: str) -> None:
        # Worker threads share one callback, so serialise it for the UI.
        if progress:
            with progress_lock:
                progress(message)

    def run(index: int, path: Path) -> dict[str, Any]:
        emit(f"Processing file {index} of {total}: {path.name}")
        try:
            return process_file(
                path, config, ocr_adapter=adapter, progress=emit,
                force=force, ocr_semaphore=semaphore,
            )
        except Exception as exc:
            logging.getLogger("corpus_pipeline").error(
                f"file failed name={path.name}: {exc}"
            )
            return {"source": str(path), "status": "failed", "error": str(exc)}

    # FILE_CONCURRENCY=0 means every document starts at once. Each document still
    # applies its own page limit and the shared OCR cap.
    workers = (total if config.file_concurrency <= 0
               else max(1, min(config.file_concurrency, total)))
    if workers == 1:
        results = [run(index, path) for index, path in enumerate(targets, start=1)]
    else:
        ordered: list[dict[str, Any] | None] = [None] * total
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="corpus-file") as pool:
            futures = {
                pool.submit(run, index, path): index
                for index, path in enumerate(targets, start=1)
            }
            for future in as_completed(futures):
                ordered[futures[future] - 1] = future.result()
        results = [result for result in ordered if result is not None]

    total_chars = sum(result.get("total_characters", 0) for result in results)
    ocr_pages = sum(result.get("pages_routed_to_ocr", 0) for result in results)
    benchmark = None
    if config.benchmark_enabled:
        from .benchmark import record_batch
        try:
            benchmark = record_batch(config, results)
        except (OSError, ValueError) as exc:
            logging.getLogger("corpus_pipeline").warning(f"benchmark history not written: {exc}")
    return {
        "files_processed": sum(r.get("status") in {"complete", "partial"} for r in results),
        "files_skipped": sum(r.get("status") == "skipped" for r in results),
        "files_failed": sum(r.get("status") == "failed" for r in results),
        "pages_routed_to_ocr": ocr_pages,
        "provider": config.ocr_provider,
        "total_characters": total_chars,
        "benchmark": benchmark,
        "results": results,
    }
