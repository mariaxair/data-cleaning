r"""Run a bounded 1,000-page Heiyu OCR cost sample using project routing.

This file is intentionally standalone. It does not change the extraction pipeline,
its cache, or its outputs. It writes only the sample JSONL and a cost report under
the directory passed with --output.

Example (PowerShell):
    python test_ocr_1000_pages.py --dataset-root C:\Users\acer\Desktop\zte \
        --rpm 20 --input-price-per-1m 0 --output-price-per-1m 0

The provider's current prices are not inferred. Pass the prices for the selected
Heiyu model in USD per one million input/output tokens to make the cost fields
numeric. If the response has no usage object, use --estimated-input-tokens-per-page
and --estimated-output-tokens-per-page for a clearly marked estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

from config import PipelineConfig
from corpus_pipeline.extraction import PdfPageSource
from corpus_pipeline.models import PageRoute
from corpus_pipeline.ocr.heiyu import HeiyuAdapter
from corpus_pipeline.routing import classify_page, measure_page


DEFAULT_ROOT = Path(r"C:\Users\acer\Desktop\zte")
DEFAULT_ENDPOINT = "https://www.heiyucode.com/"
DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_TOTAL_PAGES = 112_487


@dataclass(frozen=True)
class PageTarget:
    pdf: Path
    relative_pdf: str
    page: int
    total_pages_in_pdf: int
    route: str
    route_warnings: tuple[str, ...] = ()


class RateLimiter:
    def __init__(self, rpm: float):
        if rpm <= 0:
            raise ValueError("--rpm must be greater than zero; use a measured provider limit.")
        self.interval = 60.0 / rpm
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        # Reserve a start slot while holding the lock. Workers can render pages
        # concurrently, but request starts remain globally paced.
        with self.lock:
            now = time.monotonic()
            delay = self.next_start - now
            self.next_start = max(self.next_start, now) + self.interval
        if delay > 0:
            time.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded 1,000-page Heiyu OCR cost sample")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT,
                        help="Root containing batch1 and batch2 (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=Path("data/ocr-1000-test"),
                        help="Output directory for JSONL and report")
    parser.add_argument("--pages", type=int, default=1000,
                        help="Pages to submit; keep at 1000 for the requested sample")
    parser.add_argument("--rpm", type=float, default=20,
                        help="Maximum request starts per minute; default is conservative 20")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Simultaneous page workers; matches the project's default OCR_CONCURRENCY=4")
    parser.add_argument("--model", default=os.getenv("OCR_MODEL", DEFAULT_MODEL))
    parser.add_argument("--endpoint", default=os.getenv("OCR_BASE_URL", DEFAULT_ENDPOINT),
                        help="Heiyu base URL; https://www.heiyucode.com/ becomes /v1/chat/completions")
    parser.add_argument("--api-key", default="",
                        help="Session key; otherwise HEIYU_API_KEY/OCR_API_KEY is read")
    parser.add_argument("--languages", default="ar,fr,en",
                        help="Comma-separated OCR language hints")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--input-price-per-1m", type=float, default=None,
                        help="USD per 1M input tokens; required for numeric cost")
    parser.add_argument("--output-price-per-1m", type=float, default=None,
                        help="USD per 1M output tokens; required for numeric cost")
    parser.add_argument("--estimated-input-tokens-per-page", type=float, default=0,
                        help="Fallback estimate when provider usage is absent")
    parser.add_argument("--estimated-output-tokens-per-page", type=float, default=0,
                        help="Fallback estimate when provider usage is absent")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume successful pages already present in sample.jsonl")
    return parser.parse_args()


def discover_targets(root: Path, wanted: int, route_config: PipelineConfig | None = None) -> tuple[list[PageTarget], dict[str, Any]]:
    if wanted < 1:
        raise ValueError("--pages must be greater than zero.")
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")
    route_config = route_config or PipelineConfig()
    pdfs = sorted((path for path in root.rglob("*.pdf") if path.is_file()),
                  key=lambda path: path.relative_to(root).as_posix().casefold())
    targets: list[PageTarget] = []
    total_pages = 0
    readable = 0
    unreadable: list[dict[str, str]] = []
    by_batch: dict[str, dict[str, int]] = {}
    route_counts = {route.value: 0 for route in PageRoute}
    for path in pdfs:
        relative = path.relative_to(root).as_posix()
        batch = relative.split("/", 1)[0]
        try:
            with PdfPageSource(path, route_config.render_dpi) as document:
                count = document.page_count
                page_data = list(document.pages())
        except Exception as exc:
            unreadable.append({"pdf": relative, "error": str(exc)[:240]})
            continue
        readable += 1
        total_pages += count
        stats = by_batch.setdefault(batch, {"pdfs": 0, "pages": 0})
        stats["pdfs"] += 1
        stats["pages"] += count
        for page_number, text, image_ratio in page_data:
            signals = measure_page(text, image_ratio)
            route, warnings = classify_page(text, signals, route_config)
            route_counts[route.value] += 1
            if route != PageRoute.DIGITAL and len(targets) < wanted:
                targets.append(PageTarget(path, relative, page_number, count, route.value, tuple(warnings)))
    if len(targets) < wanted:
        eligible = route_counts[PageRoute.SCANNED.value] + route_counts[PageRoute.SUSPECT.value]
        raise ValueError(f"Only {eligible} pages require OCR; {wanted} requested.")
    return targets, {"pdfs_found": len(pdfs), "pdfs_readable": readable,
                     "pdfs_unreadable": unreadable, "total_pdf_pages": total_pages,
                     "by_batch": by_batch, "route_counts": route_counts,
                     "selection": "first pages routed SCANNED or SUSPECT by project rules; DIGITAL pages skipped"}


def usage_from_response(data: dict[str, Any]) -> dict[str, float] | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    def number(*names: str) -> float:
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0
    prompt = number("prompt_tokens", "input_tokens")
    completion = number("completion_tokens", "output_tokens")
    total = number("total_tokens") or prompt + completion
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}


def cost_for(tokens: dict[str, float], input_price: float | None, output_price: float | None) -> float | None:
    if input_price is None or output_price is None:
        return None
    return (tokens["input_tokens"] * input_price + tokens["output_tokens"] * output_price) / 1_000_000


def redact_error(exc: Exception) -> str:
    value = str(exc).replace("\r", " ").replace("\n", " ")
    for secret in (os.getenv("HEIYU_API_KEY", ""), os.getenv("OCR_API_KEY", "")):
        if secret:
            value = value.replace(secret, "***")
    return value[:500]


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1; 0 means unlimited in the main pipeline and is unsafe here.")
    if args.pages != 1000:
        print(f"Note: requested sample size is {args.pages} pages, not 1000.", file=sys.stderr)
    route_config = PipelineConfig(render_dpi=args.dpi)
    targets, inventory = discover_targets(args.dataset_root, args.pages, route_config)
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "sample.jsonl"
    report_path = args.output / "report.json"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and result_path.is_file():
        with result_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        existing[row["page_id"]] = row
                except (json.JSONDecodeError, KeyError):
                    continue

    key = args.api_key.strip() or os.getenv("HEIYU_API_KEY", "").strip() or os.getenv("OCR_API_KEY", "").strip()
    config = PipelineConfig(ocr_provider="heiyu", ocr_api_key=key, session_api_key_override=bool(key),
                            ocr_base_url=args.endpoint, ocr_model=args.model,
                            ocr_timeout_s=args.timeout, ocr_max_retries=args.max_retries,
                            ocr_retry_base_s=args.retry_base_seconds, render_dpi=args.dpi,
                            languages=[item.strip() for item in args.languages.split(",") if item.strip()])
    adapter = HeiyuAdapter(config)
    limiter = RateLimiter(args.rpm)
    processed = 0
    skipped = 0
    failed = 0
    totals = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0}
    measured_pages = 0
    estimated_pages = 0
    started = time.time()
    def submit_page(target: PageTarget) -> dict[str, Any]:
        page_id = f"{target.relative_pdf}#page={target.page}"
        row: dict[str, Any] = {"page_id": page_id, "pdf": target.relative_pdf, "page": target.page,
                               "route": target.route, "route_warnings": list(target.route_warnings),
                               "status": "failed", "started_at": datetime.now(timezone.utc).isoformat()}
        started_page = time.time()
        try:
            with pymupdf.open(target.pdf) as document:
                image = document[target.page - 1].get_pixmap(
                    matrix=pymupdf.Matrix(args.dpi / 72, args.dpi / 72), alpha=False
                ).tobytes("png")
            limiter.wait()
            response = adapter._post_json(
                adapter._endpoint(),
                {"Authorization": f"Bearer {adapter._api_key()}", "Content-Type": "application/json"},
                adapter._build_payload(image, config.languages),
            )
            text = adapter._extract_text(response)
            usage = usage_from_response(response)
            if usage is None:
                usage = {"input_tokens": float(args.estimated_input_tokens_per_page),
                         "output_tokens": float(args.estimated_output_tokens_per_page),
                         "total_tokens": float(args.estimated_input_tokens_per_page + args.estimated_output_tokens_per_page)}
                usage_source = "estimate"
            else:
                usage_source = "provider"
            row.update({"status": "ok", "characters": len(text), "transcription": text,
                        **usage, "usage_source": usage_source,
                        "elapsed_seconds": round(time.time() - started_page, 3)})
        except Exception as exc:
            row.update({"error": redact_error(exc), "elapsed_seconds": round(time.time() - started_page, 3)})
        return row

    # Append-only output makes an interrupted run resumable without holding 1,000 images in memory.
    with result_path.open("a", encoding="utf-8", newline="\n") as handle:
        pending = [(index, target) for index, target in enumerate(targets, 1)
                   if f"{target.relative_pdf}#page={target.page}" not in existing]
        with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="ocr-test") as pool:
            futures = {pool.submit(submit_page, target): index for index, target in pending}
            for completed_index, future in enumerate(as_completed(futures), 1):
                index = futures[future]
                row = future.result()
                page_id = row["page_id"]
                if row["status"] == "ok":
                    processed += 1
                    for name in totals:
                        totals[name] += float(row.get(name, 0) or 0)
                    if row.get("usage_source") == "provider":
                        measured_pages += 1
                    else:
                        estimated_pages += 1
                else:
                    failed += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if completed_index % 10 == 0 or completed_index == len(futures):
                    print(f"{completed_index}/{len(futures)} new pages; ok={processed + skipped}, failed={failed}", flush=True)

        # Count resumed rows after the concurrent work so report totals include them.
        for target in targets:
            page_id = f"{target.relative_pdf}#page={target.page}"
            if page_id in existing:
                row = existing[page_id]
                for name in totals:
                    totals[name] += float(row.get(name, 0) or 0)
                measured_pages += int(row.get("usage_source") == "provider")
                estimated_pages += int(row.get("usage_source") == "estimate")
                skipped += 1

    successful = processed + skipped
    measured = {"input_tokens": totals["input_tokens"], "output_tokens": totals["output_tokens"],
                "total_tokens": totals["total_tokens"]}
    measured_cost = cost_for(measured, args.input_price_per_1m, args.output_price_per_1m)
    per_page = {name: value / successful for name, value in measured.items()} if successful else {name: 0 for name in measured}
    full_total = args.pages if args.pages else 1000
    projected_tokens = {name: per_page[name] * DEFAULT_TOTAL_PAGES for name in measured}
    projected_cost = cost_for(projected_tokens, args.input_price_per_1m, args.output_price_per_1m)
    report = {
        "sample": {"requested_pages": args.pages, "successful_pages": successful, "new_pages": processed,
                   "resumed_pages": skipped, "failed_pages": failed, "rpm_limit": args.rpm,
                   "concurrency": args.concurrency, "model": args.model,
                   "endpoint": adapter._endpoint(), "dpi": args.dpi},
        "inventory": inventory,
        "usage": {"sample": measured, "sample_per_successful_page": per_page,
                  "provider_usage_pages": measured_pages, "estimated_usage_pages": estimated_pages,
                  "source": "provider usage where returned; otherwise configured fallback estimate"},
        "pricing_usd_per_million_tokens": {"input": args.input_price_per_1m, "output": args.output_price_per_1m},
        "cost_usd": {"sample": measured_cost, "projected_for_112487_pages": projected_cost,
                     "pricing_required": measured_cost is None,
                     "note": "Costs are unavailable until Heiyu model prices are supplied; token usage is still recorded."},
        "projection_for_112487_pages": {"pages": DEFAULT_TOTAL_PAGES, "tokens": projected_tokens,
                                         "cost_usd": projected_cost},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_file": str(result_path.resolve()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path.resolve()), "successful_pages": successful,
                      "failed_pages": failed, "sample_cost_usd": measured_cost,
                      "projected_cost_usd": projected_cost}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
