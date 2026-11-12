import hashlib
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pymupdf as fitz
from PIL import Image

from config import PipelineConfig
from corpus_pipeline.cache import PipelineCache
from corpus_pipeline.extraction import PdfPageSource, _fix_ligature_pairs
from corpus_pipeline.models import OCRResult
from corpus_pipeline.processor import process_batch, process_file


def _write_scanned_pdf(path, colours: list[str]) -> None:
    document = fitz.open()
    for colour in colours:
        page = document.new_page()
        buffer = io.BytesIO()
        Image.new("RGB", (240, 240), colour).save(buffer, format="PNG")
        page.insert_image(fitz.Rect(0, 0, 595, 842), stream=buffer.getvalue())
    document.save(path)
    document.close()


def _page_labels(pdf_path, dpi: int) -> dict[str, str]:
    """Map the exact rendered bytes of each page to a stable label."""
    with PdfPageSource(pdf_path, dpi) as source:
        return {
            hashlib.sha256(source.render(number)).hexdigest(): f"page-{number}"
            for number in range(1, source.page_count + 1)
        }


def test_page_order_is_preserved_when_ocr_finishes_out_of_order(tmp_path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    _write_scanned_pdf(pdf_path, ["red", "green", "blue"])
    dpi = 72
    labels = _page_labels(pdf_path, dpi)
    delays = {"page-1": 0.15, "page-2": 0.08, "page-3": 0.0}

    class SlowOCR:
        def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
            label = labels[hashlib.sha256(image).hexdigest()]
            time.sleep(delays[label])
            return OCRResult(label, 1.0, "mock")

    config = PipelineConfig(
        output_dir=tmp_path / "out",
        cache_db=tmp_path / "cache.sqlite3",
        render_dpi=dpi,
        page_concurrency=3,
        ocr_concurrency=3,
    )
    result = process_file(pdf_path, config, ocr_adapter=SlowOCR())

    assert [page["page_number"] for page in result["pages"]] == [1, 2, 3]
    assert [page["route"] for page in result["pages"]] == ["SCANNED"] * 3
    positions = [result["cleaned_text"].index(f"page-{number}") for number in (1, 2, 3)]
    assert positions == sorted(positions)


def test_batch_processes_files_concurrently_and_keeps_input_order(tmp_path) -> None:
    paths = [tmp_path / "first.pdf", tmp_path / "second.pdf"]
    _write_scanned_pdf(paths[0], ["red"])
    _write_scanned_pdf(paths[1], ["blue"])

    class TrackingOCR:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.2)
            with self.lock:
                self.active -= 1
            return OCRResult("parallel text", 1.0, "mock")

    tracker = TrackingOCR()
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        cache_db=tmp_path / "cache.sqlite3",
        render_dpi=72,
        file_concurrency=2,
        page_concurrency=2,
        ocr_concurrency=2,
    )
    summary = process_batch(paths, config, ocr_adapter=tracker)

    assert summary["files_processed"] == 2
    assert [result["source"] for result in summary["results"]] == [str(path) for path in paths]
    assert tracker.max_active >= 2


def test_zero_concurrency_starts_every_file_and_page_at_once(tmp_path) -> None:
    paths = [tmp_path / "first.pdf", tmp_path / "second.pdf"]
    _write_scanned_pdf(paths[0], ["red", "green", "blue", "yellow"])
    _write_scanned_pdf(paths[1], ["cyan", "magenta", "orange", "purple"])

    class TrackingOCR:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.5)
            with self.lock:
                self.active -= 1
            return OCRResult("parallel text", 1.0, "mock")

    tracker = TrackingOCR()
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        cache_db=tmp_path / "cache.sqlite3",
        render_dpi=72,
        file_concurrency=0,
        page_concurrency=0,
        ocr_concurrency=0,
    )
    summary = process_batch(paths, config, ocr_adapter=tracker)

    assert summary["files_processed"] == 2
    assert tracker.max_active == 8


def test_ligature_pairs_swap_only_at_synthetic_origins() -> None:
    ligature_origin = (10.0, 5.0)
    chars = [
        {"c": "\u0627", "origin": ligature_origin},   # ا synthesised after the glyph
        {"c": "\u0644", "origin": ligature_origin},   # ل shared ligature origin
    ]
    _fix_ligature_pairs(chars, {ligature_origin})
    assert [char["c"] for char in chars] == ["\u0644", "\u0627"]

    article = [
        {"c": "\u0627", "origin": (1.0, 5.0)},
        {"c": "\u0644", "origin": (2.0, 5.0)},
    ]
    _fix_ligature_pairs(article, {ligature_origin})
    assert [char["c"] for char in article] == ["\u0627", "\u0644"]


def test_cache_accepts_concurrent_writes(tmp_path) -> None:
    cache_path = tmp_path / "cache.sqlite3"

    def write(index: int) -> None:
        PipelineCache(cache_path).put_ocr(f"key-{index}", OCRResult(f"text-{index}", 1.0, "mock"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(40)))

    cache = PipelineCache(cache_path)
    assert all(cache.get_ocr(f"key-{index}").text == f"text-{index}" for index in range(40))
