import io
import json
import time
from pathlib import Path

import pymupdf as fitz
from PIL import Image

from config import PipelineConfig
from corpus_pipeline.benchmark import (
    build_document_metrics,
    compare_records,
    compare_texts,
    find_reference,
    normalize_text,
    split_pages,
    text_quality,
)
from corpus_pipeline.models import OCRResult, PageReport, PageSignals, PageRoute
from corpus_pipeline.processor import process_batch, process_file


def _write_scanned_pdf(path: Path, colours: list[str]) -> None:
    document = fitz.open()
    for colour in colours:
        page = document.new_page()
        buffer = io.BytesIO()
        Image.new("RGB", (240, 240), colour).save(buffer, format="PNG")
        page.insert_image(fitz.Rect(0, 0, 595, 842), stream=buffer.getvalue())
    document.save(path)
    document.close()


def _config(tmp_path: Path, **overrides) -> PipelineConfig:
    values = {
        "output_dir": tmp_path / "out",
        "cache_db": tmp_path / "cache.sqlite3",
        "benchmark_dir": tmp_path / "benchmarks",
        "benchmark_reference_dir": tmp_path / "reference",
    }
    values.update(overrides)
    return PipelineConfig(**values)


def _reports(*page_texts: str) -> list[PageReport]:
    return [
        PageReport(number, PageRoute.DIGITAL, PageSignals(len(text), 1.0, 0.0))
        for number, text in enumerate(page_texts, start=1)
    ]


def test_normalisation_removes_marks_and_markers() -> None:
    text = "--- Page 7 ---\nمُحَمَّدٌ عَـلِي\n"
    assert normalize_text(text) == "محمد علي"
    assert normalize_text("مُحَمَّد", strip_diacritics=False).startswith("مُحَمَّد")


def test_normalisation_optionally_unifies_alef_and_ya() -> None:
    assert normalize_text("أحمد") == "أحمد"
    assert normalize_text("أحمد", unify_alef=True) == "احمد"
    assert normalize_text("على", unify_ya=True) == "علي"


def test_character_and_word_metrics_on_known_edits() -> None:
    exact = compare_texts("abcdef", "abcdef", 100)
    assert exact["character"]["error_rate"] == 0.0
    assert exact["character"]["accuracy"] == 1.0
    assert exact["character"]["precision"] == 1.0
    assert exact["character"]["f1"] == 1.0
    assert not exact["truncated"]

    substituted = compare_texts("abcdef", "abcxef", 100)
    assert substituted["character"]["substitutions"] == 1
    assert substituted["character"]["error_rate"] == round(1 / 6, 6)
    assert substituted["character"]["precision"] == round(5 / 6, 6)

    words = compare_texts("alpha beta gamma", "alpha beta delta", 100)
    assert words["word"]["substitutions"] == 1
    assert words["word"]["error_rate"] == round(1 / 3, 6)
    assert words["word"]["recall"] == round(2 / 3, 6)


def test_long_comparison_is_capped_and_flagged() -> None:
    result = compare_texts("a" * 500, "a" * 400 + "b" * 100, 50)
    assert result["truncated"] is True
    assert result["evaluated_reference_chars"] == 50
    assert result["evaluated_hypothesis_chars"] == 50


def test_quality_proxies_flag_replacement_characters_and_repeats() -> None:
    quality = text_quality("abc\ufffd\nsame\nsame\nsame\n", ["abc", ""])
    assert quality["replacement_character_ratio"] > 0
    assert quality["valid_character_ratio"] < 1
    assert quality["max_repeated_line_run"] == 3
    assert quality["empty_page_ratio"] == 0.5


def test_split_pages_and_reference_resolution(tmp_path: Path) -> None:
    pages = split_pages("--- Page 1 ---\nfirst\n--- Page 2 ---\nsecond\n")
    assert pages == {1: "first\n", 2: "second\n"}

    source = tmp_path / "law.pdf"
    source.write_bytes(b"placeholder")
    config = _config(tmp_path)
    config.benchmark_reference_dir.mkdir(parents=True)
    reference = config.benchmark_reference_dir / "law.txt"
    reference.write_text("reference", encoding="utf-8")
    assert find_reference(source, config) == reference

    reference.unlink()
    beside = tmp_path / "law.ref.txt"
    beside.write_text("reference", encoding="utf-8")
    assert find_reference(source, config) == beside


def test_page_aligned_reference_metrics_and_raw_comparison(tmp_path: Path) -> None:
    reference_text = "--- Page 1 ---\nabc\n--- Page 2 ---\ndef\n"
    raw_text = "--- Page 1 ---\nabd\n--- Page 2 ---\ndef\n"
    reference = tmp_path / "doc.txt"
    reference.write_text(reference_text, encoding="utf-8")

    metrics = build_document_metrics(
        tmp_path / "doc.pdf",
        _config(tmp_path),
        page_reports=_reports("abd", "def"),
        cleaned_pages=["abd", "def"],
        raw_text=raw_text,
        cleaned_text=raw_text,
        source_hash="deadbeef",
        status="complete",
        wall_s=1.5,
        reference_path=reference,
    )

    assert metrics["reference"]["matched"] is True
    assert metrics["reference"]["page_aligned"] is True
    assert [page["page"] for page in metrics["reference"]["per_page"]] == [1, 2]
    assert metrics["reference"]["per_page"][0]["character"]["error_rate"] == round(1 / 3, 6)
    assert metrics["reference"]["per_page"][1]["character"]["error_rate"] == 0.0


def test_process_file_writes_metrics_and_measures_cleaning_gain(tmp_path: Path) -> None:
    source = tmp_path / "memo.txt"
    source.write_text("cafÃ©  service", encoding="utf-8")
    config = _config(tmp_path)
    config.benchmark_reference_dir.mkdir(parents=True)
    (config.benchmark_reference_dir / "memo.txt").write_text("café service", encoding="utf-8")

    result = process_file(source, config)

    metrics_path = Path(result["metrics_path"])
    assert metrics_path.is_file()
    assert result["benchmark_row"]["reference_matched"] is True
    assert result["benchmark_row"]["cer"] == 0.0
    assert result["benchmark_row"]["raw_cer"] > 0.0
    assert result["benchmark_row"]["cer_improvement"] > 0.0


def test_batch_appends_history_and_groups_comparison(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("hello world", encoding="utf-8")
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / "sample.txt").write_text("hello world", encoding="utf-8")

    for model in ("model-a", "model-b"):
        config = _config(
            tmp_path / model,
            ocr_provider="openai",
            ocr_model=model,
            benchmark_dir=tmp_path / "benchmarks",
            benchmark_reference_dir=reference_dir,
        )
        summary = process_batch([source], config)
        assert summary["benchmark"]["documents"] == 1

    history = (tmp_path / "benchmarks" / "history.jsonl").read_text(encoding="utf-8")
    assert len(history.splitlines()) == 2
    comparison = compare_records(
        [json.loads(line) for line in history.splitlines()],
        history_path=str(tmp_path / "benchmarks" / "history.jsonl"),
    )
    assert [group["model"] for group in comparison["groups"]] == ["model-a", "model-b"]
    assert all(group["mean_cer"] == 0.0 for group in comparison["groups"])
    assert (tmp_path / "benchmarks" / "comparison.json").is_file()
    assert (tmp_path / "benchmarks" / "comparison.csv").is_file()


def test_benchmark_can_be_disabled(tmp_path: Path) -> None:
    source = tmp_path / "plain.txt"
    source.write_text("text", encoding="utf-8")
    config = _config(tmp_path, benchmark_enabled=False)
    result = process_file(source, config)
    assert result["metrics_path"] is None
    assert result["benchmark_row"] is None


def test_scanned_pdf_records_latency_confidence_and_cache_hits(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    _write_scanned_pdf(pdf, ["red", "blue"])

    class FakeOCR:
        def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
            time.sleep(0.01)
            return OCRResult("page text", 0.9, "mock")

    config = _config(
        tmp_path, render_dpi=72, ocr_provider="openai", ocr_model="mock-model",
        page_concurrency=2, ocr_concurrency=2,
    )
    first = process_file(pdf, config, ocr_adapter=FakeOCR())
    metrics = json.loads(Path(first["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["ocr"]["pages_routed"] == 2
    assert metrics["ocr"]["pages_succeeded"] == 2
    assert metrics["ocr"]["latency"]["count"] == 2
    assert metrics["ocr"]["confidence"]["mean"] == 0.9
    assert metrics["ocr"]["pages_from_cache"] == 0

    second = process_file(pdf, config, ocr_adapter=FakeOCR(), force=True)
    cached = json.loads(Path(second["metrics_path"]).read_text(encoding="utf-8"))
    assert cached["ocr"]["pages_from_cache"] == 2
    assert cached["ocr"]["pages_attempted"] == 0
    assert cached["ocr"]["latency"]["count"] == 0
    assert cached["ocr"]["cache_hit_rate"] == 1.0
