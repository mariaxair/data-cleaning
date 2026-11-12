"""GUI stage callbacks must work offline and return displayable results."""

import json
import time
from pathlib import Path

import pytest

from app import (
    _concurrency_value,
    _load_benchmarks,
    _read_stage_inputs,
    _run_governance,
    _run_review,
    _run_stream,
    _run_synthesis,
)
from config import PipelineConfig
from corpus_pipeline.processor import process_batch


def _cleaned_source(path: Path) -> Path:
    path.write_text(
        "Article 1 This article describes the first rule and its conditions.\n"
        "Article 2 This article describes the second rule and its conditions.\n",
        encoding="utf-8",
    )
    return path


def _governed_record(evidence: dict | list) -> dict:
    return {
        "id": "r1",
        "question": "What is the first rule?",
        "answer": "The first rule applies.",
        "reasoning_chain": "The source states the first rule.",
        "language": "en",
        "source_text": "This article describes the first rule and its conditions.",
        "provenance": {"source_document": "law.txt", "pages": [1]},
        "evidence": evidence,
    }


def test_synthesis_tab_dry_run_selects_units(tmp_path: Path) -> None:
    source = _cleaned_source(tmp_path / "law.txt")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps({str(source.resolve()): {"tier": "statutory_law"}}), encoding="utf-8")
    stage_output = tmp_path / "training"

    summary, status, artifacts = _run_synthesis(
        [str(source)], "", str(overrides), "", 10, True, str(stage_output),
        "heiyu", "", "https://example.invalid/v1", "text-model")

    assert summary["selected_units"] == 1
    assert summary["eligible_units"] == 1
    assert "Dry run" in status and "no model calls" in status
    assert str(stage_output / "synthesis" / "plan.json") in artifacts


def test_governance_tab_writes_buckets(tmp_path: Path) -> None:
    sample = tmp_path / "samples.jsonl"
    sample.write_text(json.dumps(_governed_record({})) + "\n", encoding="utf-8")
    stage_output = tmp_path / "training"

    summary, status, artifacts = _run_governance(
        [str(sample)], "", str(stage_output))

    assert summary["manual"] == 1
    assert summary["pass"] == 0
    assert "1 records" in status and "1 manual" in status
    assert str(stage_output / "governance" / "manual.jsonl") in artifacts
    assert (stage_output / "governance" / "summary.json").is_file()


def test_review_tab_runs_offline_on_governed_records(tmp_path: Path) -> None:
    sample = tmp_path / "manual.jsonl"
    evidence = {"question": [{"claim": "first rule", "quote": "the first rule"}], "answer": []}
    sample.write_text(json.dumps(_governed_record(evidence)) + "\n", encoding="utf-8")
    stage_output = tmp_path / "training"

    summary, status, artifacts = _run_review(
        [str(sample)], "", False, str(stage_output),
        "heiyu", "", "https://example.invalid/v1", "text-model")

    assert summary["manual"] == 1
    assert summary["repair_attempts"] == 0
    assert "offline deterministic review" in status
    assert str(stage_output / "review" / "audit.jsonl") in artifacts
    assert (stage_output / "review" / "summary.json").is_file()


def test_review_tab_reports_malformed_evidence_instead_of_crashing(tmp_path: Path) -> None:
    sample = tmp_path / "broken.jsonl"
    sample.write_text(
        json.dumps(_governed_record([{"claim": "first rule"}])) + "\n", encoding="utf-8")

    summary, status, artifacts = _run_review(
        [str(sample)], "", False, str(tmp_path / "training"),
        "heiyu", "", "https://example.invalid/v1", "text-model")

    assert summary is None and artifacts == []
    assert status.startswith("Review failed:")


def test_uploads_take_precedence_over_prefilled_paths(tmp_path: Path) -> None:
    uploaded = tmp_path / "uploaded.jsonl"
    uploaded.write_text("{}\n", encoding="utf-8")
    missing = tmp_path / "missing.jsonl"

    paths = _read_stage_inputs([str(uploaded)], str(missing))

    assert paths == [uploaded.resolve()]


def test_extraction_stream_reports_live_activity(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("hello", encoding="utf-8")

    def fake_process_batch(paths, config, progress=None):
        progress("Reading fake.txt")
        time.sleep(0.3)
        progress("Finished fake.txt")
        return {
            "results": [{
                "source": str(source), "status": "complete", "detected_type": "digital",
                "route_summary": "1 digital / 0 OCR / 0 suspect", "warnings": [],
                "output_path": str(tmp_path / "out.txt"), "original_text": "hello",
                "cleaned_text": "hello", "pages": [],
            }],
            "files_processed": 1, "files_skipped": 0, "files_failed": 0,
            "pages_routed_to_ocr": 0, "total_characters": 5, "provider": "heiyu",
        }

    monkeypatch.setattr("app.process_batch", fake_process_batch)
    chunks = list(_run_stream(
        [str(source)], "", "heiyu", "", "https://example.invalid/v1", "model", 1, 1, 1))

    assert len(chunks) >= 2
    assert any("Reading fake.txt" in chunk[6] for chunk in chunks)
    assert chunks[-1][4].startswith("1 processed")
    assert "Finished fake.txt" in chunks[-1][6]


def test_concurrency_zero_means_unlimited() -> None:
    assert _concurrency_value(0, 4, "Pages") == 0
    assert _concurrency_value(None, 4, "Pages") == 4
    assert _concurrency_value("", 4, "Pages") == 4
    with pytest.raises(ValueError, match="negative"):
        _concurrency_value(-1, 4, "Pages")


def test_benchmark_tab_reads_history_from_extraction(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("hello world", encoding="utf-8")
    benchmark_dir = tmp_path / "benchmarks"
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / "sample.txt").write_text("hello world", encoding="utf-8")
    config = PipelineConfig(
        output_dir=tmp_path / "out",
        cache_db=tmp_path / "cache.sqlite3",
        benchmark_dir=benchmark_dir,
        benchmark_reference_dir=reference_dir,
        ocr_provider="openai",
        ocr_model="model-a",
    )
    process_batch([source], config)

    rows, status, leader, comparison, artifacts = _load_benchmarks(
        str(benchmark_dir), str(reference_dir))

    assert len(rows) == 1
    assert rows[0][0] == "openai" and rows[0][1] == "model-a"
    assert rows[0][10] == 0.0  # mean CER against an identical reference
    assert "1 saved document rows" in status
    assert "model-a" in leader and "0.0000" in leader
    assert comparison["groups"][0]["documents"] == 1
    assert str(benchmark_dir / "history.jsonl") in artifacts


def test_benchmark_tab_handles_empty_history(tmp_path: Path) -> None:
    rows, status, leader, comparison, artifacts = _load_benchmarks(
        str(tmp_path / "benchmarks"), str(tmp_path / "reference"))

    assert rows == [] and artifacts == []
    assert comparison is not None
    assert "No benchmark rows yet" in status
    assert "No reference-based CER yet" in leader
