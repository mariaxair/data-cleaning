"""GUI stage callbacks must work offline and return displayable results."""

import json
import time
from pathlib import Path

import pytest

from app import (
    STAGE_SETTINGS,
    _apply_stage_settings,
    _concurrency_value,
    _load_benchmarks,
    _read_stage_inputs,
    _run_governance,
    _run_review,
    _run_stream,
    _run_synthesis,
    _run,
    _run_metadata,
    _run_metadata_evaluation,
    _setting_default,
    _runtime_config,
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
        [str(source)], "", str(overrides), 10, True, str(stage_output),
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


def test_gui_advanced_settings_override_only_the_selected_run() -> None:
    config = PipelineConfig()
    values = [getattr(config, name) for name, _, _ in STAGE_SETTINGS["metadata"]]
    values[0] = 0.65  # semantic threshold
    values[1] = 1024  # max tokens per chunk
    values[4] = ".\n؟\n؛"
    values[5] = "next"

    _apply_stage_settings(config, "metadata", tuple(values))

    assert config.metadata_semantic_threshold == 0.65
    assert config.metadata_chunk_tokens == 1024
    assert config.metadata_delimiters == [".", "؟", "؛"]
    assert config.metadata_include_delim == "next"


def test_metadata_gui_passes_chonkie_settings_to_the_run(tmp_path: Path, monkeypatch) -> None:
    import corpus_pipeline.metadata.runner as runner
    from app import _run_metadata

    source = _cleaned_source(tmp_path / "source.txt")
    captured = {}

    def fake_discover(paths, config, *, dry_run=False):
        captured.update(enabled=config.metadata_semantic_enabled,
                        embedding=config.metadata_embedding_model,
                        dry_run=dry_run)
        return {"processed": 1, "model_calls": 0, "review_cases": 0,
                "chunkers": {"chonkie_semantic_1.7.0": 1}}

    monkeypatch.setattr(runner, "discover_metadata", fake_discover)
    monkeypatch.setattr("corpus_pipeline.settings.load_json", lambda _: {"fields": []})
    result = _run_metadata(
        [str(source)], "", False, False, 5, str(tmp_path / "training"),
        "heiyu", "", "", "", True, "minishlab/potion-multilingual-128M")

    assert captured == {"enabled": True, "embedding": "minishlab/potion-multilingual-128M", "dry_run": False}
    assert "Chonkie: 1 documents" in result[1]


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


def _settings(stage: str, **changes) -> list:
    config = PipelineConfig()
    return [changes.get(name, _setting_default(config, name, kind))
            for name, kind, _ in STAGE_SETTINGS[stage]]


def test_every_runtime_setting_is_available_or_has_a_nonredundant_equivalent():
    from dataclasses import fields
    settings = {name for schema in STAGE_SETTINGS.values() for name, _, _ in schema}
    direct_controls = {
        "input_dir", "ocr_provider", "ocr_api_key", "ocr_base_url", "ocr_model",
        "file_concurrency", "page_concurrency", "ocr_concurrency", "synthesis_model",
        "manual_overrides_path", "synthesis_max_samples", "synthesis_screening_mode",
        "topic_max_calls", "stage_output_dir", "metadata_model", "metadata_llm_enabled",
        "metadata_max_calls", "metadata_semantic_enabled", "metadata_embedding_model", "review_llm_enabled",
    }
    # Stage buttons replace CLI auto-selection flags; skip-window is fixed by the source contract.
    equivalents = {"synthesis_enabled", "metadata_enabled", "review_enabled", "corpus_processing_enabled",
                   "session_api_key_override", "metadata_skip_window"}
    assert {f.name for f in fields(PipelineConfig)} == settings | direct_controls | equivalents
    for schema in STAGE_SETTINGS.values():
        assert len({name for name, _, _ in schema}) == len(schema)


@pytest.mark.parametrize("stage,shared", [
    ("extract", ["benchmarks"]), ("synthesis", ["generation"]),
    ("metadata", ["generation"]), ("review", ["governance", "generation"]),
])
def test_stage_and_shared_controls_are_applied_together(stage, shared, monkeypatch):
    import os
    before = dict(os.environ)
    config = PipelineConfig()
    values = _settings(stage)
    for group in shared:
        changes = {"ocr_timeout_s": 123, "ocr_max_retries": 0, "ocr_retry_base_s": 0.25} if group == "generation" else {}
        values += _settings(group, **changes)
    _apply_stage_settings(config, stage, tuple(values))
    if "generation" in shared:
        assert config.ocr_timeout_s == 123 and config.ocr_max_retries == 0 and config.ocr_retry_base_s == 0.25
    assert dict(os.environ) == before


@pytest.mark.parametrize("name,value", [
    ("render_dpi", 1.5), ("ocr_max_retries", -1), ("ocr_retry_base_s", float("nan")),
    ("min_valid_char_ratio", 1.1), ("languages", []), ("page_marker_template", "{secret}"),
])
def test_invalid_gui_settings_fail_before_mutating_config(name, value):
    from dataclasses import asdict
    config = PipelineConfig()
    original = asdict(config)
    with pytest.raises((ValueError, TypeError)):
        _apply_stage_settings(config, "extract", tuple(_settings("extract", **{name: value})))
    assert asdict(config) == original


def test_review_and_governance_use_the_same_dialect_controls(tmp_path, monkeypatch):
    import corpus_pipeline.review.runner as review_runner
    import corpus_pipeline.governance.runner as governance_runner
    sample = tmp_path / "records.jsonl"
    sample.write_text("{}\n", encoding="utf-8")
    captured = []

    def fake(records, config):
        captured.append(config)
        return {}

    monkeypatch.setattr(review_runner, "review_corpus", fake)
    monkeypatch.setattr(governance_runner, "process_corpus", fake)
    shared = _settings("governance", dialect_backend="marbert", dialect_model_path="local/model",
                       dialect_validated=True, language_mixing_threshold=0.15)
    _run_governance([str(sample)], "", str(tmp_path), *shared)
    _run_review([str(sample)], "", False, str(tmp_path), "heiyu", "", "", "", *(
        _settings("review", synthesis_max_tokens=512) + shared + _settings("generation", ocr_timeout_s=30)))
    assert len(captured) == 2
    for config in captured:
        assert config.dialect_backend == "marbert" and config.dialect_model_path == Path("local/model")
        assert config.dialect_validated and config.language_mixing_threshold == 0.15
    assert captured[-1].synthesis_max_tokens == 512 and captured[-1].ocr_timeout_s == 30


def test_metadata_ui_custom_embedding_and_transport_apply_only_to_this_run(tmp_path, monkeypatch):
    import corpus_pipeline.metadata.runner as runner
    import os
    source = _cleaned_source(tmp_path / "source.txt")
    captured = []
    before = dict(os.environ)

    def fake(paths, config, *, dry_run):
        captured.append(config)
        return {"processed": 1, "model_calls": 0, "review_cases": 0}

    monkeypatch.setattr(runner, "discover_metadata", fake)
    advanced = _settings("metadata") + _settings("generation", ocr_timeout_s=9, ocr_max_retries=0)
    for enabled, model in ((True, "C:/local multilingual model"), (False, "")):
        result = _run_metadata([str(source)], "", True, False, 7, str(tmp_path / "out"), "heiyu", "", "", "",
                               enabled, model, *advanced)
        assert result[0] is not None
    assert captured[0].metadata_semantic_enabled and captured[0].metadata_embedding_model == "C:/local multilingual model"
    assert not captured[1].metadata_semantic_enabled and captured[1].metadata_embedding_model == ""
    assert captured[0].ocr_timeout_s == 9 and captured[0].ocr_max_retries == 0
    assert dict(os.environ) == before


def test_synthesis_ui_passes_limits_and_transport_to_run(tmp_path, monkeypatch):
    import corpus_pipeline.synthesis.runner as runner
    source = _cleaned_source(tmp_path / "source.txt")
    captured = {}

    def fake(paths, config, **kwargs):
        captured.update(config=config, **kwargs)
        return {"selected_units": 1, "eligible_units": 1}

    monkeypatch.setattr(runner, "synthesize", fake)
    result = _run_synthesis([str(source)], "", "", 3, True, str(tmp_path / "out"), "heiyu", "", "", "fake-text",
                            "automatic", True, 17, *(_settings("synthesis", synthesis_metadata_enabled=True,
                                                               article_group_size=4) + _settings("generation", ocr_timeout_s=18)))
    assert result[0] is not None
    assert captured["config"].synthesis_metadata_enabled
    assert captured["config"].article_group_size == 4 and captured["config"].ocr_timeout_s == 18
    assert captured["config"].topic_max_calls == 17 and captured["max_samples"] == 3 and captured["topics_only"]


def test_extraction_ui_force_preflight_and_cleaning_report(tmp_path, monkeypatch):
    calls = []
    source = tmp_path / "input.txt"
    source.write_text("data", encoding="utf-8")

    class FakeOCR:
        def test_connection(self):
            calls.append("preflight")

    def fake(paths, config, **kwargs):
        calls.append((config, kwargs))
        return {"results": [{"source": str(source), "pages": [{"page_number": 1, "route": "DIGITAL",
                 "signals": {"extracted_characters": 4, "valid_character_ratio": 1, "full_page_image_ratio": 0},
                 "warnings": [], "ocr_failed": False, "cleaning_hits": {"html": 2}}]}],
                "files_processed": 1, "files_skipped": 0, "files_failed": 0,
                "pages_routed_to_ocr": 0, "total_characters": 4, "provider": "heiyu"}

    monkeypatch.setattr("app.create_adapter", lambda _: FakeOCR())
    monkeypatch.setattr("app.process_batch", fake)
    settings = _settings("extract", output_dir=str(tmp_path / "out"), semantic_rules_enabled=True,
                         page_marker_template="Page {page}\n") + _settings("benchmarks", benchmark_dir=str(tmp_path / "metrics"))
    result = _run([str(source)], "", "heiyu", "", "", "model", 2, 3, 1, None, True, True, True, *settings)
    assert calls[0] == "preflight"
    config, kwargs = calls[1]
    assert kwargs["force"] and config.semantic_rules_enabled and config.page_marker_template == "Page {page}\n"
    assert config.benchmark_dir == tmp_path / "metrics" and config.file_concurrency == 2
    report = tmp_path / "out/rules-report.json"
    assert json.loads(report.read_text(encoding="utf-8")) == {"html": 2}
    assert str(report) in result[-1]


def test_metadata_evaluation_handles_missing_precision_without_crash(tmp_path, monkeypatch):
    monkeypatch.setattr("corpus_pipeline.benchmark.evaluate_metadata", lambda *a, **kw: {"precision": None, "recall": None})
    result, status = _run_metadata_evaluation("references.jsonl", str(tmp_path), 0.98, 0.9, 20)
    assert result is not None and "unavailable" in status


def test_cleared_generation_model_overrides_env_default(monkeypatch):
    monkeypatch.setenv("SYNTHESIS_MODEL", "ENV_MODEL")
    config = _runtime_config("heiyu", "", "", "", synthesis_model="")
    assert config.synthesis_model == ""


def test_cleared_optional_override_path_really_disables_env_override(monkeypatch):
    monkeypatch.setenv("MANUAL_OVERRIDES_PATH", "missing-overrides.json")
    config = _runtime_config("heiyu", "", "", "", manual_overrides="")
    assert config.manual_overrides_path is None


def test_gradio_events_share_controls_and_do_not_duplicate_session_fields():
    from app import demo
    import gradio as gr
    callbacks = {getattr(fn.fn, "__name__", ""): fn for fn in demo.fns.values()}
    stages = [callbacks[name] for name in (
        "_run_synthesis_agents", "_run_synthesis_metadata", "_run_governance", "_run_review", "_run_corpus")]
    roots = [[control for control in fn.inputs if control.label and control.label.startswith("Output root for")]
             for fn in stages]
    assert all(len(items) == 1 for items in roots)
    assert len({items[0]._id for items in roots}) == 1
    for label, names in (
        ("Request timeout (seconds)", ("_run_synthesis_agents", "_run_synthesis_metadata", "_run_review")),
        ("Dialect detection method", ("_run_governance", "_run_review")),
        ("Benchmark history folder", ("_run_ocr_stream", "_load_benchmarks")),
    ):
        controls = [[c for c in callbacks[name].inputs if c.label == label] for name in names]
        assert all(len(items) == 1 for items in controls)
        assert len({items[0]._id for items in controls}) == 1
    metadata = callbacks["_run_synthesis_metadata"].inputs
    embedding = next(c for c in metadata if c.label == "Embedding model")
    assert embedding.allow_custom_value
    assert isinstance(next(c for c in metadata if c.label == "Use Chonkie semantic chunking"), gr.Checkbox)
    assert not any("Evaluation minimum" in (c.label or "") for c in metadata)


def test_metadata_gui_cleared_embedding_does_not_reuse_env(tmp_path, monkeypatch):
    import corpus_pipeline.metadata.runner as runner
    source = _cleaned_source(tmp_path / "source.txt")
    monkeypatch.setenv("METADATA_EMBEDDING_MODEL", "ENV_MODEL")
    captured = []

    def fake(paths, config, *, dry_run):
        captured.append(config.metadata_embedding_model)
        return {"processed": 1, "model_calls": 0, "review_cases": 0}

    monkeypatch.setattr(runner, "discover_metadata", fake)
    result = _run_metadata([str(source)], "", True, False, 1, str(tmp_path), "heiyu", "", "", "", True, None)
    assert result[0] is not None and captured == [""]


def test_gui_has_exactly_six_ordered_pipeline_tabs():
    from app import demo
    tabs = [item["props"]["label"] for item in demo.config["components"] if item["type"] == "tabitem"]
    assert tabs == ["1 - OCR", "2 - Cleaning", "3 - Synthesis", "4 - Governance", "5 - Review", "6 - Corpus"]


def test_ocr_and_cleaning_gui_are_separate_and_resume(tmp_path):
    from app import _run_ocr_stream, _run_cleaning
    source = tmp_path / "raw.txt"
    source.write_text("word word<br>next", encoding="utf-8")
    root = tmp_path / "data"
    ocr_settings = _settings("ocr", cache_db=str(root / "pipeline.sqlite3"), benchmark_enabled=False)
    final = list(_run_ocr_stream([str(source)], "", "openai", "", "", "fake", 1, 1, 1,
                                 False, False, str(root), *(ocr_settings + _settings("benchmarks"))))[-1]
    raw = next((root / "ocr").glob("*.txt"))
    assert "word word<br>next" in raw.read_text(encoding="utf-8")
    assert len(final) == 6 and "1 processed" in final[3]
    result = _run_cleaning(None, str(root / "ocr"), str(root), True, False, *_settings("cleaning"))
    cleaned = next((root / "cleaning").glob("*.txt"))
    body = cleaned.read_text(encoding="utf-8")
    assert body.endswith("word\nnext\n")
    assert body.count("--- Page 1 ---") == 1
    assert result[0]["hit_counts"]["adjacent_word_duplicates"] == 1
    assert not (root / "synthesis").exists()
    resumed = _run_cleaning(None, str(root / "ocr"), str(root), True, False, *_settings("cleaning"))
    assert resumed[0]["files_skipped"] == 1


def test_metadata_gui_exports_exact_chunks_under_synthesis(tmp_path):
    from app import _run_synthesis_metadata
    from corpus_pipeline.artifacts import read_jsonl
    source = _cleaned_source(tmp_path / "law.txt")
    root = tmp_path / "data"
    result = _run_synthesis_metadata([str(source)], "", False, False, 1, str(root),
                                      "openai", "", "", "", False, "")
    assert result[0] is not None, result[1]
    assert (root / "synthesis/metadata/documents.jsonl").is_file()
    chunks = read_jsonl(root / "synthesis/chunking/chunks.jsonl")
    assert "".join(row["text"] for row in chunks) == source.read_bytes().decode("utf-8")
    assert all(row["chunking"]["degraded"] for row in chunks)
    assert not (root / "metadata").exists()


def test_corpus_gui_excludes_manual_and_rejected_records(tmp_path):
    from app import _run_corpus
    from corpus_pipeline.artifacts import write_jsonl, read_jsonl
    source = tmp_path / "review.jsonl"
    records = [{"question": "Question?", "answer": "Answer.", "status": status,
                "provenance": {"source_file": "source.txt"}, "pending_checks": []}
               for status in ("accept", "manual", "reject")]
    write_jsonl(source, records + [records[0]])
    result = _run_corpus([str(source)], "", str(tmp_path / "data"))
    assert result[0]["exported"] == 1 and result[0]["excluded"] == 3
    corpus = read_jsonl(tmp_path / "data/corpus/corpus.jsonl")
    assert corpus == [{"instruction": "Question?", "input": "", "output": "Answer."}]


def test_saved_chunk_agents_dry_run_needs_no_key(tmp_path, monkeypatch):
    from app import _run_synthesis_agents
    from corpus_pipeline.artifacts import write_jsonl
    source = tmp_path / "chunks.jsonl"
    write_jsonl(source, [{"text": "substantive text " * 5, "source": "example.txt"}])
    result = _run_synthesis_agents([str(source)], "", 2, True, str(tmp_path / "data"),
                                    "openai", "", "", "", False)
    assert result[0]["selected_units"] == 1 and "no model calls" in result[1]


def test_cleaning_preserves_partial_pdf_pages_and_changes_marker_once(tmp_path):
    from corpus_pipeline.artifacts import write_json
    from corpus_pipeline.cleaning import clean_files
    from corpus_pipeline.processor import _join_pages
    from dataclasses import replace
    config = PipelineConfig(stage_output_dir=tmp_path, include_page_markers=True, page_marker_template="--- Page {page} ---")
    source = tmp_path / "ocr.txt"
    pages = ["one one", "", "three three"]
    source.write_text(_join_pages(pages, config), encoding="utf-8")
    write_json(source.with_suffix(".report.json"), {"stage": "ocr", "source": "original.pdf",
               "status": "partial", "failed_pages": [2], "page_texts": pages,
               "page_markers": [True, config.page_marker_template],
               "pages": [{"page_number": n} for n in (1, 2, 3)]})
    result = clean_files([source], replace(config, page_marker_template="[Page {page}]"))
    record = result["results"][0]
    assert record["status"] == "partial" and record["failed_pages"] == [2]
    assert "--- Page" not in record["cleaned_text"]
    assert record["cleaned_text"].count("[Page") == 3
    assert record["pages"][0]["cleaning_hits"]["adjacent_word_duplicates"] == 1


def test_format_and_semantic_cleaning_are_independent():
    from corpus_pipeline.cleaning import clean_text
    original = "Article I word word<br>next"
    off = PipelineConfig(semantic_rules_enabled=False)
    on = PipelineConfig(semantic_rules_enabled=True)
    assert clean_text(original, config=off, format_cleaning=False).text == original
    semantic = clean_text(original, config=on, format_cleaning=False)
    assert "Article 1" in semantic.text and "word word<br>" in semantic.text
    assert semantic.hit_counts["article_digit_confusion"] == 1


def test_chunking_gui_never_enables_metadata_network_calls(tmp_path, monkeypatch):
    from app import _run_synthesis_chunking
    captured = {}
    def fake(*args):
        captured["use_model"] = args[3]
        return ({}, "done", [], [], [])
    monkeypatch.setattr("app._run_synthesis_metadata", fake)
    _run_synthesis_chunking(None, "", False, 10, str(tmp_path), "openai", "", "", "", False, "")
    assert captured["use_model"] is False


def test_default_processor_output_remains_legacy_cleaned_text(tmp_path):
    from corpus_pipeline.processor import process_file
    source = tmp_path / "input.txt"
    source.write_text("word word<br>next", encoding="utf-8")
    config = PipelineConfig(output_dir=tmp_path / "legacy", cache_db=tmp_path / "cache.sqlite3",
                            semantic_rules_enabled=False, include_page_markers=False, benchmark_enabled=False)
    result = process_file(source, config)
    assert Path(result["output_path"]).read_bytes() == b"word\r\nnext\r\n"
    assert "stage" not in result and "page_texts" not in result
