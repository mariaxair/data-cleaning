import json
import sys

import pytest

import cli
from config import PipelineConfig
from corpus_pipeline.cleaning import clean_text
from corpus_pipeline.ocr.base import OCRAdapter
from corpus_pipeline.synthesis.runner import synthesize
from corpus_pipeline.synthesis.segmentation import SynthesisUnit, segment_document
from corpus_pipeline.synthesis.screening import ScreeningResult


def test_cross_family_mapping_is_rejected_if_not_idempotent(tmp_path):
    words = tmp_path / "words.tsv"
    words.write_text("wrong\tcorrect\nexample\tword word\n", encoding="utf-8")
    config = PipelineConfig(semantic_rules_enabled=True, wrong_words_path=words)
    with pytest.raises(ValueError, match="idempotent"):
        clean_text("example", config=config)


def test_error_detail_masks_arbitrary_exact_key_before_truncation():
    class Response:
        def json(self):
            return {"error": {"message": "rejected " + "opaque-key-without-sk-prefix" * 20}}
    key = "opaque-key-without-sk-prefix" * 20
    detail = OCRAdapter._response_detail(Response(), key)
    assert "opaque-key" not in detail and "***" in detail


def test_chapter_boundaries_and_page_provenance():
    source = ("--- Page 1 ---\nChapter One\nArticle 1 " + "substance " * 5
              + "\n--- Page 2 ---\nArticle 2 " + "content " * 8
              + "\nChapter Two\nArticle 3 " + "substance " * 5
              + "\nArticle 4 " + "content " * 8)
    from pathlib import Path
    units = segment_document(source, Path("law.txt"), ScreeningResult("statutory_law", False))
    assert [u.article_numbers for u in units] == [["1", "2"], ["3", "4"]]
    assert units[0].pages == [1, 2] and units[1].pages == [2]


def test_cli_dry_run_and_governance_review_offline(tmp_path, monkeypatch):
    source = tmp_path / "plain.txt"
    source.write_text("Unclassified input", encoding="utf-8")
    root = tmp_path / "training"
    monkeypatch.setattr(sys, "argv", ["cli.py", "--synthesize", "--dry-run", "--stage-output", str(root), str(source)])
    assert cli.main() == 0
    plan = json.loads((root / "synthesis/plan.json").read_text(encoding="utf-8"))
    assert plan["selected_units"] == 0
    sample = tmp_path / "samples.jsonl"
    sample.write_text(json.dumps({"question": "هل يجوز؟", "answer": "يجوز.",
                                  "reasoning_chain": "ذلك جائز وفق النص.", "language": "ar"}), encoding="utf-8")
    for stage in ("--process-corpus", "--review"):
        monkeypatch.setattr(sys, "argv", ["cli.py", stage, "--stage-output", str(root), str(sample)])
        assert cli.main() == 0
    assert json.loads((root / "governance/summary.json").read_text())["manual"] == 1
    assert json.loads((root / "review/summary.json").read_text())["manual"] == 1


def test_cli_conflicting_stages_and_invalid_dry_run(monkeypatch):
    for args in (["--review", "--synthesize"], ["--dry-run"]):
        monkeypatch.setattr(sys, "argv", ["cli.py", *args])
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2


def test_optional_environment_settings_are_read_at_config_creation(monkeypatch):
    monkeypatch.setenv("SYNTHESIS_ENABLED", "true")
    monkeypatch.setenv("SYNTHESIS_MAX_SAMPLES", "7")
    monkeypatch.setenv("DIALECT_PASS_THRESHOLD", "0.15")
    config = PipelineConfig()
    assert config.synthesis_enabled and config.synthesis_max_samples == 7
    assert config.dialect_pass_threshold == 0.15
    config.validate_training()
    config.synthesis_max_samples = 0
    with pytest.raises(ValueError, match="positive"):
        config.validate_training()
