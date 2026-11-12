import json
from dataclasses import replace

import pytest

from config import PipelineConfig
from corpus_pipeline.synthesis.client import CompatibleGenerationAdapter
from corpus_pipeline.synthesis.runner import synthesize
from corpus_pipeline.synthesis.screening import ScreeningResult, screen_document
from corpus_pipeline.synthesis.segmentation import SynthesisUnit, segment_document
from corpus_pipeline.synthesis.validation import (
    article_consistent, language_pure, privacy_clear, question_unnamed, validate_generation)

SOURCE = "المادة 1 يحق للموظف طلب إجازة وفق الشروط المحددة في النص.\nالمادة 2 يقدم الطلب إلى الجهة المختصة للفصل فيه."
UNIT = SynthesisUnit("law.txt", "sha", SOURCE, "statutory_law", "قانون الاختبار", "99-10", ["1", "2"])


class FakeGeneration:
    def __init__(self):
        self.calls = []

    def generate(self, phase, prompt):
        self.calls.append((phase, prompt))
        if phase == "question":
            return {"text": "هل يحق للموظف طلب إجازة؟",
                    "evidence": [{"claim": "يحق للموظف طلب إجازة", "quote": "يحق للموظف طلب إجازة"}]}
        if phase == "answer":
            return {"text": "يحق للموظف طلب إجازة.",
                    "evidence": [{"claim": "يحق للموظف طلب إجازة", "quote": "يحق للموظف طلب إجازة"}]}
        return {"text": "يتعلق السؤال بالموظف الذي يطلب إجازة، ويبين الجواب أن للموظف حق طلب الإجازة وفق الإطار المبين."}


def setup_synthesis(tmp_path):
    path = tmp_path / "law.txt"
    path.write_text(SOURCE, encoding="utf-8")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"law.txt": {"tier": "statutory_law", "law_title": "قانون الاختبار",
                                                "law_number": "99-10"}}), encoding="utf-8")
    config = PipelineConfig(stage_output_dir=tmp_path / "training", manual_overrides_path=overrides,
                            synthesis_concurrency=2, min_article_chars=20)
    return path, config


def test_screening_requires_evidence_or_manual_override():
    assert screen_document("text", __import__("pathlib").Path("x"), {}, {}).tier == "manual"
    assert screen_document("agenda", __import__("pathlib").Path("x"),
                           {"administrative_noise": ["agenda"], "validated": True}, {}).tier == "administrative_noise"
    assert screen_document("law agenda", __import__("pathlib").Path("x"),
                           {"statutory_law": ["law"], "administrative_noise": ["agenda"]}, {}).tier == "manual"


def test_segmentation_boundaries_filters_and_case_law(tmp_path):
    source = "masthead\n" + "\n".join(f"Article {i} " + "substantive content " * 3 for i in range(1, 8))
    units = segment_document(source, tmp_path / "law.txt", ScreeningResult("statutory_law", False))
    assert [len(unit.article_numbers) for unit in units] == [3, 2, 2]
    assert "masthead" not in units[0].text
    ruling = segment_document(source, tmp_path / "law.txt", ScreeningResult("case_law", False))
    assert len(ruling) == 1 and ruling[0].text == source.strip()
    assert not segment_document("Article 1 \ufffd", tmp_path / "law.txt", ScreeningResult("statutory_law", False))
    assert not segment_document(source, tmp_path / "law.txt", ScreeningResult("administrative_noise", False))
    filtered = segment_document(source, tmp_path / "law.txt", ScreeningResult("statutory_law", False),
                                policy={"exclude_article_terms": ["Article 3"]})
    assert "3" not in [number for unit in filtered for number in unit.article_numbers]


@pytest.mark.parametrize("check,good,bad", [
    (lambda text: language_pure(text, "ar"), "الجواب", "الجواب latin"),
    (lambda text: article_consistent(text, UNIT), "المادة 1", "المادة 99"),
    (lambda text: privacy_clear(text, ["اسم حساس"]), "الموظف", "اسم حساس"),
    (lambda text: privacy_clear(text, []), "الموظف", "person@example.com"),
    (lambda text: question_unnamed(text, UNIT), "هل يجوز؟", "ما نص المادة 1؟"),
    (lambda text: question_unnamed(text, UNIT), "هل يجوز؟", "قانون الاختبار"),
])
def test_validators(check, good, bad):
    assert check(good)
    assert not check(bad)


def test_evidence_validation_and_unproven_semantics():
    valid = FakeGeneration().generate("question", "")
    assert validate_generation("question", valid, UNIT, "ar").passed
    assert "semantic_grounding" in validate_generation("question", valid, UNIT, "ar").pending
    assert "missing_evidence" in validate_generation("question", {"text": "هل يجوز؟"}, UNIT, "ar").errors
    valid["evidence"][0]["quote"] = "not in source"
    assert "invalid_evidence" in validate_generation("question", valid, UNIT, "ar").errors
    assert "unsupported_number" in validate_generation("question", {"text": "هل مر 99 يوما؟"}, UNIT, "ar").errors


def test_synthesis_provenance_resume_and_no_source_in_explanation(tmp_path):
    path, config = setup_synthesis(tmp_path)
    fake = FakeGeneration()
    assert synthesize([path], config, adapter=fake)["generated"] == 1
    record = json.loads((config.stage_output_dir / "synthesis/qa.jsonl").read_text(encoding="utf-8"))
    assert record["provenance"]["source_file"] == str(path)
    assert record["provenance"]["article_numbers"] == ["1", "2"]
    assert record["provenance"]["law_title"] == "قانون الاختبار"
    assert record["native_chain_of_thought"] is None and record["scores"] == {}
    assert SOURCE not in fake.calls[-1][1] and "المادة 2" not in fake.calls[-1][1]
    assert synthesize([path], config, adapter=fake)["generated"] == 1
    assert len(fake.calls) == 3  # Valid phases were loaded from checkpoints.


def test_dry_run_and_missing_live_configuration(tmp_path, monkeypatch):
    path, config = setup_synthesis(tmp_path)
    fake = FakeGeneration()
    assert synthesize([path], config, dry_run=True, adapter=fake)["selected_units"] == 1
    assert not fake.calls
    with pytest.raises(ValueError, match="SYNTHESIS_MODEL"):
        CompatibleGenerationAdapter(config)
    monkeypatch.delenv(config.api_key_env, raising=False)
    monkeypatch.delenv("OCR_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="requires an API key"):
        CompatibleGenerationAdapter(replace(config, synthesis_model="fake", ocr_api_key=""))


def test_invalid_responses_are_bounded_and_not_exported(tmp_path):
    path, config = setup_synthesis(tmp_path)

    class Invalid:
        calls = 0
        def generate(self, phase, prompt):
            self.calls += 1
            return {"text": "bad"}
    fake = Invalid()
    summary = synthesize([path], config, adapter=fake, max_samples=1)
    assert summary["failed"] == 1
    assert fake.calls == config.synthesis_max_attempts
    assert not (config.stage_output_dir / "synthesis/qa.jsonl").read_text()


def test_resume_keeps_question_after_answer_phase_failure(tmp_path):
    path, config = setup_synthesis(tmp_path)

    class FailingAnswer(FakeGeneration):
        def generate(self, phase, prompt):
            if phase == "answer":
                self.calls.append((phase, prompt))
                return {"text": "bad"}
            return super().generate(phase, prompt)
    first = FailingAnswer()
    assert synthesize([path], config, adapter=first)["failed"] == 1
    resumed = FakeGeneration()
    assert synthesize([path], config, adapter=resumed)["generated"] == 1
    assert [phase for phase, _ in resumed.calls] == ["answer", "explanation"]


def test_sample_cap_and_original_extraction_provenance(tmp_path):
    path, config = setup_synthesis(tmp_path)
    path.with_suffix(".report.json").write_text(json.dumps({
        "source": "original.pdf", "source_sha256": "original-sha", "failed_pages": [], "status": "complete"
    }), encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text(SOURCE, encoding="utf-8")
    overrides = json.loads(config.manual_overrides_path.read_text(encoding="utf-8"))
    overrides["other.txt"] = {"tier": "statutory_law"}
    config.manual_overrides_path.write_text(json.dumps(overrides), encoding="utf-8")
    fake = FakeGeneration()
    assert synthesize([path, other], config, adapter=fake, max_samples=1)["selected"] == 1
    assert len(fake.calls) == 3
    record = json.loads((config.stage_output_dir / "synthesis/qa.jsonl").read_text(encoding="utf-8"))
    assert record["provenance"]["source_file"] == "original.pdf"
    assert record["provenance"]["source_sha256"] == "original-sha"
    assert record["provenance"]["cleaned_source_file"] == str(path)


def test_partial_extractions_are_excluded_even_with_manual_tier(tmp_path):
    path, config = setup_synthesis(tmp_path)
    path.with_suffix(".report.json").write_text(json.dumps({
        "source": "original.pdf", "failed_pages": [2], "status": "partial"}), encoding="utf-8")
    plan = synthesize([path], config, dry_run=True)
    assert plan["selected_units"] == 0
    assert "Partial extraction" in plan["documents"][0]["warnings"][0]


def test_generation_errors_never_serialize_adapter_exception(tmp_path):
    path, config = setup_synthesis(tmp_path)

    class Broken:
        def generate(self, *args):
            raise RuntimeError("opaque-credential-must-not-leak")
    result = synthesize([path], config, adapter=Broken())
    assert result["failed"] == 1
    assert "opaque-credential-must-not-leak" not in (
        config.stage_output_dir / "synthesis/failed.jsonl").read_text(encoding="utf-8")
