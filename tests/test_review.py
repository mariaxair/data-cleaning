import json

from config import PipelineConfig
from corpus_pipeline.governance.dialect import DialectPrediction
from corpus_pipeline.review.checks import LLM_DIMENSIONS, RULE_DIMENSIONS, deterministic_review
from corpus_pipeline.review.runner import repair_fields, review_corpus, review_record, score_record


class FakeDialect:
    def __init__(self, probability=0.01):
        self.probability = probability

    def classify(self, text):
        return DialectPrediction({"MSA": 1 - self.probability, "Algerian": 0,
                                  "Moroccan": self.probability}, "fake", False, True)


class FakeReview:
    def review(self, record):
        return dict.fromkeys(LLM_DIMENSIONS, 2)


def record():
    return {
        "id": "one", "question": "هل يجوز للموظف طلب إجازة؟",
        "answer": "يجوز للموظف طلب إجازة.", "reasoning_chain": "يحق للموظف أن يطلب إجازة وفق النص.",
        "language": "ar", "source_text": "المادة 1 يجوز للموظف طلب إجازة.",
        "provenance": {"source_file": "law.txt", "article_numbers": ["1"]},
    }


def test_all_nine_listed_dimensions_and_acceptance(tmp_path):
    config = PipelineConfig(stage_output_dir=tmp_path)
    result = review_record(record(), config, FakeDialect(), reviewer=FakeReview())
    assert result["status"] == "accept"
    assert set(result["scores"]) == set(LLM_DIMENSIONS + RULE_DIMENSIONS)
    assert set(result["scores"].values()) == {2}
    summary = review_corpus([record()], config, reviewer=FakeReview(), classifier=FakeDialect())
    assert summary["accept"] == 1
    accepted = json.loads((tmp_path / "review/accepted.jsonl").read_text(encoding="utf-8"))
    assert "source_text" not in accepted and accepted["provenance"]["source_file"] == "law.txt"


def test_skipped_review_is_manual_and_cannot_accept():
    result = review_record(record(), PipelineConfig(), FakeDialect())
    assert result["status"] == "manual"
    assert set(LLM_DIMENSIONS) <= set(result["review"]["skipped_dimensions"])


def test_dialect_veto_does_not_call_review_or_repair():
    class Forbidden:
        def review(self, sample):
            raise AssertionError("No semantic call allowed on a dialect-vetoed sample.")
        def repair(self, *args):
            raise AssertionError("No repair allowed on a dialect-vetoed sample.")
    result = review_record(record(), PipelineConfig(), FakeDialect(0.9),
                           reviewer=Forbidden(), repairer=Forbidden())
    assert result["status"] == "reject"
    assert result["scores"]["dialect_compliance"] == 0
    assert result["review"]["attempts"] == 0


def test_targeted_reasoning_repair_preserves_question_and_answer():
    sample = record()
    sample["reasoning_chain"] += " English"
    original = dict(sample)

    class Repair:
        calls = []
        def repair(self, record, fields, issues):
            self.calls.append(fields)
            return {"reasoning_chain": "يحق للموظف أن يطلب إجازة وفق النص."}
    repairer = Repair()
    result = review_record(sample, PipelineConfig(), FakeDialect(),
                           reviewer=FakeReview(), repairer=repairer)
    assert result["status"] == "accept"
    assert repairer.calls == [["reasoning_chain"]]
    assert result["question"] == original["question"] and result["answer"] == original["answer"]
    assert sample == original  # Caller record was not mutated.


def test_defective_question_routes_question_and_answer_and_exhausts():
    sample = record()
    sample["question"] = "ما نص المادة 1؟"
    scores = score_record(sample, PipelineConfig(), FakeDialect(), FakeReview())
    assert repair_fields(scores) == ["question", "answer"]

    class UnchangedRepair:
        def repair(self, record, fields, issues):
            return {field: record[field] for field in fields}
    result = review_record(sample, PipelineConfig(review_max_attempts=1), FakeDialect(),
                           reviewer=FakeReview(), repairer=UnchangedRepair())
    assert result["status"] == "reject" and result["review"]["attempts"] == 1


def test_deterministic_failures_and_missing_source():
    sample = record()
    sample["answer"] = "المادة 99 \ufffd"
    result = deterministic_review(sample, PipelineConfig(), FakeDialect())
    assert result.scores["corpus_accuracy"] == 0
    assert repair_fields(result) == ["answer"]
    sample = record()
    sample["source_text"] = ""
    result = review_record(sample, PipelineConfig(), FakeDialect(), reviewer=FakeReview())
    assert result["status"] == "manual"


def test_repair_cannot_change_provenance_and_errors_do_not_leak():
    sample = record()
    sample["reasoning_chain"] += " foreign"

    class BadRepair:
        def repair(self, *args):
            return {"provenance": {"secret": "never-export-this"}}
    result = review_record(sample, PipelineConfig(), FakeDialect(),
                           reviewer=FakeReview(), repairer=BadRepair())
    assert result["status"] == "reject"
    assert result["provenance"] == sample["provenance"]
    assert "never-export-this" not in json.dumps(result)


def test_markdown_repetition_and_privacy_checks():
    sample = record()
    sample["answer"] = "<sup>1</sup> " + "كلمة كلمة كلمة"
    sample["question"] = "شخص person@example.com"
    result = deterministic_review(sample, PipelineConfig(), FakeDialect())
    assert result.scores["markdown_compliance"] == 0
    assert result.scores["fluency_no_repetition"] == 0
    assert result.scores["privacy_compliance"] == 0
