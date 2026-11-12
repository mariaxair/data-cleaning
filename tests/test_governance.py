import json

import pytest

from config import PipelineConfig
from corpus_pipeline.governance.dedup import normalize_question
from corpus_pipeline.governance.dialect import (
    DialectPrediction, LexiconDialectClassifier, create_classifier, dialect_verdict)
from corpus_pipeline.governance.language import assess_language
from corpus_pipeline.governance.runner import govern_record, process_corpus


class FakeDialect:
    def __init__(self, probability=0.1, degraded=False):
        self.probability, self.degraded = probability, degraded

    def classify(self, text):
        return DialectPrediction({"MSA": 1 - self.probability, "Algerian": 0,
                                  "Moroccan": self.probability}, "fake", self.degraded, True)


def sample():
    return {"id": "x", "question": "هل يجوز للموظف طلب الإجازة؟", "answer": "يجوز للموظف طلب الإجازة.",
            "reasoning_chain": "يتعلق الأمر بطلب إجازة من الموظف.", "language": "ar"}


def test_dedup_arabic_diacritics_and_letter_forms(tmp_path):
    assert normalize_question("أَهْلاً؟") == normalize_question("اهلا")
    first = sample()
    config = PipelineConfig(stage_output_dir=tmp_path)
    result = process_corpus([first, {**first, "id": "y", "question": first["question"] + "!"}],
                            config, classifier=FakeDialect())
    assert result["pass"] == result["duplicates"] == 1
    duplicate = json.loads((tmp_path / "governance/duplicates.jsonl").read_text(encoding="utf-8"))
    assert duplicate["duplicate_of"] == "x"


def test_language_misplacement_mixing_and_latin_uncertainty():
    assert assess_language("عربي فقط", "fr", 0.2)["misplaced"]
    assert assess_language("عربي عربي bonjour bonjour", "ar", 0.2)["decision"] == "intercept"
    assert assess_language("bonjour only latin", "fr", 0.2)["decision"] == "manual"
    assert assess_language("", "ar", 0.2)["decision"] == "manual"


@pytest.mark.parametrize("probability,verdict", [(0.19, "pass"), (0.2, "manual"),
    (0.5, "manual"), (0.51, "intercept")])
def test_validated_dialect_thresholds(probability, verdict):
    assert dialect_verdict(FakeDialect(probability).classify(""), PipelineConfig()) == verdict


def test_degraded_classifier_is_conservative(tmp_path):
    config = PipelineConfig(dialect_lexicon_path=tmp_path / "missing.json")
    prediction = LexiconDialectClassifier(config.dialect_lexicon_path).classify("نص")
    assert prediction.degraded and dialect_verdict(prediction, config) == "manual"
    assert dialect_verdict(FakeDialect(0.8, True).classify(""), config) == "manual"
    assert dialect_verdict(FakeDialect(0.99, True).classify(""), config) == "intercept"
    classifier = create_classifier(PipelineConfig(dialect_backend="marbert",
        dialect_model_path=tmp_path / "not-present", dialect_lexicon_path=config.dialect_lexicon_path))
    assert isinstance(classifier, LexiconDialectClassifier)


def test_per_field_classification_and_default_fallback():
    record = govern_record(sample(), PipelineConfig(), FakeDialect())
    assert record["governance"]["decision"] == "pass"
    assert set(record["governance"]["dialect"]) == {"question", "answer", "reasoning_chain"}
    fallback = govern_record(sample(), PipelineConfig(), LexiconDialectClassifier(__import__("pathlib").Path("missing")))
    assert fallback["governance"]["degraded"] and fallback["governance"]["decision"] == "manual"
