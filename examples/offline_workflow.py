"""Demonstrate all stages using synthetic text and fake adapters, with no network."""

import json
from pathlib import Path

from config import PipelineConfig
from corpus_pipeline.artifacts import read_jsonl, write_json
from corpus_pipeline.governance.dialect import DialectPrediction
from corpus_pipeline.governance.runner import process_corpus
from corpus_pipeline.processor import process_file
from corpus_pipeline.review.checks import LLM_DIMENSIONS
from corpus_pipeline.review.runner import review_corpus
from corpus_pipeline.synthesis.runner import synthesize

SOURCE = (
    "المادة 1 يحق للموظف طلب إجازة وفق الشروط المحددة في النص.\n"
    "المادة 2 يقدم الطلب إلى الجهة المختصة للفصل فيه.\n")


class DemoGeneration:
    def generate(self, phase: str, prompt: str) -> dict:
        if phase == "question":
            return {"text": "هل يحق للموظف طلب إجازة؟", "evidence": [
                {"claim": "يحق للموظف طلب إجازة", "quote": "يحق للموظف طلب إجازة"}]}
        if phase == "answer":
            return {"text": "يحق للموظف طلب إجازة.", "evidence": [
                {"claim": "يحق للموظف طلب إجازة", "quote": "يحق للموظف طلب إجازة"}]}
        return {"text": "يتعلق السؤال بطلب إجازة من الموظف، ويبين الجواب أن للموظف حق طلب الإجازة وفق النص المبين."}


class DemoClassifier:
    def classify(self, text: str) -> DialectPrediction:
        return DialectPrediction({"MSA": 1.0, "Algerian": 0.0, "Moroccan": 0.0},
                                 "DEMONSTRATION_FAKE", False, True)


class DemoReviewer:
    def review(self, record: dict) -> dict[str, int]:
        return dict.fromkeys(LLM_DIMENSIONS, 2)


def run(root: Path = Path("data/demo")) -> dict:
    """Write demonstration artifacts; fake scores are not evidence of legal quality."""
    root.mkdir(parents=True, exist_ok=True)
    source = root / "synthetic.txt"
    source.write_text(SOURCE, encoding="utf-8")
    config = PipelineConfig(
        ocr_provider="mock", synthesis_model="DEMONSTRATION_FAKE",
        output_dir=root / "text", cache_db=root / "cache.sqlite3", stage_output_dir=root / "training",
        min_article_chars=20, semantic_rules_enabled=True,
        synthesis_enabled=False, corpus_processing_enabled=False, review_enabled=False,
        review_llm_enabled=False)
    cleaned = process_file(source, config, ocr_adapter=object())
    overrides = root / "overrides.json"
    write_json(overrides, {Path(cleaned["output_path"]).name: {"tier": "statutory_law"}})
    config.manual_overrides_path = overrides
    synthesis = synthesize([Path(cleaned["output_path"])], config, adapter=DemoGeneration())
    records = read_jsonl(config.stage_output_dir / "synthesis/qa.jsonl")
    records = [{**record, "demonstration_only": True} for record in records]
    governance = process_corpus(records, config, classifier=DemoClassifier())
    review = review_corpus(read_jsonl(config.stage_output_dir / "governance/pass.jsonl"),
                           config, classifier=DemoClassifier(), reviewer=DemoReviewer())
    return {"demonstration_only": True, "synthesis": synthesis, "governance": governance, "review": review}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
