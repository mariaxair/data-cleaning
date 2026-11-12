"""Open theme discovery must be automatic, cached, bounded, and fully offline in tests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from config import PipelineConfig
from corpus_pipeline.artifacts import read_jsonl
from corpus_pipeline.synthesis.runner import synthesize
from corpus_pipeline.synthesis.topics import TopicDiscovery, sample_document
from corpus_pipeline.synthesis.passages import passage_spans, PAGE_PATTERN


class FakeTopicModel:
    """A fake semantic judge matches themes with different words; no keyword classifier."""

    def __init__(self):
        self.calls = []
        self.topic_index = 0

    def generate(self, phase, prompt):
        self.calls.append((phase, prompt))
        if phase == "topic":
            self.topic_index += 1
            if self.topic_index == 1:
                return {"topic": "Réseaux informatiques", "summary": "Communication entre ordinateurs.",
                        "structure": "general", "confidence": 0.9}
            if self.topic_index == 2:
                return {"topic": "Acheminement des paquets", "summary": "Transmission des données sur Internet.",
                        "structure": "general", "confidence": 0.9}
            return {"topic": "Culture des céréales", "summary": "Travaux agricoles et récoltes.",
                    "structure": "general", "confidence": 0.9}
        if phase == "topic_match":
            groups = json.loads(prompt.split("EXISTING GROUPS:\n", 1)[1])
            if "Acheminement des paquets" in prompt:
                return {"group_id": groups[0]["id"], "similarity": 0.94}
            return {"group_id": None, "similarity": 0}
        if phase == "question":
            return {"text": "What do networks connect?", "evidence": [
                {"claim": "networks connect", "quote": "networks connect"}]}
        if phase == "answer":
            return {"text": "Computer networks connect computers.", "evidence": [
                {"claim": "networks connect computers", "quote": "networks connect computers"}]}
        return {"text": "Computer networks make communication possible by connecting computers, which is the concept described in the answer."}


def sources(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"document-{index}.txt"
        path.write_text(f"Document {index}: Computer networks connect computers. " + "Content " * 12,
                        encoding="utf-8")
        paths.append(path)
    return paths


def settings(tmp_path):
    return PipelineConfig(synthesis_screening_mode="automatic", synthesis_concurrency=1,
                          synthesis_language="en", stage_output_dir=tmp_path / "training",
                          general_chunk_chars=300, min_article_chars=20, topic_max_calls=20)


def test_themes_discovered_and_similar_subjects_grouped_without_label_list(tmp_path):
    fake = FakeTopicModel()
    result = TopicDiscovery(settings(tmp_path), fake, False).discover(sources(tmp_path))
    assert len(result["groups"]) == 2
    assert len(result["groups"][0]["members"]) == 2
    assert result["documents"][0]["topic"] != result["documents"][1]["topic"]
    assert result["documents"][0]["group_id"] == result["documents"][1]["group_id"]
    assert not result["degraded"]


def test_identical_documents_reuse_model_result_even_with_different_upload_path(tmp_path):
    paths = sources(tmp_path)[:1]
    copied = tmp_path / "renamed.txt"
    copied.write_bytes(paths[0].read_bytes())
    fake = FakeTopicModel()
    result = TopicDiscovery(settings(tmp_path), fake, False).discover([paths[0], copied])
    assert result["model_calls"] == 1
    assert len(result["groups"]) == 1
    assert len(result["groups"][0]["members"]) == 2


def test_resume_reads_topic_and_matching_cache_without_model_calls(tmp_path):
    config, paths = settings(tmp_path), sources(tmp_path)
    first = TopicDiscovery(config, FakeTopicModel(), False).discover(paths)
    fake = FakeTopicModel()
    second = TopicDiscovery(config, fake, False).discover(paths)
    assert first["groups"] == second["groups"]
    assert not fake.calls and second["model_calls"] == 0
    assert second["cache_hits"] == 5
    # Model changes invalidate the discovery cache.
    changed = TopicDiscovery(replace(config, synthesis_model="different"), FakeTopicModel(), False).discover(paths)
    assert changed["model_calls"] == 5


def test_dry_run_uses_no_model_and_unknown_topics_do_not_block_general_text(tmp_path):
    config, fake = settings(tmp_path), FakeTopicModel()
    plan = synthesize(sources(tmp_path), config, dry_run=True, adapter=fake)
    assert plan["selected_units"] > 0 and plan["topic_pending_documents"] == 3
    assert not fake.calls
    assert all(unit["tier"] == "general" for unit in plan["units"])
    assert not (config.stage_output_dir / "synthesis/qa.jsonl").exists()


def test_topics_only_never_generates_qa_or_overwrites_prior_qa(tmp_path):
    config = settings(tmp_path)
    qa = config.stage_output_dir / "synthesis/qa.jsonl"
    qa.parent.mkdir(parents=True)
    qa.write_text("preserve prior QA", encoding="utf-8")
    fake = FakeTopicModel()
    result = synthesize(sources(tmp_path), config, adapter=fake, topics_only=True)
    assert result["topic_groups"] == 2
    assert all(phase in {"topic", "topic_match"} for phase, _ in fake.calls)
    assert qa.read_text() == "preserve prior QA"
    artifact = json.loads((qa.parent / "topics.json").read_text(encoding="utf-8"))
    assert len(artifact["documents"]) == 3


def test_call_budget_bounds_discovery_and_can_resume(tmp_path):
    config = replace(settings(tmp_path), topic_max_calls=1)
    paths = sources(tmp_path)
    first = TopicDiscovery(config, FakeTopicModel(), False).discover(paths)
    assert first["model_calls"] == 1 and first["degraded"]
    assert any(doc["status"] == "unclassified" for doc in first["documents"])
    resumed = TopicDiscovery(replace(config, topic_max_calls=20), FakeTopicModel(), False).discover(paths)
    assert resumed["documents"][0]["status"] == "discovered"
    assert all(doc["status"] == "discovered" for doc in resumed["documents"])


def test_invalid_topic_and_group_responses_fail_safely_without_exception_leak(tmp_path):
    class Bad:
        def generate(self, *args):
            raise RuntimeError("credential-not-to-be-exported")
    config = replace(settings(tmp_path), topic_max_calls=2)
    result = TopicDiscovery(config, Bad(), False).discover(sources(tmp_path)[:1])
    assert result["degraded"] and result["model_calls"] == 2
    assert "credential-not-to-be-exported" not in json.dumps(result)


def test_multidomain_generation_prompts_and_provenance(tmp_path):
    config, fake = settings(tmp_path), FakeTopicModel()
    result = synthesize(sources(tmp_path)[:1], config, adapter=fake)
    assert result["generated"] == 1 and result["failed"] == 0
    record = read_jsonl(config.stage_output_dir / "synthesis/qa.jsonl")[0]
    assert record["provenance"]["tier"] == "general"
    assert record["provenance"]["topic"] == "Réseaux informatiques"
    assert record["provenance"]["topic_group_id"]
    question_prompt = next(prompt for phase, prompt in fake.calls if phase == "question")
    assert "self-contained learning or practical question" in question_prompt
    assert record["reasoning_type"] == "concept_explanation"
    assert record["provenance"]["char_start"] == 0


def test_general_passages_are_bounded_preserve_content_and_page_origins():
    text = "--- Page 1 ---\n" + "first paragraph. " * 9 + "\n\n--- Page 2 ---\n" + "second paragraph. " * 9
    spans = passage_spans(text, 120)
    assert all(end - start <= 120 for start, end, _ in spans)
    actual = "".join(PAGE_PATTERN.sub("", text[start:end]) for start, end, _ in spans)
    # Only whitespace changes between non-overlapping spans.
    assert "".join(actual.split()) == "".join(PAGE_PATTERN.sub("", text).split())
    assert any(1 in pages for _, _, pages in spans)
    assert any(2 in pages for _, _, pages in spans)


def test_document_sampling_reaches_middle_and_end():
    text = "A" * 1000 + "B" * 1000 + "C" * 1000
    sample = sample_document(text, 300)
    assert "A" in sample and "B" in sample and "C" in sample
    assert len(sample) < 330


def test_legal_route_falls_back_to_passages_when_no_article_groups(tmp_path):
    class MistakenStructure(FakeTopicModel):
        def generate(self, phase, prompt):
            result = super().generate(phase, prompt)
            if phase == "topic":
                result["structure"] = "statutory_law"
            return result
    config = settings(tmp_path)
    result = synthesize(sources(tmp_path)[:1], config, adapter=MistakenStructure())
    assert result["generated"] == 1
    plan = json.loads((config.stage_output_dir / "synthesis/plan.json").read_text(encoding="utf-8"))
    assert plan["units"][0]["tier"] == "general"


def test_gui_automatic_dry_run_needs_no_manual_overrides(tmp_path):
    from app import _run_synthesis
    path = sources(tmp_path)[0]
    summary, status, artifacts = _run_synthesis(
        [str(path)], "", None, "", 10, True, str(tmp_path / "gui"), "heiyu", "",
        "https://example.invalid/v1", "", "automatic", False, 10)
    assert summary["selected_units"] > 0
    assert "no model calls" in status and "themes pending" in status
    assert any(Path(path).name == "topics.json" for path in artifacts)


def test_invalid_group_ids_are_not_accepted(tmp_path):
    class BadMatch(FakeTopicModel):
        def generate(self, phase, prompt):
            if phase == "topic_match":
                self.calls.append((phase, prompt))
                return {"group_id": "invented-not-in-catalogue", "similarity": 1.0}
            return super().generate(phase, prompt)
    config, fake = settings(tmp_path), BadMatch()
    result = TopicDiscovery(config, fake, False).discover(sources(tmp_path)[:2])
    assert result["documents"][1]["group_id"] is None
    assert result["documents"][1]["grouping_status"] == "pending"
    assert len(result["groups"]) == 1 and len(result["groups"][0]["members"]) == 1


def test_general_questions_can_name_real_sections_and_review_is_subject_appropriate(tmp_path, monkeypatch):
    from corpus_pipeline.synthesis.segmentation import SynthesisUnit
    from corpus_pipeline.synthesis.validation import article_consistent, question_unnamed
    from corpus_pipeline.review.adapters import ModelReviewAdapter
    source = "This passage discusses Article 7 of the research paper."
    unit = SynthesisUnit("research.txt", "hash", source, "general")
    assert question_unnamed("What does Article 7 explain?", unit)
    assert article_consistent("Article 7", unit)
    assert not article_consistent("Article 99", unit)
    fake = FakeTopicModel()
    monkeypatch.setattr("corpus_pipeline.review.adapters.CompatibleGenerationAdapter", lambda config: fake)
    ModelReviewAdapter(settings(tmp_path)).review({"provenance": {"tier": "general"}})
    assert "without imposing a legal framework" in fake.calls[0][1]


def test_discovered_groups_are_shown_in_gui_table(tmp_path):
    from app import _topic_group_rows
    config = settings(tmp_path)
    synthesize(sources(tmp_path), config, adapter=FakeTopicModel(), topics_only=True)
    rows = _topic_group_rows(str(config.stage_output_dir), "automatic")
    assert len(rows) == 2 and rows[0][1] == 2
    assert "document-0.txt" in rows[0][2] and "document-1.txt" in rows[0][2]
    assert _topic_group_rows(str(config.stage_output_dir), "legal") == []


def test_cli_topic_only_and_automatic_dry_runs(tmp_path, monkeypatch):
    import cli
    import sys
    path = sources(tmp_path)[0]
    output = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", [
        "cli.py", "--discover-topics", "--dry-run", "--stage-output", str(output), str(path)])
    assert cli.main() == 0
    plan = json.loads((output / "synthesis/plan.json").read_text(encoding="utf-8"))
    assert plan["screening_mode"] == "automatic" and plan["topic_model_calls"] == 0
    assert plan["selected_units"] > 0
    assert not (output / "synthesis/qa.jsonl").exists()
