"""Metadata contracts and opt-in workflows, always offline."""

from corpus_pipeline.artifacts import read_jsonl, write_jsonl
from corpus_pipeline.models import Evidence, MetadataDocument, MetadataObservation
from corpus_pipeline.metadata.legal import legal_extract, normalize_date
import pytest
from config import PipelineConfig
from types import SimpleNamespace
from corpus_pipeline.metadata.chunking import chunk_document
from corpus_pipeline.metadata.extraction import metadata_prompt, validate_observations
from corpus_pipeline.metadata.registry import FieldRegistry
from corpus_pipeline.metadata.runner import discover_metadata
from corpus_pipeline.settings import load_json


def test_metadata_contract_unknown_multidomain_dates_roundtrip(tmp_path):
    signed = MetadataObservation("signature", "2026-01-13", "date", evidence=[
        Evidence("2026-01-13", 0, 10, 1)], role="signature", confidence=1, status="verified")
    published = MetadataObservation("publication", "2026-02-14", "date", role="publication")
    record = MetadataDocument("doc", "example.txt", "hash", dates=[signed, published],
                              domains=[{"original_label": "droit", "value": "droit"},
                                       {"original_label": "sport", "value": "sport"}])
    path = tmp_path / "documents.jsonl"
    write_jsonl(path, [record.to_dict()])
    restored = MetadataDocument.from_dict({**read_jsonl(path)[0], "future_optional": True})
    assert restored == record
    assert restored.language["value"] == "unknown"
    assert len(restored.domains) == len(restored.dates) == 2
    assert restored.metadata_status == "extraction_pending"
    assert restored.dates[0].evidence[0].quote == "2026-01-13"


def test_metadata_old_optional_fields_are_compatible():
    record = MetadataDocument.from_dict({"document_id": "id", "source": "x", "source_sha256": "h"})
    assert record.metadata == record.dates == []
    assert record.metadata_schema_version == "1.0"


@pytest.mark.parametrize("heading,number", [
    ("## المادة الأولى", "الأولى"), ("# المادة ٢", "٢"), ("**المادة 3** : نص", "3"),
    ("### Article 4", "4"), ("**Art. ٥**", "٥")])
def test_legal_structure_variants_exact_evidence(heading, number):
    text = ("--- Page 1 ---\n# الجمهورية الجزائرية الديمقراطية الشعبية\n"
            "## مرسوم رئاسي رقم ٢٦-٩٠ مؤرخ في 26 جانفي سنة 2026\n"
            "# الفصل الأول\n" + heading + "\nالمضمون يبقى كما ورد.\n--- Page 2 ---\nالمادة 6 : باقي النص.")
    result = legal_extract(text)
    assert result["status"] == "complete"
    assert result["spans"] and "".join(text[s["start"]:s["end"]] for s in result["spans"]) == text
    assert number in [n for s in result["spans"] for n in s["article_numbers"]]
    for observation in result["observations"]:
        assert all(text[e.start:e.end] == e.quote for e in observation.evidence)
    number_obs = next(o for o in result["observations"] if o.label == "legal_instrument_number")
    assert number_obs.value == "٢٦-٩٠" and number_obs.normalized_value == "26-90"


def test_legal_missing_identifier_or_heading_needs_fallback():
    for text in ("الجريدة الرسمية\nالمادة الأولى\nالنص", "قانون رقم 26-90\nالمادة : لا رقم"):
        assert legal_extract(text)["status"] == "needs_fallback"
    assert not legal_extract("This research discusses article 2 of a law.")["likely"]


def test_legal_dates_roles_and_invalid_normalization():
    text = "قانون رقم 26-90\nالمادة 1 : النص\nحرر بالجزائر في 13 جانفي سنة 2026\nنشر في 2026-02-14"
    dates = [o for o in legal_extract(text)["observations"] if o.value_type == "date"]
    assert [o.role for o in dates] == ["signature", "publication"]
    assert [o.normalized_value for o in dates] == ["2026-01-13", "2026-02-14"]
    assert normalize_date("31 février 2026") is None
    assert normalize_date("03/04/2026") is None
    assert normalize_date("2026-02-30") is None


@pytest.mark.parametrize("text", ["", "قصير", "Bonjour ؟ العربية؛\n\n" * 30, "x" * 1000])
def test_fallback_chunks_reconstruct_and_bound_multilingual_text(text):
    config = PipelineConfig(metadata_chunk_chars=60, metadata_chunk_tokens=80)
    chunks, info = chunk_document(text, config)
    assert "".join(c["text"] for c in chunks) == text
    assert info["degraded"] and info["reconstruction_exact"]
    for chunk in chunks:
        assert chunk["text"] == text[chunk["start"]:chunk["end"]]
        assert chunk["char_count"] <= 60 and chunk["token_estimate_upper"] <= 80


def test_semantic_adapter_validation_legal_exclusion_and_dry_run(monkeypatch):
    import corpus_pipeline.metadata.chunking as module
    text = "Bonjour. العربية؟ fin؛"
    config = PipelineConfig(metadata_semantic_enabled=True)
    calls = []

    class Fake:
        def chunk(self, source):
            calls.append(source)
            return [SimpleNamespace(start_index=0, end_index=len(source), text=source, token_count=10)]

    chunks, info = chunk_document(text, config, chunker=Fake())
    assert info["chunker"] == "chonkie_semantic_1.5.2" and chunks[0]["text"] == text
    calls.clear()
    chunks, info = chunk_document(text, config, legal_spans=[{"start": 0, "end": len(text)}], chunker=Fake())
    assert info["chunker"] == "legal_structure" and not calls
    chunks, info = chunk_document(text, config, dry_run=True, chunker=Fake())
    assert not calls and "Dry run" in info["fallback_reason"]
    monkeypatch.setattr(module, "_semantic", lambda _: (_ for _ in ()).throw(ImportError()))
    assert chunk_document(text, config)[1]["degraded"]
    config.metadata_skip_window = 1
    with pytest.raises(ValueError, match="contiguous"):
        chunk_document(text, config)


def test_invalid_semantic_offsets_fail_closed():
    class Wrong:
        def chunk(self, source):
            return [SimpleNamespace(start_index=1, end_index=len(source), text=source[1:], token_count=2)]
    chunks, info = chunk_document("التجربة؛ Bonjour.", PipelineConfig(metadata_semantic_enabled=True), chunker=Wrong())
    assert info["degraded"] and chunks[0]["start"] == 0


def observation(label="university", value="Example University", **kwargs):
    return {"label": label, "value": value, "value_type": "string", "scope": "document",
            "evidence": [{"quote": "Example University"}], "confidence": 0.95, **kwargs}


def test_open_observations_unseen_fields_and_evidence():
    text = "--- Page 2 ---\nExample University\n2026-02-14"
    raw = observation(label="new unexpected field")
    result = validate_observations({"observations": [raw]}, text, 0, len(text))[0]
    assert result.status == "verified" and result.evidence[0].page == 2
    assert result.evidence[0].quote == text[result.evidence[0].start:result.evidence[0].end]
    assert "untrusted" in metadata_prompt(text, {"start": 0, "end": len(text)})


@pytest.mark.parametrize("overrides", [
    {"evidence": []}, {"evidence": [{"quote": "invented"}]}, {"value": "different value"},
    {"confidence": float("nan")}, {"value_type": "unapproved"}, {"value_type": {}}, {"scope": []},
    {"evidence": [{"quote": "Example University", "start": 3, "end": 21}]},
    {"evidence": [{"quote": "Example University", "page": 7}]},
])
def test_bad_open_observations_are_audited(overrides):
    text = "Example University"
    record = validate_observations({"observations": [observation(**overrides)]}, text, 0, len(text))[0]
    assert record.status == "rejected"


def test_open_ambiguous_quotes_inference_and_date_consistency():
    text = "Example University Example University"
    record = validate_observations({"observations": [observation()]}, text, 0, len(text))[0]
    assert record.status == "rejected"
    text = "Example University"
    record = validate_observations({"observations": [observation(assertion="inferred")]}, text, 0, len(text))[0]
    assert record.status == "uncertain"
    text = "2026-02-14 through 2026-01-13"
    raw = observation(value=["2026-02-14", "2026-01-13"], value_type="date_range", evidence=[{"quote": text}])
    assert validate_observations({"observations": [raw]}, text, 0, len(text))[0].status == "rejected"
    with pytest.raises(ValueError, match="observations array"):
        validate_observations({"schema": {}}, text, 0, len(text))


def verified(label="author", value="Example", **kwargs):
    return MetadataObservation(label, value, evidence=[Evidence(str(value), 0, len(str(value)))],
                               confidence=0.95, status="verified", **kwargs)


def test_registry_recurrence_reruns_migrations_and_multilingual_aliases(tmp_path):
    registry = FieldRegistry(tmp_path, PipelineConfig(metadata_promotion_documents=2))
    first = registry.match(verified(), "doc-1")
    field_id = first.field_id
    registry.match(verified(), "doc-1")
    assert registry.entries[field_id]["occurrence_count"] == 1
    registry.match(verified(), "doc-2")
    assert registry.entries[field_id]["status"] == "stable"
    assert not registry.entries[field_id]["quality_approved"]
    old_version = registry.version
    registry.migrate("rename", field_id, label="auteur", reason="Reviewed translation")
    assert registry.match(verified(), "doc-3").field_id == field_id
    registry.migrate("rename", field_id, label="مؤلف", reason="Reviewed translation")
    assert registry.match(verified("auteur"), "doc-4").field_id == field_id
    assert registry.version > old_version
    # The journal can recover a missing snapshot after interruption.
    (tmp_path / "field-registry.json").unlink()
    recovered = FieldRegistry(tmp_path, registry.config)
    assert recovered.version == registry.version
    assert recovered.entries[field_id]["occurrence_count"] == 4
    history = read_jsonl(tmp_path / "registry-history.jsonl")
    assert {"new_field", "promotion", "rename", "occurrence"} <= {e["action"] for e in history}


def test_registry_type_scope_homonym_and_spelling_conflicts(tmp_path):
    registry = FieldRegistry(tmp_path, PipelineConfig())
    registry.match(verified("date", description="signature date"), "a")
    for obs in (verified("date", value_type="date"), verified("date", scope="chunk"),
                verified("date", description="fruit of the date palm")):
        result = registry.match(obs, "b")
        assert result.mapping_status == "uncertain" and result.field_id is None
    # Semantic similarity alone is never asserted by a spelling-only matcher.
    assert registry.match(verified("signed date"), "b").mapping_status == "candidate"


def test_registry_merge_split_type_change_deprecation_history(tmp_path):
    registry = FieldRegistry(tmp_path, PipelineConfig())
    left = registry.match(verified("author"), "a").field_id
    right = registry.match(verified("auteur"), "b").field_id
    registry.migrate("merge", left, target_id=right, reason="Manual synonym evidence")
    assert registry.entries[left]["status"] == "deprecated"
    assert registry.match(verified("author"), "c").field_id == right
    children = registry.migrate("split", right, children=["writer", "editor"], reason="Different roles")
    registry.migrate("type_change", children[1], value_type="number", reason="Reviewed type")
    registry.migrate("deprecation", children[2], reason="Not useful")
    actions = {e["action"] for e in read_jsonl(tmp_path / "registry-history.jsonl")}
    assert {"merge", "split", "type_change", "deprecation"} <= actions
    assert all(e["fields_before"] for e in read_jsonl(tmp_path / "registry-history.jsonl")
               if e["action"] in {"merge", "split", "type_change", "deprecation"})


def metadata_config(tmp_path, **kwargs):
    return PipelineConfig(stage_output_dir=tmp_path / "training", metadata_chunk_tokens=10000, **kwargs)


def test_metadata_legal_offline_and_generic_unknown(tmp_path):
    law, general = tmp_path / "law.md", tmp_path / "research.txt"
    law.write_text("# مرسوم رئاسي رقم 26-90\n## المادة الأولى\nنص القانون.\nالمادة ٢ : النص.", encoding="utf-8")
    general.write_text("Example University publishes this research.", encoding="utf-8")
    config = metadata_config(tmp_path)
    result = discover_metadata([law, general], config)
    assert result["processed"] == 2 and result["model_calls"] == 0
    docs = read_jsonl(config.stage_output_dir / "metadata/documents.jsonl")
    legal = next(d for d in docs if d["document_type"] == "structured_legal")
    assert legal["metadata_status"] == "complete"
    assert legal["domains"][0]["value"] == "law"
    assert legal["processing"]["chunker"] == "legal_structure"
    assert "".join(c["text"] for c in legal["chunks"]) == law.read_bytes().decode("utf-8")
    generic = next(d for d in docs if d["document_type"] != "structured_legal")
    assert generic["metadata_status"] == "extraction_pending"
    assert generic["language"]["value"] == generic["domains"][0]["value"] == "unknown"


class FakeMetadata:
    def __init__(self):
        self.calls = []

    def generate(self, phase, prompt):
        self.calls.append((phase, prompt))
        return {"observations": [observation(), observation(label="domain", value="education", value_type="domain"),
                                  observation(label="domain", value="research", value_type="domain")]}


def test_metadata_model_provenance_cache_incremental_and_dry_run(tmp_path):
    source = tmp_path / "research.txt"
    source.write_text("Example University researches scientific questions.", encoding="utf-8")
    config = metadata_config(tmp_path, metadata_model="fake")
    fake = FakeMetadata()
    result = discover_metadata([source], config, adapter=fake)
    assert result["model_calls"] == 1 and len(fake.calls) == 1
    output = config.stage_output_dir / "metadata"
    doc = read_jsonl(output / "documents.jsonl")[0]
    assert {d["value"] for d in doc["domains"]} == {"education", "research"}
    assert doc["processing"]["response_fingerprints"]
    assert doc["metadata_schema_version"] == "1.0" and doc["registry_version"] > 0
    version = load_json(output / "field-registry.json")["version"]
    fake.calls.clear()
    again = discover_metadata([source], config, adapter=fake)
    assert again["cache_hits"] == 1 and not fake.calls
    assert load_json(output / "field-registry.json")["version"] == version
    original = (output / "documents.jsonl").read_bytes()
    plan = discover_metadata([source], config, dry_run=True, adapter=fake)
    assert plan["model_calls"] == 0 and (output / "documents.jsonl").read_bytes() == original
    other = tmp_path / "other.txt"
    other.write_text("Unrelated text.", encoding="utf-8")
    discover_metadata([other], config)
    assert len(read_jsonl(output / "documents.jsonl")) == 2


def test_metadata_budget_resume_failure_no_secret_leak_and_preflight(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Example University\n" * 4, encoding="utf-8")
    config = metadata_config(tmp_path, metadata_model="fake", metadata_chunk_chars=22, metadata_max_calls=1)
    result = discover_metadata([source], config, adapter=FakeMetadata())
    assert result["model_calls"] == 1
    assert result["statuses"]["extraction_pending"] == 1
    key = "private-test-credential"

    class Bad:
        def generate(self, phase, prompt):
            raise RuntimeError(key)

    config.ocr_api_key = key
    config.session_api_key_override = True
    discover_metadata([source], config, adapter=Bad())
    for path in (config.stage_output_dir / "metadata").rglob("*"):
        if path.is_file():
            assert key not in path.read_text(encoding="utf-8")
    config.metadata_model = ""
    config.metadata_llm_enabled = True
    with pytest.raises(ValueError, match="METADATA_MODEL"):
        discover_metadata([source], config)


def test_metadata_cli_and_gui_dry_run_offline(tmp_path, monkeypatch):
    import cli
    from app import _run_metadata
    path = tmp_path / "source.txt"
    path.write_text("Example University", encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["cli.py", "--metadata", "--dry-run", "--stage-output", str(output), str(path)])
    assert cli.main() == 0
    result = _run_metadata([str(path)], "", True, False, 5, str(output), "heiyu", "", "", "")
    assert result[0]["model_calls"] == 0 and "Dry run" in result[1]
    assert result[-1]


def test_registry_synonym_evidence_and_multiple_plausible_fields(tmp_path):
    registry = FieldRegistry(tmp_path, PipelineConfig())
    first = registry.match(verified("signature date", description="Date on which document was signed"), "a")
    second = registry.match(verified("date de signature", description="Date on which document was signed"), "b")
    assert second.field_id == first.field_id
    assert "date de signature" in registry.entries[first.field_id]["aliases"]
    other = registry.match(verified("another field"), "c").field_id
    # An explicit alias collision creates ambiguity, not an arbitrary winner.
    registry.entries[other]["aliases"].append("signature date")
    assert registry.match(verified("signature date"), "d").mapping_status == "uncertain"


def test_absence_invalid_dates_and_ocr_damage_are_not_verified(tmp_path):
    missing = observation(value=None, assertion="absent", evidence=[])
    assert validate_observations({"observations": [missing]}, "", 0, 0)[0].status == "missing"
    for date_text in ("2026-02-30", "31 février 2026", "31/02/2026"):
        raw = observation(value=date_text, value_type="date", evidence=[{"quote": date_text}])
        assert validate_observations({"observations": [raw]}, date_text, 0, len(date_text))[0].status == "rejected"
    source = tmp_path / "damaged.txt"
    source.write_text("المادة 1 \ue000 broken", encoding="utf-8")
    fake = FakeMetadata()
    config = metadata_config(tmp_path)
    result = discover_metadata([source], config, adapter=fake)
    assert result["model_calls"] == 0 and result["statuses"]["partial"] == 1


def test_metadata_additive_provenance_and_disabled_qa_bytes(tmp_path):
    from examples.offline_workflow import DemoGeneration, SOURCE
    from corpus_pipeline.synthesis.runner import synthesize
    from corpus_pipeline.artifacts import write_json
    path = tmp_path / "law.txt"
    path.write_text(SOURCE, encoding="utf-8", newline="\n")
    config = metadata_config(tmp_path, min_article_chars=20)
    override = tmp_path / "overrides.json"
    write_json(override, {path.name: {"tier": "statutory_law"}})
    config.manual_overrides_path = override
    synthesize([path], config, adapter=DemoGeneration())
    qa = config.stage_output_dir / "synthesis/qa.jsonl"
    original = qa.read_bytes()
    discover_metadata([path], config, dry_run=True)
    synthesize([path], config, adapter=DemoGeneration())
    assert qa.read_bytes() == original
    discover_metadata([path], config)
    assert qa.read_bytes() == original
    config.synthesis_metadata_enabled = True
    synthesize([path], config, adapter=DemoGeneration())
    record = read_jsonl(qa)[0]
    added = record["provenance"].pop("discovered_metadata")
    assert added["shadow_mode"] and added["schema_version"] == "1.0"
    import json
    assert record == json.loads(original)


def test_metadata_evaluation_synthetic_annotations(tmp_path):
    from corpus_pipeline.benchmark import evaluate_metadata
    source = tmp_path / "research.txt"
    source.write_text("Example University", encoding="utf-8", newline="\n")
    config = metadata_config(tmp_path)
    discover_metadata([source], config, adapter=FakeMetadata())
    path = config.stage_output_dir / "metadata/documents.jsonl"
    document = read_jsonl(path)[0]
    references = tmp_path / "annotations.jsonl"
    write_jsonl(references, [{"document_id": document["document_id"], "source_text": "Example University",
                            "observations": [{"label": row["label"], "value": row["value"],
                                              "expected_field_id": row["field_id"]} for row in document["metadata"]],
                            "chunk_boundaries": []}])
    result = evaluate_metadata(path, references, tmp_path / "evaluation.json")
    assert result["precision"] == result["recall"] == result["evidence_support_rate"] == 1
    assert result["source_reconstruction_rate"] == 1 and result["field_match_error_rate"] == 0
    assert not any(field["quality_eligible"] for field in result["fields"].values())
    assert not result["registry_modified"]


@pytest.mark.parametrize("text,label,value", [
    ("Université Exemple : mémoire en biologie.", "université", "Université Exemple"),
    ("Event location: Example Hall.", "event location", "Example Hall"),
    ("جامعة المثال تقدم تقريرا تقنيا.", "الجامعة", "جامعة المثال"),
])
def test_metadata_multifamily_manual_smoke_annotations(tmp_path, text, label, value):
    from corpus_pipeline.benchmark import evaluate_metadata
    source = tmp_path / "synthetic.txt"
    source.write_text(text, encoding="utf-8", newline="\n")

    class Fake:
        def generate(self, phase, prompt):
            return {"observations": [observation(label, value, evidence=[{"quote": value}])]}

    config = metadata_config(tmp_path)
    discover_metadata([source], config, adapter=Fake())
    references = tmp_path / "reference.jsonl"
    write_jsonl(references, [{"source": str(source), "source_text": text,
                             "observations": [{"label": label, "value": value}], "chunk_boundaries": []}])
    result = evaluate_metadata(config.stage_output_dir / "metadata/documents.jsonl", references, tmp_path / "evaluation.json")
    assert result["precision"] == result["recall"] == result["source_reconstruction_rate"] == 1


def test_chonkie_pinned_constructor_and_delimiter_policy(monkeypatch):
    import sys
    import corpus_pipeline.metadata.chunking as module
    seen = {}

    def constructor(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "chonkie", SimpleNamespace(SemanticChunker=constructor))
    monkeypatch.setattr(module, "version", lambda _: "1.5.2")
    config = PipelineConfig(metadata_embedding_model="FAKE_LOCAL_MODEL")
    module._semantic(config)
    assert seen["skip_window"] == 0 and seen["embedding_model"] == "FAKE_LOCAL_MODEL"
    assert {"؟", "؛"} <= set(seen["delim"])
    assert seen["include_delim"] == "prev" and seen["similarity_window"] == 3
    monkeypatch.setattr(module, "version", lambda _: "different")
    with pytest.raises(ValueError, match="version"):
        module._semantic(config)


def test_metadata_failure_envelope_recovers_and_cache_corruption_regenerates(tmp_path):
    path = tmp_path / "source.txt"
    config = metadata_config(tmp_path)
    failed = discover_metadata([path], config)
    assert failed["failed"] == 1
    docs_path = config.stage_output_dir / "metadata/documents.jsonl"
    assert read_jsonl(docs_path)[0]["metadata_status"] == "failed"
    path.write_text("Example University", encoding="utf-8")
    discover_metadata([path], config, adapter=FakeMetadata())
    assert len(read_jsonl(docs_path)) == 1
    assert read_jsonl(docs_path)[0]["metadata_status"] == "complete"
    cache = next((config.stage_output_dir / "metadata/cache").glob("*.json"))
    cache.write_text("{broken", encoding="utf-8")
    fake = FakeMetadata()
    result = discover_metadata([path], config, adapter=fake)
    assert result["model_calls"] == 1 and fake.calls


def test_legal_invalid_calendar_value_is_audited_and_missing_article_detected():
    result = legal_extract("قانون رقم 26-90\nالمادة 1 : نص\nحرر في 2026-02-30")
    invalid = [obs for obs in result["observations"] if obs.value_type == "date"]
    assert invalid and invalid[0].status == "rejected"
    result = legal_extract("قانون رقم 26-90\nالمادة 1 : نص\n## المادة غير واضحة\nباقي النص")
    assert result["status"] == "needs_fallback"


def test_registry_uncertainty_quality_approval_and_invalid_merge(tmp_path):
    registry = FieldRegistry(tmp_path, PipelineConfig())
    field_id = registry.match(verified(), "doc").field_id
    with pytest.raises(ValueError):
        registry.migrate("merge", field_id, target_id=field_id, reason="invalid")
    registry.migrate("quality_approval", field_id, reason="Operator evaluated independent evidence")
    assert registry.entries[field_id]["quality_approved"]
    registry.migrate("uncertainty", field_id, reason="Later evidence contradicts semantics")
    assert registry.entries[field_id]["status"] == "uncertain"
    assert not registry.entries[field_id]["quality_approved"]


def test_metadata_config_environment_and_illegal_caps(monkeypatch):
    monkeypatch.setenv("METADATA_MAX_CALLS", "7")
    monkeypatch.setenv("METADATA_MODEL", "FAKE_MODEL")
    monkeypatch.setenv("METADATA_DELIMITERS", '["؟", "؛"]')
    config = PipelineConfig()
    assert config.metadata_max_calls == 7 and config.metadata_model == "FAKE_MODEL"
    assert config.metadata_delimiters == ["؟", "؛"]
    config.metadata_max_calls = 0
    with pytest.raises(ValueError, match="positive"):
        config.validate_metadata()


def test_metadata_source_with_configured_key_is_withheld(tmp_path):
    secret = "credential-in-source-test"
    source = tmp_path / "source.txt"
    source.write_text("قانون رقم 26-90\nالمادة 1 : " + secret, encoding="utf-8")
    config = metadata_config(tmp_path, ocr_api_key=secret, session_api_key_override=True)
    summary = discover_metadata([source], config)
    assert summary["failed"] == 1 and summary["model_calls"] == 0
    for artifact in (config.stage_output_dir / "metadata").glob("*"):
        if artifact.is_file():
            assert secret not in artifact.read_text(encoding="utf-8")


def test_metadata_evaluation_cli_missing_predictions_do_not_score_as_exact(tmp_path, monkeypatch):
    import cli
    output = tmp_path / "stages"
    write_jsonl(output / "metadata/documents.jsonl", [])
    references = tmp_path / "reference.jsonl"
    write_jsonl(references, [{"document_id": "missing", "observations": []}])
    monkeypatch.setattr("sys.argv", ["cli.py", "--metadata-evaluate", str(references), "--stage-output", str(output)])
    assert cli.main() == 0
    evaluation = load_json(output / "metadata/evaluation.json")
    assert evaluation["exact_document_rate"] == 0
    assert evaluation["source_reconstruction_rate"] == 0
