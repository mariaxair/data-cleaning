"""Bounded, resumable three-phase synthesis with a no-network dry run."""

import json
import hashlib
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from typing import Callable

from config import PipelineConfig
from ..artifacts import fingerprint, write_json, write_jsonl
from ..settings import load_json
from .client import CompatibleGenerationAdapter, GenerationAdapter
from .screening import screen_document
from .segmentation import SynthesisUnit, segment_document
from .validation import validate_generation

PROMPTS = Path(__file__).parent / "prompts"
PHASES = ("question", "answer", "explanation")
REASONING_TYPES = ("conditional_application", "exception_analysis", "procedural_tracing", "deadline_calculation")
GENERAL_TASKS = ("concept_explanation", "comparison", "practical_application", "procedural_explanation")


def prompt_templates(*, general: bool = False) -> dict[str, Template]:
    """Read and validate every prompt even when no units are eligible."""
    result = {}
    for phase in PHASES:
        filename = f"general_{phase}.txt" if general else f"{phase}.txt"
        template = Template((PROMPTS / filename).read_text(encoding="utf-8"))
        if not template.is_valid():
            raise ValueError(f"Invalid {phase} prompt template.")
        result[phase] = template
    return result


def generate_unit(unit: SynthesisUnit, config: PipelineConfig, adapter: GenerationAdapter,
                  templates: dict[str, Template], index: int, checkpoint_dir: Path,
                  run_fingerprint: str) -> dict:
    """Checkpoint valid phase outputs; retry failed validation without losing earlier phases."""
    identity = fingerprint([unit.id, run_fingerprint, index % len(REASONING_TYPES), index % 2])
    checkpoint = checkpoint_dir / f"{identity}.json"
    previous = load_json(checkpoint)
    responses = previous.get("responses", {})
    validations = {}
    tasks = GENERAL_TASKS if unit.tier == "general" else REASONING_TYPES
    fields = {"language": config.synthesis_language, "source": unit.text,
              "perspective": ("first_person", "third_person_generic_roles")[index % 2],
              "reasoning_type": tasks[index % len(tasks)],
              "question": "", "answer": ""}
    for phase in PHASES:
        errors = []
        for attempt in range(config.synthesis_max_attempts):
            response = responses.get(phase)
            if response is None:
                # Explanation template has no source placeholder, so raw source is never transmitted.
                prompt = templates[phase].substitute(fields)
                if errors:
                    prompt += "\nCorrect validation failures: " + ", ".join(errors)
                response = adapter.generate(phase, prompt)
            validation = validate_generation(
                phase, response, unit, config.synthesis_language,
                question=fields["question"], answer=fields["answer"])
            if validation.passed:
                response = {key: response[key] for key in ("text", "evidence") if key in response}
                responses[phase] = response
                validations[phase] = asdict(validation)
                if phase in fields:
                    fields[phase] = response["text"]
                write_json(checkpoint, {"responses": responses})
                break
            errors = validation.errors
            responses.pop(phase, None)
        else:
            return {"id": identity, "status": "validation_failed", "phase": phase,
                    "errors": errors, "source_file": unit.source_file}
    metadata = {
        "source_file": unit.source_document or unit.source_file,
        "source_sha256": unit.source_document_sha256 or unit.source_sha256,
        "cleaned_source_file": unit.source_file, "cleaned_sha256": unit.source_sha256,
        "unit_id": unit.id,
        "law_title": unit.law_title, "law_number": unit.law_number,
        "article_numbers": unit.article_numbers, "pages": unit.pages, "category": unit.category,
        "tier": unit.tier,
    }
    if unit.synthesis_mode == "automatic":
        metadata.update(topic=unit.topic, topic_group_id=unit.topic_group_id,
                        char_start=unit.char_start, char_end=unit.char_end)
    return {
        "id": identity, "status": "pending_review", "question": responses["question"]["text"],
        "answer": responses["answer"]["text"], "reasoning_chain": responses["explanation"]["text"],
        "native_chain_of_thought": None, "rationale_kind": "public_explanation",
        "source_text": unit.text, "provenance": metadata,
        "sensitive_values": unit.sensitive_values,
        "evidence": {phase: responses[phase].get("evidence", []) for phase in ("question", "answer")},
        "language": config.synthesis_language, "question_type": fields["perspective"],
        "reasoning_type": fields["reasoning_type"], "scores": {}, "validations": validations,
        "pending_checks": ["semantic_grounding", "personal_names_review", "dialect_screening", "quality_review"],
        "provider": config.ocr_provider, "model": config.synthesis_model,
    }


def synthesize(paths: list[Path], config: PipelineConfig, *, dry_run: bool = False,
               adapter: GenerationAdapter | None = None, max_samples: int | None = None,
               topics_only: bool = False, progress: Callable[[str], None] | None = None) -> dict:
    """Write a plan, candidates, failures, and summary without modifying legacy exports."""
    config.validate_training()
    limit = config.synthesis_max_samples if max_samples is None else max_samples
    if limit < 1:
        raise ValueError("--max-samples must be positive.")
    templates = prompt_templates()
    general_templates = prompt_templates(general=True)
    automatic = config.synthesis_screening_mode == "automatic"
    if topics_only and not automatic:
        raise ValueError("Topic discovery requires automatic screening mode.")
    if any(path.suffix.lower() not in {".txt", ".md"} for path in paths):
        raise ValueError("Synthesis consumes cleaned TXT/MD; run extraction first.")
    # Validate live credentials/model before processing a batch. Fakes and dry runs need neither.
    if not dry_run and adapter is None:
        adapter = CompatibleGenerationAdapter(config)
    policy, overrides = load_json(config.screening_path), load_json(config.manual_overrides_path)
    discovery = None
    topic_screens = {}
    if automatic:
        from .topics import TopicDiscovery, topic_screening
        automatic_paths = [path for path in paths if not overrides.get(
            str(path.resolve()), overrides.get(path.name))]
        discovery = TopicDiscovery(config, adapter, dry_run, progress).discover(automatic_paths)
        topic_screens = {doc["source_file"]: topic_screening(doc) for doc in discovery["documents"]}
        write_json(config.stage_output_dir / "synthesis" / "topics.json", discovery)
    units: list[SynthesisUnit] = []
    documents = []
    for path in paths:
        if path.suffix.lower() not in {".txt", ".md"}:
            raise ValueError("Synthesis consumes cleaned TXT/MD; run extraction first.")
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        overridden = bool(overrides.get(str(path.resolve()), overrides.get(path.name)))
        screening = (topic_screens[str(path)] if automatic and not overridden
                     else screen_document(text, path, policy, overrides))
        source_hash = hashlib.sha256(raw).hexdigest()
        segmented = segment_document(text, path, screening, group_size=config.article_group_size,
                                     min_chars=config.min_article_chars, policy=policy, source_hash=source_hash,
                                     general_chunk_chars=config.general_chunk_chars)
        fallback_warning = []
        if automatic and not overridden and not segmented and screening.tier in {"statutory_law", "case_law"}:
            # A thematic classifier must never make article headings a prerequisite for general text.
            screening.tier = "general"
            segmented = segment_document(text, path, screening, min_chars=config.min_article_chars,
                                         source_hash=source_hash, general_chunk_chars=config.general_chunk_chars)
            fallback_warning.append("No usable legal segments; using general passages.")
        if automatic:
            for unit in segmented:
                unit.synthesis_mode = "automatic"
                unit.topic = screening.metadata.get("topic", "")
                unit.topic_group_id = screening.metadata.get("topic_group_id")
        extraction_report = load_json(path.with_suffix(".report.json"))
        for unit in segmented:
            unit.source_document = extraction_report.get("source", "")
            unit.source_document_sha256 = extraction_report.get("source_sha256", "")
        # Unvalidated keyword-only decisions require manual adjudication, except exclusion.
        partial = bool(extraction_report.get("failed_pages")) or extraction_report.get("status") == "partial"
        allowed = not screening.degraded or (automatic and screening.tier == "general")
        eligible = segmented if allowed and screening.tier not in {"administrative_noise", "manual"} and not partial else []
        units.extend(eligible)
        documents.append({"source_file": str(path), "bytes": len(raw), "screening": asdict(screening),
                          "eligible_units": len(eligible), "segmented_units": len(segmented),
                          "warnings": ((["Unvalidated screening: manual adjudication required."]
                                       if screening.degraded and not automatic else []) +
                                       screening.metadata.get("topic_warnings", []) + fallback_warning +
                                       (["Partial extraction: repair failed pages before synthesis."] if partial else []))})
    selected = units[:limit]
    output = config.stage_output_dir / "synthesis"
    identity_settings = {
        "version": 1, "prompts": {k: v.template for k, v in templates.items()},
        "model": config.synthesis_model, "provider": config.ocr_provider,
        "endpoint": config.ocr_base_url, "language": config.synthesis_language,
        "tokens": config.synthesis_max_tokens,
    }
    if automatic:
        identity_settings["general_prompts"] = {k: v.template for k, v in general_templates.items()}
    identity = fingerprint(identity_settings)
    plan = {"dry_run": dry_run, "documents": documents, "eligible_units": len(units),
            "selected_units": len(selected), "max_samples": limit,
            "max_generation_requests": len(selected) * 3 * config.synthesis_max_attempts,
            "units": [{k: v for k, v in asdict(u).items() if k != "sensitive_values"} for u in selected]}
    if automatic:
        plan.update(screening_mode="automatic", topic_groups=len(discovery["groups"]),
                    topic_model_calls=discovery["model_calls"], topics_only=topics_only,
                    topic_max_calls=config.topic_max_calls,
                    topic_pending_documents=sum(doc["status"] != "discovered" for doc in discovery["documents"]),
                    topic_ungrouped_documents=sum(doc["grouping_status"] != "grouped" for doc in discovery["documents"]))
    write_json(output / "plan.json", plan)
    if dry_run or topics_only:
        write_json(output / ("dry-run-summary.json" if dry_run else "topics-summary.json"), plan)
        return plan

    def run(item: tuple[int, SynthesisUnit]) -> dict:
        index, unit = item
        try:
            selected_templates = general_templates if unit.tier == "general" else templates
            return generate_unit(unit, config, adapter, selected_templates, index, output / "checkpoints", identity)
        except Exception:
            # Provider errors or custom adapter exceptions may contain credentials: never serialize them.
            return {"id": fingerprint([unit.id, identity]), "status": "generation_failed",
                    "source_file": unit.source_file, "errors": [
                        f"Generation failed for provider {config.ocr_provider}; check settings or retry."]}

    with ThreadPoolExecutor(max_workers=config.synthesis_concurrency, thread_name_prefix="synthesis") as pool:
        results = list(pool.map(run, enumerate(selected)))
    candidates = [record for record in results if record["status"] == "pending_review"]
    failures = [record for record in results if record["status"] != "pending_review"]
    write_jsonl(output / "qa.jsonl", candidates)
    write_jsonl(output / "failed.jsonl", failures)
    summary = {"generated": len(candidates), "failed": len(failures), "selected": len(selected),
               "provider": config.ocr_provider, "model": config.synthesis_model,
               "degraded": any(doc["screening"]["degraded"] for doc in documents),
               "manual_documents": sum(doc["screening"]["tier"] == "manual" or doc["screening"]["degraded"]
                                       for doc in documents)}
    if automatic:
        summary.update(screening_mode="automatic", topic_groups=len(discovery["groups"]),
                       topic_model_calls=discovery["model_calls"],
                       topic_pending_documents=sum(doc["status"] != "discovered" for doc in discovery["documents"]),
                       topic_ungrouped_documents=sum(doc["grouping_status"] != "grouped" for doc in discovery["documents"]))
    write_json(output / "summary.json", summary)
    return summary
