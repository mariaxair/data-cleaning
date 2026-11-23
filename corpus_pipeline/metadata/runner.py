"""Shadow-mode metadata artifacts with bounded, cached model extraction."""

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

from config import PipelineConfig
from ..artifacts import fingerprint, read_jsonl, write_json, write_jsonl
from ..governance.language import assess_language
from ..models import METADATA_SCHEMA_VERSION, MetadataDocument, MetadataObservation
from ..ocr.base import redact_secrets
from ..settings import load_json
from ..synthesis.client import CompatibleGenerationAdapter, GenerationAdapter
from .chunking import chunk_document, recursive_spans
from .extraction import PROMPT_VERSION, metadata_prompt, validate_observations
from .legal import legal_extract
from .registry import FieldRegistry, normalize_label


def _safe(value, config: PipelineConfig):
    """Never persist credentials, including adapter-provided/raw audit strings."""
    if isinstance(value, str):
        for secret in {config.effective_api_key(), config.ocr_api_key} - {""}:
            value = value.replace(secret, "***")
        return redact_secrets(value)
    if isinstance(value, dict):
        return {_safe(str(key), config): _safe(item, config) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item, config) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _language(text: str) -> dict:
    assessment = assess_language(text, "", 0.2)
    detected = assessment["detected"]
    total = sum(assessment["counts"].values())
    if detected == "ar":
        return {"value": "ar", "method": "script_detector", "confidence": assessment["counts"]["arabic"] / max(1, total)}
    return {"value": "unknown", "method": "script_detector", "confidence": None,
            "reason": f"Script result {detected} does not reliably identify a language."}


def _windows(text: str, chunks: list[dict], config: PipelineConfig) -> list[dict]:
    """Long legal fallback articles retain structure while requests stay bounded."""
    result = []
    for chunk in chunks:
        for start, end in recursive_spans(chunk["text"], config.metadata_chunk_chars, config.metadata_chunk_tokens):
            result.append({**chunk, "start": chunk["start"] + start, "end": chunk["start"] + end})
    return result


def _prepare(path: Path, config: PipelineConfig, dry_run: bool, chunker=None) -> tuple:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if (redact_secrets(text) != text or
            any(secret and secret in text for secret in (config.effective_api_key(), config.ocr_api_key))):
        raise ValueError("Source contains a configured credential; extraction was withheld.")
    source_hash = hashlib.sha256(raw).hexdigest()
    document = MetadataDocument(source_hash, str(path), source_hash, language=_language(text))
    report_path = path.with_suffix(".report.json")
    report = load_json(report_path)
    legal = legal_extract(text)
    chunks, info = chunk_document(text, config, legal_spans=legal["spans"] if legal["likely"] else None,
                                  dry_run=dry_run, chunker=chunker)
    for chunk in chunks:
        chunk.update(chunk_id=fingerprint([source_hash, chunk["start"], chunk["end"]]), document_id=source_hash,
                     parent_id=source_hash, metadata_inheritance="document_reference")
    document.chunks = chunks
    document.document_type = "structured_legal" if legal["likely"] else "unknown"
    document.processing = {**info, "route": "legal" if legal["status"] == "complete" else
                          "legal_fallback" if legal["likely"] else "generic", "legal_status": legal["status"],
                          "cleaned_source": str(path), "cleaned_sha256": source_hash,
                          "original_source": report.get("source"), "original_source_sha256": report.get("source_sha256"),
                          "page_map": [{"start": c["start"], "end": c["end"], "pages": c["pages"]} for c in chunks],
                          "provider": config.ocr_provider, "model": config.metadata_model,
                          "prompt_version": PROMPT_VERSION, "temperature": 0,
                          "max_output_tokens": config.metadata_max_tokens, "response_fingerprints": [],
                          "shadow_mode": True, "source_partial": bool(report.get("failed_pages") or report.get("status") == "partial")}
    if info["fallback_reason"]:
        document.warnings.append(info["fallback_reason"])
    if legal["likely"] and legal["status"] != "complete":
        document.warnings.extend(legal["warnings"])
    damaged = "\ufffd" in text or any("\ue000" <= c <= "\uf8ff" for c in text)
    document.processing["source_damaged"] = damaged
    if damaged or document.processing["source_partial"]:
        document.warnings.append("Damaged or partial source requires review; model extraction is deferred.")
    # Chunk boundaries can be structurally legal while metadata remains open-ended:
    # let the configured metadata model inspect all readable documents, including laws.
    windows = _windows(text, chunks, config) if not damaged and not document.processing["source_partial"] else []
    return document, text, legal, windows


def discover_metadata(paths: list[Path], config: PipelineConfig, *, dry_run: bool = False,
                      adapter: GenerationAdapter | None = None, chunker=None,
                      progress: Callable[[str], None] | None = None) -> dict:
    """Extract to a dedicated area; dry runs only write their plan/summary."""
    config.validate_metadata()
    if any(path.suffix.lower() not in {".txt", ".md"} for path in paths):
        raise ValueError("Metadata consumes cleaned TXT/MD; run extraction first.")
    output = config.stage_output_dir / "metadata"
    prepared, failed = [], []
    for path in sorted(set(paths), key=lambda item: str(item)):
        try:
            prepared.append(_prepare(path, config, dry_run, chunker))
        except Exception as error:
            credential_reason = "Source contains a configured credential; extraction was withheld."
            reason = credential_reason if isinstance(error, ValueError) and str(error) == credential_reason else (
                "Source/report could not be read or chunked; check UTF-8, JSON, and file permissions.")
            failed.append({"source": _safe(str(path), config), "status": "failed",
                           "document_id": fingerprint(["unreadable", str(path)]), "source_sha256": "",
                           "reason": reason})
    plan = {"schema_version": METADATA_SCHEMA_VERSION, "dry_run": dry_run, "shadow_mode": True,
            "max_calls": config.metadata_max_calls,
            "documents": [{"source": doc.source, "document_id": doc.document_id,
                           "route": doc.processing["route"], "chunks": len(doc.chunks),
                           "chunker": doc.processing["chunker"], "degraded": doc.processing["degraded"],
                           "fallback_reason": doc.processing["fallback_reason"], "warnings": doc.warnings,
                           "estimated_requests": len(windows),
                            "legal_structure_spans": len(legal["spans"])} for doc, _, legal, windows in prepared],
            "estimated_requests": sum(len(windows) for _, _, _, windows in prepared), "failed": failed}
    write_json(output / "plan.json", _safe(plan, config))
    if dry_run:
        summary = {**plan, "model_calls": 0, "note": "No provider calls or embeddings initialized; cached extraction is not rewritten."}
        write_json(output / "dry-run-summary.json", _safe(summary, config))
        return summary
    live = config.metadata_llm_enabled or adapter is not None
    if live and plan["estimated_requests"] and adapter is None:
        if not config.metadata_model.strip():
            raise ValueError("Set METADATA_MODEL explicitly before live metadata extraction (or use --dry-run).")
        adapter = CompatibleGenerationAdapter(replace(config, synthesis_model=config.metadata_model,
                                                       synthesis_max_tokens=config.metadata_max_tokens))
    registry = FieldRegistry(output, config)
    documents_path, audit_path = output / "documents.jsonl", output / "observations.jsonl"
    documents = {row["document_id"]: row for row in read_jsonl(documents_path)} if documents_path.is_file() else {}
    for failure in failed:
        documents[failure["document_id"]] = MetadataDocument(
            failure["document_id"], failure["source"], failure["source_sha256"], metadata_status="failed",
            warnings=[failure["reason"]], processing={"shadow_mode": True, "hash_reason": "Source could not be read."}).to_dict()
    old_audit = read_jsonl(audit_path) if audit_path.is_file() else []
    audits = {row["observation_id"]: row for row in old_audit}
    model_calls, cache_hits, processed = 0, 0, []
    start_run = time.perf_counter()
    for document, text, legal, windows in prepared:
        started = time.perf_counter()
        document_id = document.document_id
        documents = {key: row for key, row in documents.items() if not (
            row.get("source") == document.source and row.get("metadata_status") == "failed")}
        if progress:
            progress(f"Metadata: {Path(document.source).name} ({document.processing['route']})")
        # Replace this document's audit rows only; unrelated documents survive incremental runs.
        audits = {key: row for key, row in audits.items() if row["document_id"] != document_id}
        # Legal syntax is used to recognize documents and preserve their structure,
        # but content metadata is owned by the open-ended metadata extractor.
        observations = []
        pending, errors = 0, 0
        for window in windows:
            prompt = metadata_prompt(text, window, context=text[:min(1000, config.metadata_chunk_chars)])
            cache_id = fingerprint([METADATA_SCHEMA_VERSION, PROMPT_VERSION, prompt, config.ocr_provider,
                                    config.ocr_base_url, config.metadata_model, config.metadata_max_tokens, 0])
            cache_path = output / "cache" / f"{cache_id}.json"
            response, reason = None, "Model extraction disabled; enable METADATA_LLM_ENABLED with an explicit model."
            try:
                cached = load_json(cache_path)
                if cached.get("cache_id") == cache_id:
                    validate_observations(cached["response"], text, window["start"], window["end"])
                    response = cached["response"]
                    cache_hits += 1
            except (ValueError, OSError, KeyError, TypeError):
                pass
            if response is None and live:
                for _ in range(config.metadata_max_attempts):
                    if model_calls >= config.metadata_max_calls:
                        reason = "Metadata model-call budget exhausted; rerun to resume from cached windows."
                        break
                    model_calls += 1
                    try:
                        generated = adapter.generate("metadata", prompt)
                        validated = validate_observations(generated, text, window["start"], window["end"])
                        response = _safe(generated, config)
                        if response != generated:
                            reason = "Unsafe/invalid model output was withheld."
                            response = None
                            continue
                        # Invalid evidence should be retried on a future run, not cached as success.
                        if all(obs.status != "rejected" for obs in validated):
                            write_json(cache_path, {"cache_id": cache_id, "response": response})
                        break
                    except Exception:
                        reason = f"Metadata request failed for provider {config.ocr_provider}; check settings or retry."
            if response is None:
                pending += 1
                observations.append((MetadataObservation("extraction_pending", None, reason=reason,
                                                        chunk_id=window["chunk_id"]), {"request_id": cache_id}))
                continue
            response_hash = fingerprint(response)
            document.processing["response_fingerprints"].append(response_hash)
            validated = validate_observations(response, text, window["start"], window["end"])
            for index, obs in enumerate(validated):
                obs.chunk_id = window["chunk_id"]
                observations.append((obs, {"request_id": cache_id, "response_fingerprint": response_hash,
                                          "raw_observation": response["observations"][index]}))
        canonical = {}
        for obs, raw in observations:
            if obs.confidence < config.metadata_min_confidence and obs.status == "verified":
                obs.status, obs.reason = "uncertain", "Below configured confidence threshold."
            obs = registry.match(obs, document_id)
            if obs.status != "verified" or obs.mapping_status == "uncertain":
                errors += 1
            else:
                key = fingerprint([obs.field_id, obs.value, obs.scope, obs.role, obs.chunk_id if obs.scope == "chunk" else None])
                if key in canonical:
                    current = canonical[key]
                    for evidence in obs.evidence:
                        if evidence not in current.evidence:
                            current.evidence.append(evidence)
                else:
                    canonical[key] = obs
            row = {"document_id": document_id, "source": document.source, "schema_version": METADATA_SCHEMA_VERSION,
                   "observation": obs.to_dict(), **raw}
            row["observation_id"] = fingerprint([document_id, row])
            audits[row["observation_id"]] = _safe(row, config)
        values = list(canonical.values())
        document.metadata = [obs for obs in values if obs.scope == "document"]
        document.dates = [obs for obs in document.metadata if obs.value_type in {"date", "date_range"}]
        domain_obs = [obs for obs in document.metadata if obs.value_type == "domain"]
        document.domains = [{"value": normalize_label(obs.value), "original_label": obs.value,
                             "domain_id": fingerprint(["domain", normalize_label(obs.value)])[:24],
                             "method": obs.method, "confidence": obs.confidence,
                             "evidence": [e.__dict__ for e in obs.evidence]} for obs in domain_obs]
        if not document.domains:
            document.domains = [{"value": "unknown", "original_label": None, "domain_id": None,
                                 "method": "none", "confidence": None, "reason": "No reliable domain observation."}]
        languages = [obs for obs in document.metadata if obs.value_type == "language"]
        if languages:
            agreed = {obs.value for obs in languages}
            if len(agreed) == 1:
                document.language = {"value": languages[0].value, "method": "llm_open_extraction",
                                     "confidence": min(obs.confidence for obs in languages),
                                     "evidence": [e.__dict__ for obs in languages for e in obs.evidence]}
            else:
                document.language = {"value": "unknown", "method": "llm_open_extraction",
                                     "confidence": None, "reason": "Conflicting language observations; review required."}
                errors += 1
        for chunk in document.chunks:
            chunk["inherited_document_metadata_id"] = document_id
            chunk["metadata"] = [obs.to_dict() for obs in values if obs.scope == "chunk" and (
                obs.chunk_id == chunk["chunk_id"] or any(chunk["start"] <= e.start < chunk["end"] for e in obs.evidence))]
        document.registry_version = registry.version
        document.metadata_status = ("needs_fallback" if legal["likely"] and legal["status"] != "complete" and pending else
                                    "extraction_pending" if pending else "partial" if errors or document.processing["source_partial"]
                                    or document.processing["source_damaged"] else "complete")
        if not text.strip():
            document.metadata_status = "extraction_pending"
            document.warnings.append("Empty source; no extraction evidence is available.")
        document.processing.update(elapsed_s=time.perf_counter() - started, pending_windows=pending,
                                   unresolved_observations=errors)
        documents[document_id] = _safe(document.to_dict(), config)
        processed.append(document_id)
    registry.save()
    write_jsonl(documents_path, [documents[key] for key in sorted(documents)])
    rows = [audits[key] for key in sorted(audits)]
    write_jsonl(audit_path, rows)
    queue = [row for row in rows if row["observation"]["status"] != "verified" or
             row["observation"]["mapping_status"] == "uncertain"] + failed
    for document in documents.values():
        processing = document.get("processing", {})
        if (processing.get("degraded") or processing.get("source_damaged") or processing.get("source_partial") or
                processing.get("route") == "legal_fallback"):
            queue.append({"document_id": document["document_id"], "source": document["source"],
                          "status": "review", "reason": " ".join(document.get("warnings", [])) or
                          "Source/layout/chunk boundaries need quality review."})
    write_jsonl(output / "review-queue.jsonl", queue)
    current = [documents[key] for key in dict.fromkeys(processed)]
    summary = {"schema_version": METADATA_SCHEMA_VERSION, "shadow_mode": True,
               "processed": len(current), "failed": len(failed), "failure_details": failed,
               "corpus_documents": len(documents), "model_calls": model_calls, "cache_hits": cache_hits,
               "max_calls": config.metadata_max_calls, "max_http_attempts_per_model_call": config.ocr_max_retries + 1,
               "elapsed_s": time.perf_counter() - start_run, "provider": config.ocr_provider,
               "model": config.metadata_model, "registry_version": registry.version,
               "routes": dict(Counter(d["processing"]["route"] for d in current)),
               "statuses": dict(Counter(d["metadata_status"] for d in current)),
               "chunkers": dict(Counter(d["processing"]["chunker"] for d in current)),
               "degraded_documents": sum(d["processing"]["degraded"] for d in current),
               "review_cases": len(queue), "fields": dict(Counter(row["observation"].get("field_id") or "unmapped" for row in rows)),
               "confidence_bands": dict(Counter("high" if row["observation"]["confidence"] >= 0.9 else "low" for row in rows)),
               "failure_reasons": dict(Counter(row.get("observation", {}).get("reason", row.get("reason")) for row in queue)),
               "cost": None, "cost_reason": "Provider pricing/token usage unavailable; model-call count is recorded."}
    write_json(output / "summary.json", _safe(summary, config))
    return summary


def metadata_provenance(source_hash: str, config: PipelineConfig) -> dict:
    """Add references only; never inject discovered field values into training text."""
    path = config.stage_output_dir / "metadata" / "documents.jsonl"
    if not path.is_file():
        return {"status": "missing", "cleaned_sha256": source_hash}
    record = next((row for row in read_jsonl(path) if row.get("source_sha256") == source_hash), None)
    if record is None:
        return {"status": "missing", "cleaned_sha256": source_hash}
    return {"document_id": record["document_id"], "schema_version": record["metadata_schema_version"],
            "registry_version": record["registry_version"], "status": record["metadata_status"],
            "artifact": str(path), "shadow_mode": True}


def export_synthesis_chunks(config: PipelineConfig, sources: list[Path], *, metadata_root: Path | None = None) -> dict:
    """Export source slices with open-ended metadata, not a hard-coded legal schema."""
    import uuid

    metadata_dir = (metadata_root or config.stage_output_dir) / "metadata"
    documents = read_jsonl(metadata_dir / "documents.jsonl")
    selected = {str(path.resolve()) for path in sources}
    rows = []
    for document in documents:
        if str(Path(document["source"]).resolve()) not in selected:
            continue
        raw = Path(document["source"]).read_bytes()
        source_text = raw.decode("utf-8-sig")
        if hashlib.sha256(raw).hexdigest() != document["source_sha256"]:
            continue  # A failed rerun must never export an old version of a changed source.
        for chunk in document["chunks"]:
            language = document["language"].get("value", "unknown")
            discovered = list(document.get("metadata", []))
            discovered.extend(chunk.get("metadata", []))
            row = {"text": chunk["text"], "lang": language, "source": Path(document["source"]).name,
                   "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                   "metadata": discovered,
                   "document_type": document.get("document_type", "unknown"),
                   "provenance": {"source_file": document["processing"].get("original_source") or document["source"],
                                   "source_sha256": document["processing"].get("original_source_sha256") or document["source_sha256"],
                                   "cleaned_source_file": document["source"], "cleaned_sha256": document["source_sha256"],
                                   "document_id": document["document_id"], "unit_id": chunk["chunk_id"],
                                   "char_start": chunk["start"], "char_end": chunk["end"], "pages": chunk["pages"],
                                   "discovered_metadata": {"artifact": str(metadata_dir / "documents.jsonl"),
                                   "status": document["metadata_status"]}},
                   "article_numbers": chunk.get("normalized_article_numbers", chunk.get("article_numbers", [])),
                   "level": chunk.get("level", "passage"),
                   "source_partial": document["processing"].get("source_partial", False),
                   "source_damaged": document["processing"].get("source_damaged", False),
                   "chunking": {k: document["processing"].get(k) for k in ("chunker", "degraded", "fallback_reason")}}
            rows.append(row)
    output = config.stage_output_dir / "synthesis" / "chunking"
    write_jsonl(output / "chunks.jsonl", rows)
    summary = {"chunks": len(rows), "documents": len(selected), "output": str(output / "chunks.jsonl")}
    write_json(output / "summary.json", summary)
    return summary
