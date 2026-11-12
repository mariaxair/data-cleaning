"""Open-topic model discovery and semantic grouping with no supplied label list."""

import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from typing import Callable

from config import PipelineConfig
from ..artifacts import fingerprint, write_json
from ..settings import load_json
from .client import GenerationAdapter
from .screening import ScreeningResult

PROMPTS = Path(__file__).parent / "prompts"


def sample_document(text: str, limit: int) -> str:
    """Sample distributed excerpts so classification is not based only on the abstract."""
    if len(text) <= limit:
        return text
    size = max(1, limit // 3)
    middle = max(size, (len(text) - size) // 2)
    return "\n[...]\n".join((text[:size], text[middle:middle + size], text[-size:]))


def _description(response: dict) -> dict:
    """Discard unrelated model fields; require descriptive, bounded output."""
    if not isinstance(response, dict):
        raise ValueError("Expected a topic object.")
    values = {}
    for key, limit in (("topic", 200), ("summary", 1200)):
        value = response.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError("Invalid topic description.")
        values[key] = value.strip()
    structure = response.get("structure", "general")
    if structure not in {"general", "statutory_law", "case_law"}:
        raise ValueError("Invalid structural route.")
    confidence = response.get("confidence")
    if type(confidence) not in {int, float} or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid model confidence.")
    return {**values, "structure": structure, "confidence": confidence}


class TopicDiscovery:
    """One run's bounded model budget and content-addressed response cache."""

    def __init__(self, config: PipelineConfig, adapter: GenerationAdapter | None, dry_run: bool,
                 progress: Callable[[str], None] | None = None):
        self.config, self.adapter, self.dry_run = config, adapter, dry_run
        self.root = config.stage_output_dir / "synthesis" / "topic-cache"
        self.lock = threading.Lock()
        self.key_locks: dict[str, threading.Lock] = {}
        self.model_calls = 0
        self.cache_hits = 0
        self.progress = progress
        self.templates = {phase: Template((PROMPTS / f"{phase}.txt").read_text(encoding="utf-8"))
                          for phase in ("topic", "topic_match")}

    def request(self, phase: str, prompt: str, validate) -> dict:
        key = fingerprint(["topic-v1", phase, prompt, self.config.ocr_provider,
                           self.config.ocr_base_url, self.config.synthesis_model,
                           self.config.synthesis_max_tokens])
        with self.lock:
            key_lock = self.key_locks.setdefault(key, threading.Lock())
        with key_lock:
            path = self.root / f"{key}.json"
            try:
                cached = load_json(path)
                if cached:
                    result = validate(cached)
                    with self.lock:
                        self.cache_hits += 1
                    return result
            except (ValueError, OSError, TypeError):
                pass  # Invalid cache entries are never trusted; live runs may regenerate them.
            if self.dry_run:
                raise RuntimeError("Dry run: model discovery is pending.")
            errors = 0
            while errors < self.config.synthesis_max_attempts:
                with self.lock:
                    if self.model_calls >= self.config.topic_max_calls:
                        raise RuntimeError("Topic model-call budget exhausted.")
                    self.model_calls += 1
                try:
                    response = self.adapter.generate(phase, prompt)
                    result = validate(response)
                    write_json(path, result)
                    return result
                except Exception:
                    errors += 1
            # Never serialize an injected/provider exception, which may contain credentials.
            raise RuntimeError("Topic model request failed or returned invalid JSON.")

    def describe(self, path: Path) -> dict:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        source_hash = hashlib.sha256(raw).hexdigest()
        prompt = self.templates["topic"].substitute(
            source_hash=source_hash,
            source=json.dumps(sample_document(text, self.config.topic_sample_chars), ensure_ascii=False))
        try:
            description = self.request("topic", prompt, _description)
            return {"source_file": str(path), "source_sha256": source_hash,
                    **description, "status": "discovered", "warnings": []}
        except RuntimeError as exc:
            return {"source_file": str(path), "source_sha256": source_hash, "topic": "",
                    "summary": "", "structure": "general", "confidence": None,
                    "status": "pending" if self.dry_run else "unclassified",
                    "warnings": [str(exc)]}

    def discover(self, paths: list[Path]) -> dict:
        """Read descriptions concurrently, then match against learned groups in bounded batches."""
        def descriptions():
            with ThreadPoolExecutor(max_workers=self.config.synthesis_concurrency,
                                    thread_name_prefix="topic-discovery") as pool:
                for start in range(0, len(paths), self.config.synthesis_concurrency):
                    yield from pool.map(self.describe, paths[start:start + self.config.synthesis_concurrency])

        documents = []
        groups: list[dict] = []
        assignments: dict[str, str] = {}
        for document in descriptions():
            documents.append(document)
            if self.progress:
                self.progress(f"Theme discovery/grouping: document {len(documents)} of {len(paths)}; "
                              f"{self.model_calls}/{self.config.topic_max_calls} model calls.")
            if document["status"] != "discovered":
                document["group_id"] = None
                document["grouping_status"] = "pending"
                continue
            sha = document["source_sha256"]
            if sha in assignments:
                group_id = assignments[sha]
            else:
                best_id, best_score, complete = None, -1.0, True
                size = self.config.topic_group_batch_size
                for start in range(0, len(groups), size):
                    candidates = groups[start:start + size]
                    if self.progress:
                        self.progress(f"Comparing theme {len(documents)} with groups "
                                      f"{start + 1}–{start + len(candidates)} of {len(groups)}.")
                    ids = {group["id"] for group in candidates}

                    def validate(response: dict) -> dict:
                        if not isinstance(response, dict):
                            raise ValueError("Invalid theme match.")
                        match, similarity = response.get("group_id"), response.get("similarity")
                        if match is not None and match not in ids:
                            raise ValueError("Unknown topic group.")
                        if type(similarity) not in {int, float} or not math.isfinite(similarity) or not 0 <= similarity <= 1:
                            raise ValueError("Invalid topic similarity.")
                        return {"group_id": match, "similarity": similarity}

                    prompt = self.templates["topic_match"].substitute(
                        document=json.dumps({k: document[k] for k in ("topic", "summary")}, ensure_ascii=False),
                        groups=json.dumps([{k: group[k] for k in ("id", "topic", "summary")}
                                           for group in candidates], ensure_ascii=False))
                    try:
                        match = self.request("topic_match", prompt, validate)
                    except RuntimeError as exc:
                        document["warnings"].append(str(exc))
                        complete = False
                        break
                    if match["group_id"] is not None and match["similarity"] > best_score:
                        best_id, best_score = match["group_id"], match["similarity"]
                if not complete:
                    document["group_id"] = None
                    document["grouping_status"] = "pending"
                    continue
                if best_id is not None and best_score >= self.config.topic_similarity_threshold:
                    group_id = best_id
                else:
                    group_id = "topic-" + fingerprint([sha, document["topic"], document["summary"]])[:16]
                    groups.append({"id": group_id, "topic": document["topic"], "summary": document["summary"],
                                   "members": []})
                assignments[sha] = group_id
            document["group_id"] = group_id
            document["grouping_status"] = "grouped"
            next(group for group in groups if group["id"] == group_id)["members"].append({
                "source_file": document["source_file"], "source_sha256": sha})
        return {"mode": "automatic", "method": "model_discovery_and_semantic_matching",
                "documents": documents, "groups": groups, "model_calls": self.model_calls,
                "cache_hits": self.cache_hits, "max_model_calls": self.config.topic_max_calls,
                "dry_run": self.dry_run, "degraded": any(
                    doc["status"] != "discovered" or doc.get("grouping_status") != "grouped"
                    for doc in documents)}


def topic_screening(document: dict) -> ScreeningResult:
    """An unknown theme does not make otherwise usable general text ineligible."""
    return ScreeningResult(document["structure"], document["status"] != "discovered", metadata={
        "synthesis_mode": "automatic", "topic": document["topic"], "topic_summary": document["summary"],
        "topic_group_id": document.get("group_id"), "topic_status": document["status"],
        "topic_confidence": document["confidence"], "topic_warnings": document["warnings"]})
