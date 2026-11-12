"""Quality scoring and bounded repair with a hard dialect veto."""

from copy import deepcopy
from dataclasses import asdict

from config import PipelineConfig
from ..artifacts import fingerprint, write_json, write_jsonl
from ..governance.dialect import DialectClassifier, create_classifier
from .adapters import ModelRepairAdapter, ModelReviewAdapter, RepairAdapter, ReviewAdapter
from .checks import FIELDS, LLM_DIMENSIONS, ReviewResult, deterministic_review


def score_record(record: dict, config: PipelineConfig, classifier: DialectClassifier,
                 reviewer: ReviewAdapter | None = None) -> ReviewResult:
    result = deterministic_review(record, config, classifier)
    if result.dialect_veto:
        for name in LLM_DIMENSIONS:
            result.scores.setdefault(name, 1)
        result.skipped.extend(name for name in LLM_DIMENSIONS if name not in result.skipped)
        return result
    semantic = {}
    if reviewer is not None:
        try:
            semantic = reviewer.review(record)
        except Exception:
            result.skipped.append("semantic_reviewer_failed")
    for name in LLM_DIMENSIONS:
        value = semantic.get(name) if isinstance(semantic, dict) else None
        if type(value) is not int or value not in {0, 1, 2}:
            value = 1
            result.skipped.append(name)
        result.scores[name] = min(result.scores.get(name, 2), value)
    return result


def repair_fields(result: ReviewResult) -> list[str]:
    if result.dialect_veto:
        return []
    scores = result.scores
    if any(scores.get(name) == 0 for name in ("question_accuracy", "question_self_containedness", "privacy_compliance")):
        fields = {"question", "answer"}
        if scores.get("privacy_compliance") == 0:
            fields.add("reasoning_chain")
    else:
        fields = set()
    if scores.get("reasoning_completeness") == 0:
        fields.add("answer")
    for dimension, failed in result.failed_fields.items():
        if scores.get(dimension) == 0:
            fields.update(failed)
    if "question" in fields:
        fields.add("answer")
    return [field for field in FIELDS if field in fields]


def route_review(result: ReviewResult) -> str:
    if result.dialect_veto:
        return "reject"
    if any(score == 0 for score in result.scores.values()):
        return "repair" if repair_fields(result) else "reject"
    if result.skipped or any(score == 1 for score in result.scores.values()):
        return "manual"
    return "accept"


def review_record(record: dict, config: PipelineConfig, classifier: DialectClassifier, *,
                  reviewer: ReviewAdapter | None = None, repairer: RepairAdapter | None = None) -> dict:
    current = deepcopy(record)
    history = []
    for attempt in range(config.review_max_attempts + 1):
        result = score_record(current, config, classifier, reviewer)
        decision = route_review(result)
        history.append({"attempt": attempt, "decision": decision, **asdict(result)})
        if decision != "repair":
            break
        if repairer is None:
            decision = "manual"
            history[-1]["repair_skipped"] = "No repair adapter configured."
            break
        if attempt >= config.review_max_attempts:
            decision = "reject"
            history[-1]["repair_exhausted"] = True
            break
        fields = repair_fields(result)
        try:
            patch = repairer.repair(deepcopy(current), fields, result.reasons)
            if not isinstance(patch, dict) or set(patch) - (set(fields) | {"evidence"}):
                raise ValueError("Repair changed fields outside its scope.")
            if any(not isinstance(patch.get(field), str) or not patch[field].strip() for field in fields):
                raise ValueError("Repair omitted requested text.")
            for field in fields:
                current[field] = patch[field]
            if "evidence" in patch:
                if not isinstance(patch["evidence"], dict) or set(patch["evidence"]) - set(fields):
                    raise ValueError("Repair evidence must match requested fields.")
                current["evidence"] = {**current.get("evidence", {}), **patch["evidence"]}
            history[-1]["repaired_fields"] = fields
        except Exception:
            decision = "reject"
            history[-1]["repair_error"] = "Repair adapter failed or returned an invalid patch."
            break
    current["scores"] = result.scores
    current["review"] = {"decision": decision, "attempts": len(history) - 1,
                         "history": history, "degraded": result.degraded,
                         "skipped_dimensions": result.skipped}
    current["status"] = decision
    if decision == "accept":
        current["pending_checks"] = []
    return current


def review_corpus(records: list[dict], config: PipelineConfig, *,
                  reviewer: ReviewAdapter | None = None, repairer: RepairAdapter | None = None,
                  classifier: DialectClassifier | None = None) -> dict:
    config.validate_training()
    classifier = classifier or create_classifier(config)
    if config.review_llm_enabled:
        reviewer = reviewer or ModelReviewAdapter(config)
        repairer = repairer or ModelRepairAdapter(config)
    buckets = {"accept": [], "reject": [], "manual": []}
    for record in records:
        result = review_record({**record, "id": record.get("id") or fingerprint(record)}, config,
                               classifier, reviewer=reviewer, repairer=repairer)
        buckets[result["review"]["decision"]].append(result)
    output = config.stage_output_dir / "review"
    write_jsonl(output / "audit.jsonl", [r for records_in_bucket in buckets.values() for r in records_in_bucket])
    for name, filename in (("accept", "accepted"), ("reject", "rejected"), ("manual", "manual")):
        records_in_bucket = buckets[name]
        if name == "accept":
            records_in_bucket = [{k: v for k, v in record.items()
                                  if k not in {"source_text", "sensitive_values", "evidence", "governance", "review"}}
                                 for record in records_in_bucket]
        write_jsonl(output / f"{filename}.jsonl", records_in_bucket)
    summary = {name: len(records_in_bucket) for name, records_in_bucket in buckets.items()}
    summary["repair_attempts"] = sum(r["review"]["attempts"] for items in buckets.values() for r in items)
    summary["skipped_semantic_records"] = sum(bool(r["review"]["skipped_dimensions"])
                                             for items in buckets.values() for r in items)
    write_json(output / "summary.json", summary)
    return summary
