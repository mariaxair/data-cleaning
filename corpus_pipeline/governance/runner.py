"""Separate accepted, manual, intercepted, and duplicate records with audit trails."""

from config import PipelineConfig
from ..artifacts import fingerprint, write_json, write_jsonl
from .dedup import normalize_question
from .dialect import DialectClassifier, create_classifier, dialect_verdict
from .language import assess_language

FIELDS = ("question", "answer", "reasoning_chain")


def govern_record(record: dict, config: PipelineConfig, classifier: DialectClassifier) -> dict:
    result = dict(record)
    declared = record.get("language", "")
    language = {
        field: assess_language(str(record.get(field, "")), declared, config.language_mixing_threshold)
        for field in FIELDS}
    if all(item["detected"] == "ar" for item in language.values()):
        result["language"] = "ar"
    dialects, decisions = {}, []
    for field in FIELDS:
        if language[field]["detected"] == "ar":
            try:
                prediction = classifier.classify(str(record.get(field, "")))
                verdict = dialect_verdict(prediction, config)
                dialects[field] = {**prediction.to_dict(), "decision": verdict}
            except Exception:
                verdict = "manual"
                dialects[field] = {"decision": "manual", "degraded": True,
                                   "warning": "Classifier failed; manual adjudication required."}
            decisions.append(verdict)
    decisions.extend(item["decision"] for item in language.values())
    if any(not str(record.get(field, "")).strip() for field in FIELDS):
        decisions.append("intercept")
    verdict = "intercept" if "intercept" in decisions else "manual" if "manual" in decisions else "pass"
    result["governance"] = {
        "decision": verdict, "language": language, "dialect": dialects,
        "degraded": any(item.get("degraded", True) for item in dialects.values())
                    or any(item["degraded"] for item in language.values()),
    }
    return result


def process_corpus(records: list[dict], config: PipelineConfig, *,
                   classifier: DialectClassifier | None = None) -> dict:
    config.validate_training()
    classifier = classifier or create_classifier(config)
    seen: dict[str, str] = {}
    buckets: dict[str, list[dict]] = {"pass": [], "manual": [], "intercept": [], "duplicates": []}
    for record in records:
        identity = record.get("id") or fingerprint(record)
        key = normalize_question(str(record.get("question", "")))
        if key and key in seen:
            buckets["duplicates"].append({**record, "duplicate_of": seen[key]})
            continue
        if key:
            seen[key] = identity
        governed = govern_record({**record, "id": identity}, config, classifier)
        buckets[governed["governance"]["decision"]].append(governed)
    output = config.stage_output_dir / "governance"
    for name, records_in_bucket in buckets.items():
        write_jsonl(output / f"{name}.jsonl", records_in_bucket)
    summary = {name: len(items) for name, items in buckets.items()}
    summary["degraded_records"] = sum(item["governance"]["degraded"]
                                     for name, items in buckets.items() if name != "duplicates" for item in items)
    summary["language_misplacements"] = sum(
        any(field["misplaced"] for field in item["governance"]["language"].values())
        for name, items in buckets.items() if name != "duplicates" for item in items)
    write_json(output / "summary.json", summary)
    return summary
