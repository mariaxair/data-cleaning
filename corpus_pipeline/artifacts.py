"""Deterministic JSON artifacts and atomic replacement for optional stages."""

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def atomic_text(path: Path, text: str) -> None:
    """Replace a completed artifact without leaving a half-written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    atomic_text(path, "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                              for record in records))


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {number} must be an object.")
            records.append(value)
    return records


def export_corpus(records: list[dict], output: Path) -> dict:
    """Export accepted QA only, keeping training data separate from its provenance."""
    accepted, provenance, seen = [], [], set()
    for record in records:
        if record.get("status") != "accept" or record.get("pending_checks"):
            continue
        question, answer = record.get("question", ""), record.get("answer", "")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            continue
        identity = fingerprint([question, answer])
        if identity in seen:
            continue
        seen.add(identity)
        accepted.append({"instruction": question, "input": "", "output": answer})
        provenance.append({"row": len(accepted), "id": record.get("id"), "uuid": record.get("uuid"),
                           "provenance": record.get("provenance", {}), "scores": record.get("scores", {})})
    write_jsonl(output / "corpus.jsonl", accepted)
    write_jsonl(output / "provenance.jsonl", provenance)
    summary = {"input_records": len(records), "exported": len(accepted), "excluded": len(records) - len(accepted),
               "schema": "instruction/input/output", "output": str(output / "corpus.jsonl")}
    write_json(output / "summary.json", summary)
    return summary
