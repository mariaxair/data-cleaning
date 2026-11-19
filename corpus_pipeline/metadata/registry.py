"""Conservative corpus field matching and replayable append-only migrations."""

import json
import re
import unicodedata
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

from config import PipelineConfig
from ..artifacts import fingerprint, read_jsonl, write_json
from ..models import MetadataField, MetadataObservation
from ..settings import load_json


def normalize_label(label: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", label).casefold())
                    .replace("_", " ").split())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_label(a), normalize_label(b)).ratio() if a and b else 0


class FieldRegistry:
    """Single-writer registry; distinct scopes/types never silently merge."""

    def __init__(self, root: Path, config: PipelineConfig):
        self.root, self.config = root, config
        data = load_json(root / "field-registry.json")
        self.version = data.get("version", 0)
        self.entries = {row["field_id"]: row for row in data.get("fields", [])}
        history = root / "registry-history.jsonl"
        if history.is_file():
            for event in read_jsonl(history):
                if event["version"] > self.version:
                    self.entries.update({row["field_id"]: row for row in event["fields_after"]})
                    self.version = event["version"]

    def save(self) -> None:
        write_json(self.root / "field-registry.json", {"schema_version": "1.0", "version": self.version,
                                                     "fields": sorted(self.entries.values(), key=lambda row: row["field_id"])})

    def event(self, action: str, ids: list[str], reason: str, before: list[dict] | None = None) -> None:
        self.version += 1
        for key in ids:
            self.entries[key]["registry_version"] = self.version
        row = {"version": self.version, "action": action, "field_ids": ids, "reason": reason,
               "fields_before": before or [], "fields_after": [self.entries[key] for key in ids]}
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "registry-history.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.save()

    def match(self, observation: MetadataObservation, document_id: str) -> MetadataObservation:
        if observation.status != "verified":
            return observation
        label = normalize_label(observation.label)
        candidates, conflicts = [], []
        for entry in self.entries.values():
            if entry["status"] == "deprecated":
                continue
            exact = label in {normalize_label(v) for v in [entry["canonical_label"], *entry["aliases"]]}
            if entry["value_type"] != observation.value_type or entry["scope"] != observation.scope:
                if exact:
                    conflicts.append(entry)
                continue
            description = _similarity(entry["description"], observation.description)
            # Similar spelling alone is insufficient; compatible descriptions corroborate a fuzzy match.
            if exact:
                if entry["description"] and observation.description and description < self.config.metadata_match_threshold:
                    conflicts.append(entry)
                else:
                    candidates.append((1.0, entry))
            else:
                similarity = _similarity(entry["canonical_label"], observation.label)
                # Synonyms across scripts can share a precise description and an evidenced value shape.
                same_shape = any(type(example["value"]) is type(observation.value) for example in entry["examples"])
                synonym = (normalize_label(entry["description"]) == normalize_label(observation.description)
                           and len(normalize_label(observation.description)) >= 12 and same_shape)
                if description >= self.config.metadata_match_threshold and (synonym or similarity >= self.config.metadata_match_threshold):
                    candidates.append((description if synonym else min(description, similarity), entry))
                elif similarity >= self.config.metadata_match_threshold:
                    conflicts.append(entry)
        candidates.sort(key=lambda pair: (-pair[0], pair[1]["field_id"]))
        if conflicts or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < self.config.metadata_match_margin):
            observation.mapping_status, observation.reason = "uncertain", "Ambiguous field meaning, type, or scope; no merge."
            return observation
        if candidates:
            entry = candidates[0][1]
        else:
            field_id = fingerprint([label, observation.value_type, observation.scope, observation.description])[:24]
            entry = asdict(MetadataField(field_id, observation.label, observation.value_type, observation.scope,
                                        description=observation.description, first_seen=document_id))
            self.entries[field_id] = entry
            self.event("new_field", [field_id], "No compatible existing field; candidate created.")
        field_id = entry["field_id"]
        if observation.label != entry["canonical_label"] and observation.label not in entry["aliases"]:
            entry["aliases"].append(observation.label)
            self.event("alias", [field_id], "Compatible label with matching scope/type/meaning.")
        if document_id not in entry["document_ids"]:
            entry["document_ids"].append(document_id)
            entry["occurrence_count"] = len(entry["document_ids"])
            entry["min_confidence"] = min(entry["min_confidence"], observation.confidence)
            if len(entry["examples"]) < 3:
                entry["examples"].append({"value": observation.value, "evidence": [asdict(e) for e in observation.evidence],
                                          "document_id": document_id})
            self.event("occurrence", [field_id], "New document with evidence-backed compatible observation.")
        if (entry["status"] == "candidate" and
                entry["occurrence_count"] >= self.config.metadata_promotion_documents and
                entry["min_confidence"] >= self.config.metadata_min_confidence):
            entry["status"] = "stable"
            self.event("promotion", [field_id], f"{entry['occurrence_count']} distinct documents; "
                       f"minimum confidence {entry['min_confidence']}; extraction quality still unvalidated.")
        observation.field_id, observation.mapping_status = field_id, entry["status"]
        observation.registry_version = self.version
        return observation

    def migrate(self, action: str, field_id: str, *, reason: str, target_id: str | None = None,
                label: str | None = None, value_type: str | None = None, children: list[str] | None = None) -> list[str]:
        """Explicit operator adjudication; snapshots keep historical records interpretable."""
        if not reason.strip():
            raise ValueError("Registry migrations require an explicit reason.")
        if action not in {"merge", "split", "rename", "type_change", "deprecation", "quality_approval", "uncertainty"}:
            raise ValueError("Unsupported registry migration.")
        entry = self.entries[field_id]
        ids = [field_id] + ([target_id] if target_id else [])
        before = json.loads(json.dumps([self.entries[key] for key in ids]))
        if action == "merge":
            if target_id is None or target_id == field_id:
                raise ValueError("Merge requires a different target field.")
            target = self.entries[target_id]
            if (entry["value_type"], entry["scope"]) != (target["value_type"], target["scope"]):
                raise ValueError("Field merge requires identical type and scope.")
            target["aliases"] = sorted(set([*target["aliases"], entry["canonical_label"], *entry["aliases"]]))
            target["document_ids"] = sorted(set(target["document_ids"] + entry["document_ids"]))
            target["occurrence_count"] = len(target["document_ids"])
            target["min_confidence"] = min(entry["min_confidence"], target["min_confidence"])
            target["quality_approved"] = False
            entry.update(status="deprecated", replaced_by=target_id)
        elif action == "split":
            if not children or any(not isinstance(item, str) or not item.strip() for item in children) or len(set(children)) < 2:
                raise ValueError("Split requires at least two different labels.")
            for child in children:
                new_id = fingerprint([field_id, child, self.version + 1])[:24]
                self.entries[new_id] = asdict(MetadataField(new_id, child, entry["value_type"], entry["scope"]))
                ids.append(new_id)
            entry.update(status="deprecated", replaced_by=ids[1:])
        elif action == "rename":
            if not label or not label.strip():
                raise ValueError("Rename requires a label.")
            entry["aliases"] = sorted(set(entry["aliases"] + [entry["canonical_label"]]))
            entry["canonical_label"] = label
        elif action == "type_change":
            from .extraction import VALUE_TYPES
            if value_type not in VALUE_TYPES:
                raise ValueError("Unsupported field type.")
            entry.update(value_type=value_type, status="candidate", quality_approved=False,
                         document_ids=[], occurrence_count=0, examples=[], min_confidence=1.0)
        elif action == "quality_approval":
            entry["quality_approved"] = True
        elif action == "uncertainty":
            entry.update(status="uncertain", quality_approved=False)
        else:
            entry["status"] = "deprecated"
        self.event(action, ids, reason, before)
        return ids
