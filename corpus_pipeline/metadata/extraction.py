"""Validate generic model observations against exact source intervals."""

import json
import math
import re
from string import Template
from pathlib import Path

from ..models import MetadataObservation
from .legal import evidence_at, invalid_calendar_date, normalize_date

PROMPT_VERSION = "metadata-open-v1"
VALUE_TYPES = {"string", "number", "boolean", "date", "date_range", "domain", "language"}


def metadata_prompt(text: str, chunk: dict, context: str = "") -> str:
    template = Template(Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8"))
    return template.substitute(source=json.dumps(text[chunk["start"]:chunk["end"]], ensure_ascii=False),
                               start=chunk["start"], pages=json.dumps(chunk.get("pages", [])),
                               context=json.dumps(context[:1000], ensure_ascii=False))


def validate_observations(response: dict, text: str, start: int, end: int) -> list[MetadataObservation]:
    """Unsupported values remain auditable; model confidence never bypasses evidence."""
    if not isinstance(response, dict) or not isinstance(response.get("observations"), list):
        raise ValueError("Metadata response must contain an observations array.")
    if len(response["observations"]) > 200:
        raise ValueError("Metadata response contains too many observations.")
    result = []
    for raw in response["observations"]:
        if not isinstance(raw, dict):
            result.append(MetadataObservation("invalid", None, status="rejected", reason="Observation must be an object."))
            continue
        label = raw.get("label")
        confidence = raw.get("confidence", 0)
        value_type, scope = raw.get("value_type", "string"), raw.get("scope", "document")
        value_type = value_type if isinstance(value_type, str) else "invalid"
        scope = scope if isinstance(scope, str) else "invalid"
        value = raw.get("value")
        observation = MetadataObservation(
            label if isinstance(label, str) and 0 < len(label.strip()) <= 160 else "invalid",
            value, value_type if isinstance(value_type, str) else "invalid",
            scope if isinstance(scope, str) else "invalid", original_value=value,
            confidence=float(confidence) if type(confidence) in {int, float} and math.isfinite(confidence) else 0,
            description=raw.get("description", "")[:500] if isinstance(raw.get("description", ""), str) else "",
            role=raw.get("role", "unknown") if isinstance(raw.get("role", "unknown"), str) else "unknown",
            assertion=raw.get("assertion", "explicit") if isinstance(raw.get("assertion", "explicit"), str) else "invalid")
        if observation.assertion == "absent" and value is None:
            observation.status, observation.reason = "missing", "Source value explicitly reported absent; not canonical metadata."
            result.append(observation)
            continue
        errors = []
        if (observation.label == "invalid" or type(confidence) not in {int, float} or
                not math.isfinite(confidence) or not 0 <= confidence <= 1):
            errors.append("Invalid label or confidence.")
        valid_type = ((value_type in {"string", "date", "domain", "language"} and isinstance(value, str) and bool(value.strip())) or
                      (value_type == "number" and type(value) in {int, float} and math.isfinite(value)) or
                      (value_type == "boolean" and type(value) is bool) or
                      (value_type == "date_range" and isinstance(value, list) and len(value) == 2 and
                       all(isinstance(v, str) for v in value))) if isinstance(value_type, str) else False
        if value_type not in VALUE_TYPES or scope not in {"document", "chunk"} or not valid_type:
            errors.append("Unsupported value type, scope, or value.")
        if observation.assertion not in {"explicit", "ambiguous", "inferred", "absent"}:
            errors.append("Invalid assertion status.")
        citations = raw.get("evidence", [])
        if not isinstance(citations, list) or not citations:
            errors.append("Missing exact evidence.")
            citations = []
        for citation in citations:
            if not isinstance(citation, dict) or not isinstance(citation.get("quote"), str) or not citation["quote"]:
                errors.append("Invalid evidence quote.")
                continue
            quote = citation["quote"]
            a, b = citation.get("start"), citation.get("end")
            if a is None and b is None:
                found = text.find(quote, start, end)
                if found < 0 or text.find(quote, found + 1, end) >= 0:
                    errors.append("Quote is absent or ambiguous in supplied span.")
                    continue
                a, b = found, found + len(quote)
            if type(a) is not int or type(b) is not int or not start <= a < b <= end or text[a:b] != quote:
                errors.append("Evidence offsets do not match the supplied source.")
                continue
            evidence = evidence_at(text, a, b)
            if citation.get("page") is not None and citation["page"] != evidence.page:
                errors.append("Evidence page does not match the source page map.")
                continue
            observation.evidence.append(evidence)
        quotes = "\n".join(e.quote for e in observation.evidence)
        if valid_type and value_type in {"string", "date", "date_range", "number"}:
            values = value if isinstance(value, list) else [str(value)]
            if any(v not in quotes for v in values):
                errors.append("Value is not present verbatim in its evidence.")
        if value_type in {"date", "date_range"} and valid_type:
            normalized = [normalize_date(v) for v in value] if value_type == "date_range" else normalize_date(value)
            observation.normalized_value = normalized
            for original in (value if value_type == "date_range" else [value]):
                if invalid_calendar_date(original):
                    errors.append("Calendar-invalid date.")
            if value_type == "date_range" and all(normalized) and normalized[0] > normalized[1]:
                errors.append("Date range endpoints are reversed.")
            if observation.role != "unknown":
                patterns = {"signature": r"حرر|signé|signed|signature", "publication": r"نشر|publication|published|publié",
                            "event": r"حدث|فعالية|event|événement", "entry_into_force": r"حيز التنفيذ|entrée en vigueur|entry into force"}
                if observation.role not in patterns or not re.search(patterns[observation.role], quotes, re.I):
                    errors.append("Date role lacks explicit supporting evidence.")
        if errors:
            observation.status, observation.reason = "rejected", " ".join(dict.fromkeys(errors))
        elif observation.assertion != "explicit":
            observation.status, observation.reason = "uncertain", "Value is not explicitly asserted."
        else:
            observation.status = "verified"
        result.append(observation)
    return result
