"""Shared typed result objects."""

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any


class PageRoute(str, Enum):
    DIGITAL = "DIGITAL"
    SCANNED = "SCANNED"
    SUSPECT = "SUSPECT"


@dataclass
class OCRResult:
    text: str
    confidence: float | None
    provider: str


@dataclass
class PageSignals:
    extracted_characters: int
    valid_character_ratio: float
    full_page_image_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageReport:
    page_number: int
    route: PageRoute
    signals: PageSignals
    warnings: list[str] = field(default_factory=list)
    ocr_failed: bool = False
    ocr_provider: str | None = None
    ocr_latency_s: float | None = None
    ocr_confidence: float | None = None
    ocr_from_cache: bool = False
    cleaning_hits: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["route"] = self.route.value
        result["signals"] = self.signals.to_dict()
        return result


METADATA_SCHEMA_VERSION = "1.0"


class MetadataStatus(str, Enum):
    COMPLETE = "complete"
    PENDING = "extraction_pending"
    FALLBACK = "needs_fallback"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class Evidence:
    """An exact interval in the cleaned source, with optional physical page."""

    quote: str
    start: int
    end: int
    page: int | None = None


@dataclass
class MetadataObservation:
    """One open-ended field; labels never define a per-document schema."""

    label: str
    value: Any
    value_type: str = "string"
    scope: str = "document"
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0
    original_value: Any = None
    field_id: str | None = None
    method: str = "llm_open_extraction"
    status: str = "uncertain"
    description: str = ""
    role: str = "unknown"
    normalized_value: Any = None
    reason: str = ""
    mapping_status: str = "pending"
    registry_version: int = 0
    chunk_id: str | None = None
    assertion: str = "explicit"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MetadataObservation":
        """Ignore future optional fields while restoring typed evidence."""
        data = {key: item for key, item in value.items() if key in {f.name for f in fields(cls)}}
        data["evidence"] = [Evidence(**{key: item for key, item in row.items()
                                       if key in {f.name for f in fields(Evidence)}})
                            for row in data.get("evidence", [])]
        return cls(**data)


@dataclass
class MetadataDocument:
    """Technical envelope separated from evidence-backed content observations."""

    document_id: str
    source: str
    source_sha256: str
    metadata_schema_version: str = METADATA_SCHEMA_VERSION
    document_type: str = "unknown"
    language: dict[str, Any] = field(default_factory=lambda: {
        "value": "unknown", "method": "none", "confidence": None, "reason": "Not determined."})
    domains: list[dict[str, Any]] = field(default_factory=list)
    dates: list[MetadataObservation] = field(default_factory=list)
    metadata: list[MetadataObservation] = field(default_factory=list)
    metadata_status: str = MetadataStatus.PENDING.value
    registry_version: int = 0
    chunks: list[dict[str, Any]] = field(default_factory=list)
    processing: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MetadataDocument":
        data = {key: item for key, item in value.items() if key in {f.name for f in fields(cls)}}
        for key in ("metadata", "dates"):
            data[key] = [MetadataObservation.from_dict(row) for row in data.get(key, [])]
        return cls(**data)


@dataclass
class MetadataField:
    """Versioned registry entry, counted once per document, never per rerun."""

    field_id: str
    canonical_label: str
    value_type: str
    scope: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    first_seen: str = ""
    occurrence_count: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)
    registry_version: int = 1
    status: str = "candidate"
    document_ids: list[str] = field(default_factory=list)
    min_confidence: float = 1.0
    quality_approved: bool = False
