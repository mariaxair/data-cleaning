"""Shared typed result objects."""

from dataclasses import asdict, dataclass, field
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
    cleaning_hits: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["route"] = self.route.value
        result["signals"] = self.signals.to_dict()
        return result
