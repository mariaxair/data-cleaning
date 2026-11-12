"""Injectable semantic review and repair clients."""

import json
from pathlib import Path
from string import Template
from typing import Protocol

from config import PipelineConfig
from ..synthesis.client import CompatibleGenerationAdapter
from .checks import LLM_DIMENSIONS


class ReviewAdapter(Protocol):
    def review(self, record: dict) -> dict[str, int]: ...


class RepairAdapter(Protocol):
    def repair(self, record: dict, fields: list[str], issues: dict) -> dict: ...


class ModelReviewAdapter:
    def __init__(self, config: PipelineConfig):
        self.client = CompatibleGenerationAdapter(config)

    def review(self, record: dict) -> dict[str, int]:
        template = Template((Path(__file__).parent / "prompts/review.txt").read_text(encoding="utf-8"))
        prompt = template.substitute(
            domain_context=("Review this general-domain QA in its actual subject, without imposing a legal framework."
                            if record.get("provenance", {}).get("tier") == "general"
                            else "Review this legal QA with attention to legal framework and exact provisions."),
            dimensions=", ".join(LLM_DIMENSIONS),
            record=json.dumps({key: record.get(key) for key in (
                "question", "answer", "reasoning_chain", "source_text", "provenance")}, ensure_ascii=False))
        return self.client.generate("review", prompt).get("scores", {})


class ModelRepairAdapter:
    def __init__(self, config: PipelineConfig):
        self.client = CompatibleGenerationAdapter(config)

    def repair(self, record: dict, fields: list[str], issues: dict) -> dict:
        template = Template((Path(__file__).parent / "prompts/repair.txt").read_text(encoding="utf-8"))
        keys = ("question", "answer", "reasoning_chain", "language")
        if fields != ["reasoning_chain"]:
            keys += ("source_text", "provenance")
        return self.client.generate("repair", template.substitute(
            domain_context=("Keep the actual subject matter; do not impose laws or a legal framework."
                            if record.get("provenance", {}).get("tier") == "general"
                            else "Preserve the legal framework and exact legal references."),
            fields=json.dumps(fields), issues=json.dumps(issues, ensure_ascii=False),
            record=json.dumps({key: record.get(key) for key in keys}, ensure_ascii=False)))
