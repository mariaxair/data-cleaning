"""Text generation through the existing OpenAI-compatible HTTP client."""

import json
from dataclasses import replace
from typing import Protocol

from config import PipelineConfig
from ..ocr.openai_compatible import OpenAICompatibleAdapter


class GenerationAdapter(Protocol):
    def generate(self, phase: str, prompt: str) -> dict:
        """Return parsed JSON with text and source evidence where requested."""
        ...


class CompatibleGenerationAdapter:
    """No network in construction; requests occur only on explicit generate calls."""

    def __init__(self, config: PipelineConfig):
        if config.ocr_provider not in {"openai", "heiyu", "mistral", "openai_compatible"}:
            raise ValueError("Synthesis requires an OpenAI-compatible provider or an injected text adapter.")
        if not config.synthesis_model.strip():
            raise ValueError("Set SYNTHESIS_MODEL explicitly before live synthesis or LLM review.")
        self.config = replace(config, ocr_model=config.synthesis_model)
        self.client = OpenAICompatibleAdapter(self.config)
        self.client._api_key()  # Preflight before queuing work; does not contact the provider.
        self.client._endpoint()

    def generate(self, phase: str, prompt: str) -> dict:
        data = self.client._post_json(
            self.client._endpoint(),
            {"Authorization": f"Bearer {self.client._api_key()}", "Content-Type": "application/json"},
            {"model": self.config.synthesis_model, "temperature": 0,
             "max_tokens": self.config.synthesis_max_tokens,
             "messages": [{"role": "user", "content": prompt}]},
        )
        try:
            value = json.loads(self.client._extract_text(data))
            if not isinstance(value, dict):
                raise ValueError
            return value
        except (ValueError, TypeError):
            raise ValueError("Generation returned invalid JSON; expected an object.") from None
