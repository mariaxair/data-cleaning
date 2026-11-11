"""Mistral OCR adapter using its OpenAI-compatible chat endpoint."""

from config import PipelineConfig
from .openai_compatible import OpenAICompatibleAdapter


class MistralAdapter(OpenAICompatibleAdapter):
    """Mistral chat-completions vision adapter."""

    provider_name = "mistral"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.provider_name = "mistral"
        if not config.ocr_base_url.strip():
            self.default_base_url = "https://api.mistral.ai/v1"
