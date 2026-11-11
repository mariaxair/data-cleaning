"""OpenAI OCR adapter."""

from config import PipelineConfig
from .openai_compatible import OpenAICompatibleAdapter


class OpenAIAdapter(OpenAICompatibleAdapter):
    """OpenAI vision endpoint adapter."""

    provider_name = "openai"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.provider_name = "openai"
        if not config.ocr_base_url.strip():
            self.default_base_url = "https://api.openai.com/v1"
