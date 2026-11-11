"""HeiyuCode OpenAI-compatible OCR adapter."""

from config import PipelineConfig
from .openai_compatible import OpenAICompatibleAdapter


class HeiyuAdapter(OpenAICompatibleAdapter):
    """HeiyuCode gateway adapter (chat-completions wire format)."""

    provider_name = "heiyu"
    default_base_url = "https://www.heiyucode.com/v1"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.provider_name = "heiyu"
