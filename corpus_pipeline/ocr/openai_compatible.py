"""OpenAI-compatible chat-completions OCR adapter."""

import base64
from typing import Any
from urllib.parse import urlparse

from config import PipelineConfig
from ..models import OCRResult
from .base import OCRAdapter


class OpenAICompatibleAdapter(OCRAdapter):
    """Adapter for OpenAI-compatible vision chat-completions endpoints."""

    provider_name = "openai_compatible"
    default_base_url = "https://api.openai.com/v1"

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.provider_name = config.ocr_provider.lower()

    def _endpoint(self) -> str:
        """Build the chat-completions URL, tolerating a bare host as base URL."""
        base = (self.config.ocr_base_url or self.default_base_url).strip().rstrip("/")
        if not base:
            raise RuntimeError(
                f"OCR provider '{self.provider_name}' requires a base URL; "
                "set it in the provider profile, .env, or the GUI."
            )
        if base.endswith("/chat/completions"):
            return base
        # A bare host such as https://gateway.example.com is assumed to expose /v1.
        if urlparse(base).path in ("", "/"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def _build_payload(self, image: bytes, language: list[str]) -> dict[str, Any]:
        b64 = base64.b64encode(image).decode("ascii")
        langs = ", ".join(language) if language else "the language visible in the page"
        return {
            "model": self.config.ocr_model,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"Transcribe this document page faithfully in {langs}. "
                        "Preserve wording, numbers, dates, tables, and the source script. "
                        "Do not translate, summarize, or add commentary. Return only the transcription."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        }

    def _extract_text(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"OCR provider '{self.provider_name}' returned no transcription.") from None

    def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
        key = self._api_key()
        if not self.config.ocr_model.strip():
            raise RuntimeError(
                f"OCR provider '{self.provider_name}' requires a model; "
                "set it in the provider profile, .env, or the GUI."
            )
        data = self._post_json(
            self._endpoint(),
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            self._build_payload(image, language),
        )
        return OCRResult(self._extract_text(data), None, self.provider_name)
