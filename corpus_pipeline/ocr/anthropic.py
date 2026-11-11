"""Anthropic Messages API OCR adapter."""

import base64

from config import PipelineConfig
from ..models import OCRResult
from .base import OCRAdapter


class AnthropicAdapter(OCRAdapter):
    """Anthropic vision messages adapter."""

    provider_name = "anthropic"

    def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
        key = self._api_key()
        base = self.config.ocr_base_url.strip().rstrip("/") or "https://api.anthropic.com"
        langs = ", ".join(language) if language else "the language visible in the page"
        payload = {
            "model": self.config.ocr_model,
            "max_tokens": 4096,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"Transcribe this page faithfully in {langs}. Preserve all wording, numbers, "
                        "dates, and table order. Do not translate, summarize, or add commentary."
                    )},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(image).decode("ascii"),
                    }},
                ],
            }],
        }
        data = self._post_json(
            f"{base}/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            payload,
        )
        try:
            text = "".join(item.get("text", "") for item in data["content"] if item.get("type") == "text")
        except (KeyError, TypeError):
            raise RuntimeError("OCR provider 'anthropic' returned no transcription.") from None
        if not text:
            raise RuntimeError("OCR provider 'anthropic' returned no transcription.")
        return OCRResult(text, None, self.provider_name)
