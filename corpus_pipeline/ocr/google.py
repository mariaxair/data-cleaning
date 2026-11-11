"""Google Gemini generateContent OCR adapter."""

import base64
from urllib.parse import quote

from config import PipelineConfig
from ..models import OCRResult
from .base import OCRAdapter


class GoogleAdapter(OCRAdapter):
    """Google Gemini vision adapter."""

    provider_name = "google"

    def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
        key = self._api_key()
        base = self.config.ocr_base_url.strip().rstrip("/") or "https://generativelanguage.googleapis.com/v1beta"
        langs = ", ".join(language) if language else "the language visible in the page"
        url = f"{base}/models/{quote(self.config.ocr_model, safe='')}:generateContent?key={quote(key, safe='')}"
        payload = {
            "contents": [{"parts": [
                {"text": (
                    f"Transcribe this page faithfully in {langs}. Preserve wording, numbers, dates, "
                    "and table order. Do not translate, summarize, or add commentary."
                )},
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(image).decode("ascii")}},
            ]}],
            "generationConfig": {"temperature": 0},
        }
        data = self._post_json(url, {"Content-Type": "application/json"}, payload)
        try:
            text = "".join(
                part.get("text", "")
                for candidate in data["candidates"]
                for part in candidate["content"]["parts"]
            )
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("OCR provider 'google' returned no transcription.") from None
        if not text:
            raise RuntimeError("OCR provider 'google' returned no transcription.")
        return OCRResult(text, None, self.provider_name)
