"""Azure OpenAI vision chat-completions OCR adapter."""

import base64
from urllib.parse import urlencode

from config import PipelineConfig
from ..models import OCRResult
from .base import OCRAdapter


class AzureAdapter(OCRAdapter):
    """Azure OpenAI deployment adapter."""

    provider_name = "azure"

    def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
        key = self._api_key()
        if not self.config.ocr_base_url.strip():
            raise RuntimeError("OCR provider 'azure' requires OCR_BASE_URL for the Azure resource.")
        if not self.config.ocr_model.strip():
            raise RuntimeError("OCR provider 'azure' requires OCR_MODEL, which is the deployment name.")
        base = self.config.ocr_base_url.rstrip("/")
        endpoint = (
            f"{base}/openai/deployments/{self.config.ocr_model}/chat/completions?"
            f"{urlencode({'api-version': '2024-10-21'})}"
        )
        langs = ", ".join(language) if language else "the language visible in the page"
        payload = {
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"Transcribe this page faithfully in {langs}. Preserve wording, numbers, dates, "
                        "and table order. Do not translate, summarize, or add commentary."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")
                    }},
                ],
            }],
        }
        data = self._post_json(
            endpoint, {"api-key": key, "Content-Type": "application/json"}, payload
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("OCR provider 'azure' returned no transcription.") from None
        return OCRResult(text, None, self.provider_name)
