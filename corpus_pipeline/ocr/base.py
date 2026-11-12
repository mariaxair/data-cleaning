"""Base OCR interface and safe HTTP retry handling."""

import io
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from config import PipelineConfig
from ..models import OCRResult


_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(?i)((?:api[-_]?key|access[-_]?token|token|key)=)[^&\s\"']+"),
)


def redact_secrets(text: str) -> str:
    """Mask API keys and key-like query parameters before logging or raising."""
    redacted = _SECRET_PATTERNS[0].sub("sk-***", text)
    return _SECRET_PATTERNS[1].sub(lambda match: f"{match.group(1)}***", redacted)


def probe_image() -> bytes:
    """Build a small, non-trivial PNG used to validate provider credentials.

    Some gateways reject degenerate 1x1 images, so the probe contains visible
    text at a normal page-like aspect ratio.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 96), "white")
    ImageDraw.Draw(image).text((16, 40), "OCR TEST 123", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class OCRAdapter(ABC):
    """Provider interface with shared key validation and retry behavior."""

    provider_name = "unknown"
    default_base_url = ""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _api_key(self) -> str:
        key = self.config.effective_api_key()
        if not key or (key.startswith("YOUR_") and key.endswith("_HERE")):
            raise RuntimeError(
                f"OCR provider '{self.provider_name}' requires an API key; "
                f"set {self.config.api_key_env} in .env or enter one in the GUI."
            )
        return key

    @abstractmethod
    def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
        """Transcribe one page image."""

    def test_connection(self) -> OCRResult:
        """Send a tiny in-memory image request to validate provider settings."""
        return self.ocr_page(probe_image(), self.config.languages)

    @staticmethod
    def _response_detail(response: requests.Response, secret: str = "") -> str:
        """Extract a short, secret-free provider error message."""
        def sanitize(message: str) -> str:
            if secret:
                message = message.replace(secret, "***")
            return redact_secrets(message)[:240]

        try:
            data = response.json()
        except ValueError:
            return sanitize(response.text)
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                message = error.get("message") or error.get("code") or ""
            else:
                message = error or data.get("message") or ""
            if message:
                return sanitize(str(message))
        return ""

    def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        retries = max(0, self.config.ocr_max_retries)
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.ocr_timeout_s,
                )
            except requests.Timeout:
                if attempt < retries:
                    time.sleep(self.config.ocr_retry_base_s * (2**attempt))
                    continue
                raise RuntimeError(f"OCR provider '{self.provider_name}' timed out.") from None
            except requests.RequestException:
                if attempt < retries:
                    time.sleep(self.config.ocr_retry_base_s * (2**attempt))
                    continue
                raise RuntimeError(f"OCR provider '{self.provider_name}' could not reach the configured URL.") from None

            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"OCR provider '{self.provider_name}' rejected the API key (HTTP {response.status_code}); "
                    f"check {self.config.api_key_env}, the provider selection, the model, and the base URL."
                )
            if response.status_code == 404:
                raise RuntimeError(
                    f"OCR provider '{self.provider_name}' returned HTTP 404; check the base URL "
                    "(OpenAI-compatible bases usually end with /v1) and the model name."
                )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < retries:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = min(float(retry_after), 60.0) if retry_after else self.config.ocr_retry_base_s * (2**attempt)
                    except ValueError:
                        delay = self.config.ocr_retry_base_s * (2**attempt)
                    time.sleep(max(0.0, delay))
                    continue
            if not response.ok:
                detail = self._response_detail(response, self.config.effective_api_key())
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(
                    f"OCR provider '{self.provider_name}' returned HTTP {response.status_code}{suffix}; "
                    "check the provider, base URL, and model."
                )
            try:
                return response.json()
            except ValueError:
                raise RuntimeError(f"OCR provider '{self.provider_name}' returned an invalid response.") from None
        raise RuntimeError(f"OCR provider '{self.provider_name}' failed after retries.")
