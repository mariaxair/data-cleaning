from config import PipelineConfig
from corpus_pipeline.ocr.base import probe_image, redact_secrets
from corpus_pipeline.ocr.heiyu import HeiyuAdapter


def test_heiyu_endpoint_accepts_bare_host() -> None:
    config = PipelineConfig(ocr_provider="heiyu", ocr_base_url="https://www.heiyucode.com")
    assert HeiyuAdapter(config)._endpoint() == "https://www.heiyucode.com/v1/chat/completions"


def test_heiyu_endpoint_accepts_v1_base() -> None:
    config = PipelineConfig(ocr_provider="heiyu")
    assert HeiyuAdapter(config)._endpoint() == "https://www.heiyucode.com/v1/chat/completions"


def test_heiyu_endpoint_accepts_full_endpoint() -> None:
    config = PipelineConfig(ocr_provider="heiyu", ocr_base_url="https://host/v1/chat/completions")
    assert HeiyuAdapter(config)._endpoint() == "https://host/v1/chat/completions"


def test_redact_secrets_masks_keys_and_query_parameters() -> None:
    text = "Incorrect API key sk-06baa1234567890 for url?key=abc123&x=1"
    redacted = redact_secrets(text)
    assert "sk-06baa1234567890" not in redacted
    assert "abc123" not in redacted
    assert "sk-***" in redacted


def test_connection_probe_is_a_real_image() -> None:
    image = probe_image()
    assert image.startswith(b"\x89PNG")
    assert len(image) > 200
