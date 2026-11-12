from config import PipelineConfig


def test_gui_session_key_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("HEIYU_API_KEY", "environment-key")
    config = PipelineConfig(ocr_api_key="gui-session-key", session_api_key_override=True)
    assert config.effective_api_key() == "gui-session-key"


def test_provider_environment_key_is_used(monkeypatch) -> None:
    monkeypatch.setenv("HEIYU_API_KEY", "heiyu-key")
    monkeypatch.delenv("OCR_API_KEY", raising=False)
    config = PipelineConfig(ocr_api_key="configured-key")
    assert config.effective_api_key() == "heiyu-key"


def test_legacy_ocr_api_key_is_fallback(monkeypatch) -> None:
    monkeypatch.delenv("HEIYU_API_KEY", raising=False)
    monkeypatch.setenv("OCR_API_KEY", "legacy-key")
    config = PipelineConfig(ocr_api_key="configured-key")
    assert config.effective_api_key() == "legacy-key"


def test_provider_defaults_fill_base_url_and_model() -> None:
    heiyu = PipelineConfig(ocr_provider="heiyu")
    assert heiyu.ocr_base_url == "https://www.heiyucode.com/v1"
    assert heiyu.ocr_model == "gpt-6.1-sol"
    assert heiyu.api_key_env == "HEIYU_API_KEY"

    openai = PipelineConfig(ocr_provider="openai")
    assert openai.ocr_base_url == "https://api.openai.com/v1"
    assert openai.ocr_model == "gpt-4o-mini"
    assert openai.api_key_env == "OPENAI_API_KEY"


def test_explicit_values_override_provider_defaults() -> None:
    config = PipelineConfig(
        ocr_provider="heiyu",
        ocr_base_url="https://proxy.local/v1",
        ocr_model="custom-model",
    )
    assert config.ocr_base_url == "https://proxy.local/v1"
    assert config.ocr_model == "custom-model"


def test_unknown_provider_keeps_explicit_values() -> None:
    config = PipelineConfig(ocr_provider="mock", ocr_base_url="https://mock/v1", ocr_model="mock-model")
    assert config.ocr_provider == "mock"
    assert config.api_key_env == "OCR_API_KEY"
