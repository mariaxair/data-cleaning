from config import PROVIDER_PROFILES, PipelineConfig
from corpus_pipeline.models import OCRResult
from corpus_pipeline.ocr.base import OCRAdapter
from corpus_pipeline.ocr.registry import create_adapter, register_adapter, registered_providers


class MockAdapter(OCRAdapter):
    provider_name = "mock"

    def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
        return OCRResult("mock text", 1.0, self.provider_name)


def test_registry_switches_provider_without_pipeline_changes() -> None:
    register_adapter("mock", MockAdapter)
    adapter = create_adapter(PipelineConfig(ocr_provider="mock"))
    assert adapter.ocr_page(b"image", ["ar"]).text == "mock text"


def test_every_profile_has_an_adapter() -> None:
    assert set(PROVIDER_PROFILES) <= set(registered_providers())
