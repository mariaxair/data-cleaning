import io

import pymupdf as fitz
from PIL import Image

from config import PipelineConfig
from corpus_pipeline.models import OCRResult
from corpus_pipeline.processor import process_file


class FakeOCR:
    def ocr_page(self, image: bytes, language: list[str]) -> OCRResult:
        return OCRResult("OCR scanned page", 0.99, "mock")


def test_mixed_pdf_routes_pages_individually(tmp_path) -> None:
    pdf_path = tmp_path / "mixed.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Digital text " * 20)
    page = document.new_page()
    image_buffer = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(image_buffer, format="PNG")
    page.insert_image(fitz.Rect(0, 0, 595, 842), stream=image_buffer.getvalue())
    document.save(pdf_path)
    document.close()
    config = PipelineConfig(output_dir=tmp_path / "out", cache_db=tmp_path / "cache.sqlite3",
                            min_extracted_chars=20, full_page_image_ratio=0.65, render_dpi=72)
    result = process_file(pdf_path, config, ocr_adapter=FakeOCR())
    assert result["detected_type"] == "mixed"
    assert [item["route"] for item in result["pages"]] == ["DIGITAL", "SCANNED"]
    assert "OCR scanned page" in result["cleaned_text"]
