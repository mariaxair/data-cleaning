from config import PipelineConfig
from corpus_pipeline.cleaning import clean_text
from corpus_pipeline.models import PageRoute
from corpus_pipeline.routing import classify_page, measure_page


def test_page_routes_digital_scanned_and_suspect() -> None:
    config = PipelineConfig(min_extracted_chars=20, min_valid_char_ratio=0.8, full_page_image_ratio=0.65)
    digital_text = "Article 1: This is a clean page with enough extracted text. " * 2
    assert classify_page(digital_text, measure_page(digital_text, 0.1), config)[0] == PageRoute.DIGITAL
    assert classify_page("", measure_page("", 0.95), config)[0] == PageRoute.SCANNED
    corrupt = "�\ue000�\ue001" * 20
    assert classify_page(corrupt, measure_page(corrupt, 0.0), config)[0] == PageRoute.SUSPECT


def test_cleaning_rules_are_traceable() -> None:
    result = clean_text(
        "OCR transcription result: generated metadata\n"
        "<sup>1</sup>Hello<br>world\nrepeat\nrepeat\nrepeat\nword word"
    )
    assert "generated metadata" not in result.text
    assert "<sup>" not in result.text
    assert "Hello\nworld" in result.text
    assert result.text.count("repeat") == 1
    assert "word word" not in result.text
    assert result.hit_counts["metadata_headers"] == 1
    assert result.hit_counts["html_residue"] >= 2
    assert result.hit_counts["repeated_lines"] == 2
    assert result.hit_counts["adjacent_word_duplicates"] == 1
