import json
import os

import pytest

from config import PipelineConfig
from corpus_pipeline.cleaning import clean_text
from corpus_pipeline.processor import _config_hash, process_file
from corpus_pipeline.rules import dates, digits, duplicates, footers, html, scripts, tables


@pytest.mark.parametrize("rule,args,source,expected", [
    (digits.apply, (), "Article Il et المادة ٢l", "Article 11 et المادة ٢١"),
    (digits.apply, (), "Article ١٢ et I like l", "Article ١٢ et I like l"),
    (scripts.apply, ({"א": "ا", "а": "ا", "ա": "ا"},), "بאب بаب بաب", "باب باب باب"),
    (scripts.apply, ({"א": "ا"},), "א alone بالعربية", "א alone بالعربية"),
    (dates.apply, ({"12/25/2024": "25/12/2024"},), "le 12/25/2024", "le 25/12/2024"),
    (dates.apply, ({},), "01/02/2024", "01/02/2024"),
    (footers.apply, (["[page 12]"],), "texte [page 12] suite", "texte suite"),
    (footers.apply, ([],), "Article 12 et 42", "Article 12 et 42"),
    (html.apply, (), "a<br>b<sup>1</sup>", "a\nb1"),
    (html.apply, (), "a < 2", "a < 2"),
    (duplicates.apply, (), "word word word", "word"),
    (duplicates.apply, (), "word other", "word other"),
    (tables.replace_phrases, ({"transcripton": "transcription"},), "transcripton", "transcription"),
    (tables.replace_phrases, ({"transcripton": "transcription"},), "legal phrase", "legal phrase"),
])
def test_rule_positive_negative_and_idempotent(rule, args, source, expected):
    result, count = rule(source, *args)
    assert result == expected
    assert count == (0 if source == expected else {
        scripts.apply: 3, digits.apply: 2, html.apply: 3,
    }.get(rule, 1))
    assert rule(result, *args) == (result, 0)


def test_missing_curated_table_and_empty_data_are_noop(tmp_path):
    assert tables.load_table(tmp_path / "missing.tsv") == {}
    config = PipelineConfig(semantic_rules_enabled=True, rules_data_path=tmp_path / "missing.json",
                            wrong_words_path=tmp_path / "missing.tsv")
    result = clean_text("Article I", config=config)
    assert result.text == "Article 1"
    assert result.hit_counts["article_digit_confusion"] == 1
    assert clean_text(result.text, config=config).text == result.text


def test_tables_reject_chained_or_cyclic_corrections():
    with pytest.raises(ValueError, match="terminal"):
        tables.replace_phrases("a", {"a": "b", "b": "c"})


def test_disabled_output_and_counters_match_legacy(tmp_path):
    text = "OCR transcription result: note\n<sup>1</sup>Hello<br>world\nrepeat\nrepeat\nrepeat\nArticle I"
    result = clean_text(text, config=PipelineConfig())
    assert result.text == "1Hello\nworld\nrepeat\nArticle I"
    assert result.hit_counts == {"encoding_repairs": 0, "metadata_headers": 1,
                                "html_residue": 3, "repeated_lines": 2, "adjacent_word_duplicates": 0}
    source = tmp_path / "sample.txt"
    source.write_text(text, encoding="utf-8")
    config = PipelineConfig(output_dir=tmp_path / "out", cache_db=tmp_path / "cache.db",
                            semantic_rules_enabled=False)
    baseline = process_file(source, config)
    baseline_report_bytes = __import__("pathlib").Path(baseline["report_path"]).read_bytes()
    assert (config.output_dir / (source.stem + "-" + baseline["source_sha256"][:10] + ".txt")).read_bytes() == (
        "--- Page 1 ---\n\n\n1Hello\nworld\nrepeat\nArticle I\n".replace("\n", os.linesep).encode())
    original_hash = _config_hash(config)
    config.synthesis_enabled = config.corpus_processing_enabled = config.review_enabled = True
    rerun = process_file(source, config, force=True)
    assert __import__("pathlib").Path(rerun["report_path"]).read_bytes() == baseline_report_bytes
    assert rerun["cleaned_text"] == baseline["cleaned_text"]
    assert json.loads(open(rerun["report_path"], encoding="utf-8").read())["pages"] == baseline["pages"]
    assert _config_hash(config) == original_hash
