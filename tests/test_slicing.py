"""Slicing tests: whole-issue OCR Markdown to one record per instrument."""

import json

from corpus_pipeline.slicing import slice_folder, slice_markdown

MASTHEAD = """# الخميس 3 شعبان عام 1447 هـ
الموافق 22 جانفي سنة 2026 م

العدد 06

## الجمهورية الجزائرية الديمقراطية الشعبية
### الجريدة الرسمية
"""

INDEX = """# فهرس
## مراسيم تنظيمية
مرسوم تنفيذي رقم 26-66 مؤرخ في 24 رجب عام 1447 الموافق 13 جانفي سنة 2026، يعدل المرسوم التنفيذي رقم 24-115 ... 5
مرسوم تنفيذي رقم 26-80 مؤرخ في 25 رجب عام 1447 الموافق 14 جانفي سنة 2026، يحدد كيفيات الاستفادة ... 9
"""

DECREE_66 = """## مرسوم تنفيذي رقم 26-66 مؤرخ في 24 رجب عام 1447 الموافق 13 جانفي سنة 2026، يعدل المرسوم التنفيذي رقم 24-115.
إن الوزير الأول،
- بناء على الدستور،
يقرر ما يأتي :
## المادة الأولى : تستبدل عبارة "الوزير المكلف بالتجارة" بعبارة "الوزير المكلف بضبط السوق الوطنية".
## المادة 2 : ينشر هذا المرسوم في الجريدة الرسمية للجمهورية الجزائرية الديمقراطية الشعبية.
حرر بالجزائر في 24 رجب عام 1447 الموافق 13 جانفي سنة 2026.

سيفي غريب
"""

DECREE_80 = """## مرسوم تنفيذي رقم 26-80 مؤرخ في 25 رجب عام 1447 الموافق 14 جانفي سنة 2026، يحدد كيفيات استفادة الأشخاص ذوي الاحتياجات الخاصة.
# المادة 3 : يستفيد الأشخاص ذوو الاحتياجات الخاصة من مجانية النقل العمومي.
# المادة 4 : يستفيد حاملو بطاقة المعاق من تخفيض.
# المادة 5 : ينشر هذا المرسوم في الجريدة الرسمية.
"""


def issue(*parts: str) -> str:
    return "\n\n".join(parts)


def test_index_masthead_and_furniture_are_not_part_of_the_slice():
    slices, summary = slice_markdown(issue(MASTHEAD, INDEX, DECREE_66), "issue.md")
    assert [item.law_number for item in slices] == ["26-66"]
    assert summary["instruments"] == 1
    text = slices[0].text
    assert text.startswith("## مرسوم تنفيذي رقم 26-66")
    assert "المادة الأولى" in text and "المادة 2" in text
    assert "فهرس" not in text and "العدد 06" not in text
    assert slices[0].law_type == "مرسوم تنفيذي"
    assert slices[0].year == "2026"
    assert slices[0].lang == "ar"
    assert slices[0].category == "statutory" and slices[0].eligible


def test_missing_leading_articles_are_reported_and_excluded_from_qa():
    slices, _ = slice_markdown(issue(MASTHEAD, INDEX, DECREE_80), "issue.md")
    record = slices[0]
    assert record.article_numbers == ["3", "4", "5"]
    assert "missing_articles:1-2" in record.warnings
    assert not record.eligible


def test_reversed_number_is_repaired_from_the_index():
    body = DECREE_66.replace("رقم 26-66", "رقم 66-26")
    slices, _ = slice_markdown(issue(MASTHEAD, INDEX, body), "issue.md")
    assert slices[0].law_number == "26-66"
    assert "number_reversed_in_ocr" in slices[0].warnings


def test_heading_glued_to_previous_paragraph_is_split_into_two_records():
    glued = """**مرسوم تنفيذي مؤرخ في 2 شعبان عام 1447 الموافق
21 جانفي سنة 2026، يتضمن إنهاء مهام مدير الضرائب.**

بموجب مرسوم تنفيذي مؤرخ في 2 شعبان عام 1447 الموافق 21 جانفي سنة 2026، تنهى مهام السيد فلان. **مرسوم تنفيذي مؤرخ في 2 شعبان عام 1447 الموافق
21 جانفي سنة 2026، يتضمن تعيين مدير التربية.**

بموجب مرسوم تنفيذي مؤرخ في 2 شعبان عام 1447 الموافق 21 جانفي سنة 2026، يعين السيد فلان.
"""
    slices, summary = slice_markdown(issue(MASTHEAD, glued), "issue.md")
    assert summary["repairs"] == ["split_joined_heading"]
    assert [item.category for item in slices] == ["individual", "individual"]
    assert "الضرائب" in slices[0].title
    assert "التربية" in slices[1].title
    assert "تنهى مهام" in slices[0].text and "يعين السيد" not in slices[0].text


def test_quoted_amended_articles_do_not_break_the_instrument_sequence():
    amended = """## مرسوم رئاسي رقم 26-89 مؤرخ في 7 شعبان عام 1447 الموافق 26 جانفي سنة 2026، يعدل ويتمم المرسوم الرئاسي رقم 03-270.
إن رئيس الجمهورية،
يرسم ما يأتي :
#### المادة الأولى : يهدف هذا المرسوم إلى تعديل بعض الأحكام.
**المادة 2 :** تعدل أحكام المادتين 16 و17 وتحرر كما يأتي :
**"المادة 16 :** يكلف المجلس العلمي بإبداء الآراء.
**المادة 17 :** يضم المجلس العلمي ما يأتي :
- رؤساء المصالح الطبية.
**المادة 3 :** ينشر هذا المرسوم في الجريدة الرسمية.
"""
    slices, _ = slice_markdown(issue(MASTHEAD, amended), "issue.md")
    record = slices[0]
    assert record.article_numbers == ["1", "2", "3"]
    assert record.quoted_articles == ["16", "17"]
    assert "missing_articles" not in " ".join(record.warnings)
    assert "amending_instrument" in record.warnings


def test_annexed_treaty_becomes_its_own_linked_record():
    decree = """## مرسوم رئاسي رقم 26-90 مؤرخ في 7 شعبان عام 1447 الموافق 26 جانفي سنة 2026، يتضمن التصديق على اتفاق.
إن رئيس الجمهورية،
يرسم ما يأتي :
### المادة الأولى : يصدق على الاتفاق المرفق.
### المادة 2 : ينشر هذا المرسوم في الجريدة الرسمية.
حرر بالجزائر في 7 شعبان عام 1447 الموافق 26 جانفي سنة 2026.

عبد المجيد تبون
### اتفاق بين حكومة الجزائر وحكومة إيطاليا.
اتفقتا على ما يأتي :
## المادة الأولى
يعترف كل طرف برخص السياقة.
## المادة 2
تنتهي صلاحية الرخصة بعد ستة أشهر.
"""
    slices, _ = slice_markdown(issue(MASTHEAD, decree), "issue.md")
    assert [item.category for item in slices] == ["statutory", "agreement"]
    assert slices[0].law_number == "26-90"
    assert slices[1].law_number is None and slices[1].parent == "26-90"
    assert "annex" in slices[1].warnings
    assert "اتفاق بين حكومة" in slices[1].text


def test_preamble_is_dropped_unless_requested():
    body = issue(MASTHEAD, INDEX, DECREE_66)
    default, _ = slice_markdown(body, "issue.md")
    with_preamble, _ = slice_markdown(body, "issue.md", include_preamble=True)
    assert "إن الوزير الأول" not in default[0].text
    assert "إن الوزير الأول" in with_preamble[0].text
    assert "يقرر ما يأتي" in with_preamble[0].text


def test_record_keeps_the_china_team_schema_keys():
    slices, _ = slice_markdown(issue(MASTHEAD, INDEX, DECREE_66), "A2026006.md")
    record = slices[0].record()
    for key in ("text", "law_number", "law_type", "year", "lang", "source"):
        assert key in record
    assert isinstance(record["year"], str)
    assert record["source"] == "A2026006.md"
    assert record["article_count"] == 2


def test_folder_run_writes_one_jsonl_per_issue(tmp_path):
    first = tmp_path / "A2026006.md"
    second = tmp_path / "A2026008.md"
    first.write_text(issue(MASTHEAD, INDEX, DECREE_66), encoding="utf-8")
    second.write_text(issue(MASTHEAD, INDEX, DECREE_80), encoding="utf-8")

    summary = slice_folder([first, second], tmp_path / "slices")

    assert summary["files"] == 2
    assert summary["instruments"] == 2
    assert summary["eligible"] == 1
    rows = [json.loads(line) for line in (tmp_path / "slices" / "A2026006.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["law_number"] == "26-66"
    assert json.loads((tmp_path / "slices" / "summary.json").read_text(encoding="utf-8"))["files"] == 2
