"""PDF page extraction, Arabic text repair, image coverage, and rendering."""

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

# PyMuPDF is not thread-safe, so every call into the library is serialised.
# PDF work is cheap compared with OCR, so this costs almost nothing in practice.
_PDF_LOCK = threading.RLock()

_ARABIC = re.compile(r"[\u0600-\u06ff]")
_ALEF_FORMS = frozenset("\u0627\u0623\u0625\u0622\u0671")  # ا أ إ آ ٱ
_LAM = "\u0644"


@dataclass
class ExtractedPage:
    number: int
    text: str
    image_ratio: float
    image_bytes: bytes


def _image_coverage(page: fitz.Page) -> float:
    """Estimate image coverage by summing clipped image rectangles."""
    page_rect = page.rect
    page_area = max(1.0, page_rect.width * page_rect.height)
    area = 0.0
    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            clipped = rect & page_rect
            if not clipped.is_empty:
                area += clipped.width * clipped.height
    return min(area / page_area, 1.0)


def _ligature_origins(page: fitz.Page) -> set[tuple[float, float]]:
    """Origins where MuPDF synthesised a lam-alef half after the ligature glyph."""
    with _PDF_LOCK:
        trace = page.get_texttrace()
    origins: set[tuple[float, float]] = set()
    for span in trace:
        for char in span["chars"]:
            if char[1] == -1 and chr(char[0]) in _ALEF_FORMS:
                origins.add((round(char[2][0], 1), round(char[2][1], 1)))
    return origins


def _fix_ligature_pairs(chars: list[dict], origins: set[tuple[float, float]]) -> None:
    """Swap alef-then-lam pairs that MuPDF emits reversed inside a lam-alef ligature."""
    for index, char in enumerate(chars):
        origin = (round(char["origin"][0], 1), round(char["origin"][1], 1))
        if (
            index
            and char["c"] == _LAM
            and origin in origins
            and chars[index - 1]["c"] in _ALEF_FORMS
        ):
            chars[index - 1], chars[index] = chars[index], chars[index - 1]


def _page_text(page: fitz.Page) -> str:
    """Extract text and repair Arabic lam-alef ligatures that come out reversed.

    PyMuPDF flips each Arabic span from the PDF's visual order to logical order.
    A lam-alef ligature glyph decomposes into its two letters in *logical* order
    before that flip, so the pair ends up swapped: "أولا" is extracted as
    "أوال" and "الإدماج" as "اإلدماج". The glyph trace marks the synthesised
    half of each ligature, which lets us swap exactly those pairs back without
    touching ordinary "ال" sequences.
    """
    with _PDF_LOCK:
        text = page.get_text("text")
    if not _ARABIC.search(text):
        return text
    origins = _ligature_origins(page)
    if not origins:
        return text

    with _PDF_LOCK:
        blocks = page.get_text("rawdict")["blocks"]
    lines: list[str] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            chars = [char for span in line["spans"] for char in span["chars"]]
            _fix_ligature_pairs(chars, origins)
            lines.append("".join(char["c"] for char in chars))
    return "\n".join(lines) + ("\n" if lines else "")


class PdfPageSource:
    """One open PDF with thread-safe text extraction and lazy page rendering."""

    def __init__(self, path: Path, dpi: int):
        self.path = Path(path)
        self.dpi = dpi
        with _PDF_LOCK:
            self._document = fitz.open(self.path)
            if self._document.is_encrypted:
                self._document.close()
                raise RuntimeError(f"PDF is password-protected: {self.path.name}")
            self._matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    @property
    def page_count(self) -> int:
        with _PDF_LOCK:
            return self._document.page_count

    def pages(self):
        """Yield (page number, text, image ratio) in document order."""
        for index in range(self.page_count):
            with _PDF_LOCK:
                page = self._document[index]
                text = _page_text(page)
                image_ratio = _image_coverage(page)
            yield index + 1, text, image_ratio

    def render(self, number: int) -> bytes:
        """Render one page to PNG; safe to call from worker threads."""
        with _PDF_LOCK:
            pixmap = self._document[number - 1].get_pixmap(matrix=self._matrix, alpha=False)
            return pixmap.tobytes("png")

    def close(self) -> None:
        with _PDF_LOCK:
            self._document.close()

    def __enter__(self) -> "PdfPageSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def extract_pdf_pages(path: Path, dpi: int) -> list[ExtractedPage]:
    """Extract text and a rendered PNG for every page in original order."""
    with PdfPageSource(path, dpi) as source:
        return [
            ExtractedPage(number, text, image_ratio, source.render(number))
            for number, text, image_ratio in source.pages()
        ]


def extract_text_file(path: Path) -> str:
    """Read plain text or Markdown with BOM-aware UTF-8 decoding."""
    return path.read_text(encoding="utf-8-sig", errors="replace")


def extract_docx(path: Path) -> str:
    """Extract paragraphs and table rows from a Word document."""
    from docx import Document

    document = Document(path)
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)
