"""PDF page extraction, image coverage measurement, and rendering."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz


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


def extract_pdf_pages(path: Path, dpi: int) -> list[ExtractedPage]:
    """Extract text and a rendered PNG for every page in original order."""
    pages: list[ExtractedPage] = []
    with fitz.open(path) as document:
        if document.is_encrypted:
            raise RuntimeError(f"PDF is password-protected: {path.name}")
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        for index, page in enumerate(document):
            text = page.get_text("text")
            image_ratio = _image_coverage(page)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(ExtractedPage(
                number=index + 1,
                text=text,
                image_ratio=image_ratio,
                image_bytes=pixmap.tobytes("png"),
            ))
    return pages


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
