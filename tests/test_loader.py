import os
import tempfile
from fpdf import FPDF
from ingestion.loader import load_pdf


def _create_test_pdf(text_pages: list[str]) -> str:
    """Create a temporary PDF with given text per page."""
    pdf = FPDF()
    for text in text_pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)
    path = tempfile.mktemp(suffix=".pdf")
    pdf.output(path)
    return path


def test_load_pdf_returns_pages_with_text_and_metadata():
    path = _create_test_pdf(["Page one content.", "Page two content."])
    try:
        pages = load_pdf(path)
        assert len(pages) == 2
        assert "Page one content" in pages[0]["text"]
        assert pages[0]["page"] == 1
        assert pages[1]["page"] == 2
        assert pages[0]["source"] == path
    finally:
        os.unlink(path)


def test_load_pdf_raises_on_missing_file():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_pdf("/nonexistent/path.pdf")
