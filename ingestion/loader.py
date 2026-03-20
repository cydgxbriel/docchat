from pathlib import Path
import pdfplumber


def load_pdf(file_path: str) -> list[dict]:
    """Load a PDF and return list of dicts with text, page number, and source."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "text": text,
                    "page": i + 1,
                    "source": str(file_path),
                })
    return pages
