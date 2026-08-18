"""Extract text from PDFs while keeping the page each text came from."""

from pathlib import Path

import pymupdf


def extract_pdf(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Return non-empty text for each page in *pdf_path*.

    Page numbers are 1-based, matching how a student reads a PDF and how
    citations will be displayed later in the pipeline.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.name}")

    pages: list[tuple[int, str]] = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                pages.append((page_number, page_text))

    return pages
