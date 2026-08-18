"""Inspect PDF extraction and chunking before storing anything in SQLite.

Example:
    venv/bin/python scripts/inspect_ingestion.py \
        data/raw/CS2110/lecture_01.pdf --show-pages --show-chunks
"""

import argparse
import json
from pathlib import Path
import sys


# Running ``python scripts/inspect_ingestion.py`` makes ``scripts/`` the
# import root. Add the project root so imports from ``src/`` work consistently.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.chunk import Chunk, chunk_pages
from src.ingest.extract import extract_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a PDF, chunk its text, and save an inspectable JSON file."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF in data/raw/<course_code>/")
    parser.add_argument(
        "--course-code",
        help="Output course folder. Defaults to the PDF's parent folder name.",
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument(
        "--show-pages",
        action="store_true",
        help="Print the first 300 characters extracted from every non-empty page.",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print the first 300 characters of every generated chunk.",
    )
    return parser


def print_pages(pages: list[tuple[int, str]]) -> None:
    for page_number, page_text in pages:
        preview = page_text[:300].replace("\n", " ")
        suffix = "..." if len(page_text) > 300 else ""
        print(f"\nPAGE {page_number} ({len(page_text.split())} words)\n{preview}{suffix}")


def print_chunks(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        preview = chunk.chunk_text[:300]
        suffix = "..." if len(chunk.chunk_text) > 300 else ""
        print(
            f"\nCHUNK {chunk.chunk_index} "
            f"(pages {chunk.page_start}-{chunk.page_end}, "
            f"~{chunk.token_count} tokens)\n{preview}{suffix}"
        )


def main() -> None:
    args = build_parser().parse_args()
    pages = extract_pdf(args.pdf_path)
    chunks = chunk_pages(
        pages,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    course_code = args.course_code or args.pdf_path.parent.name
    output_path = (
        Path("data") / "processed" / course_code / f"{args.pdf_path.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "source_pdf": str(args.pdf_path),
        "course_code": course_code,
        "extracted_page_count": len(pages),
        "chunking": {
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens,
        },
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Extracted {len(pages)} non-empty pages and created {len(chunks)} chunks.")
    print(f"Wrote inspection JSON: {output_path}")

    if args.show_pages:
        print_pages(pages)
    if args.show_chunks:
        print_chunks(chunks)


if __name__ == "__main__":
    main()
