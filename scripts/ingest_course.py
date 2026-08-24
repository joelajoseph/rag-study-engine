"""Ingest one PDF through all Stage 1 steps, ending with FAISS indexing."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.db import (
    get_connection,
    get_or_create_course,
    get_or_create_document,
    initialize_database,
    insert_chunks,
)
from src.ingest.chunk import Chunk, chunk_pages
from src.ingest.embed import embed_missing_chunks
from src.ingest.extract import extract_pdf


import re


def infer_chapter_from_filename(filename: str) -> str | None:
    """Attempt to infer a chapter identifier from a PDF filename."""
    stem = Path(filename).stem
    match = re.search(r"(?:chapter|ch)[_\s-]*([0-9a-zA-Z._-]+)", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    return None



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest one PDF into the study database.")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument(
        "--course-code",
        help="Defaults to the PDF's parent directory name.",
    )
    parser.add_argument("--course-name")
    parser.add_argument(
        "--source-type",
        required=True,
        choices=("lecture", "textbook", "tutorial", "notes"),
    )
    parser.add_argument("--week", type=int)
    parser.add_argument(
        "--chapter",
        help="Chapter identifier. If omitted, will attempt to infer from filename.",
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/db/study_engine.db"),
        help="SQLite database path (default: data/db/study_engine.db).",
    )
    parser.add_argument(
        "--index-directory",
        type=Path,
        default=Path("data/db/faiss_index"),
        help="FAISS index directory (default: data/db/faiss_index).",
    )
    return parser


def write_processed_json(
    *,
    pdf_path: Path,
    course_code: str,
    pages: list[tuple[int, str]],
    chunks: list[Chunk],
    max_tokens: int,
    overlap_tokens: int,
) -> Path:
    """Write the manual inspection checkpoint before changing SQLite."""
    output_path = Path("data") / "processed" / course_code / f"{pdf_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "source_pdf": str(pdf_path),
        "course_code": course_code,
        "extracted_page_count": len(pages),
        "chunking": {
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
        },
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    course_code = args.course_code or args.pdf_path.parent.name

    chapter = args.chapter
    if not chapter:
        inferred = infer_chapter_from_filename(args.pdf_path.name)
        if inferred:
            print(f"[Info] No --chapter provided. Inferred chapter '{inferred}' from '{args.pdf_path.name}'.")
            chapter = inferred
        else:
            print(
                f"[Error] Could not infer chapter from filename '{args.pdf_path.name}'. "
                f"Please specify --chapter explicitly (e.g. --chapter 1).",
                file=sys.stderr,
            )
            sys.exit(1)

    print("Stage 1/4: extracting PDF text and creating chunks...")
    pages = extract_pdf(args.pdf_path)
    chunks = chunk_pages(
        pages,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    processed_path = write_processed_json(
        pdf_path=args.pdf_path,
        course_code=course_code,
        pages=pages,
        chunks=chunks,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    print(f"Stage 2/4: wrote inspection JSON: {processed_path}")

    print("Stage 3/4: storing document and chunks in SQLite...")
    initialize_database(args.database)
    with get_connection(args.database) as connection:
        course_id = get_or_create_course(connection, course_code, args.course_name)
        document_id, was_created = get_or_create_document(
            connection,
            course_id=course_id,
            filename=args.pdf_path.name,
            source_type=args.source_type,
            week=args.week,
            chapter=chapter,
        )

        if not was_created:
            print(
                f"Skipped database insert: {args.pdf_path.name} is already "
                f"ingested for {course_code} (document_id={document_id})."
            )
        else:
            inserted_count = insert_chunks(connection, document_id, chunks)
            print(
                f"Inserted course {course_code}, document_id={document_id}, and "
                f"{inserted_count} unembedded chunk rows."
            )

    print("Stage 4/4: embedding all chunks missing FAISS IDs...")
    embedding_result = embed_missing_chunks(args.database, args.index_directory)
    print(f"Embedded and indexed {embedding_result.embedded_chunk_count} chunks.")


if __name__ == "__main__":
    main()
