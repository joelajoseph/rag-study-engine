"""Recreate the FAISS index from the chunks currently stored in SQLite."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.db import clear_all_faiss_ids, get_connection, initialize_database
from src.ingest.embed import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    embed_missing_chunks,
)
from src.retrieval.index_store import IndexStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the FAISS index from SQLite chunks.")
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


def main() -> None:
    args = build_parser().parse_args()
    initialize_database(args.database)

    with get_connection(args.database) as connection:
        cleared_count = clear_all_faiss_ids(connection)
    print(f"Cleared {cleared_count} SQLite FAISS links.")

    index_store = IndexStore(
        args.index_directory,
        model_name=EMBEDDING_MODEL_NAME,
        dimension=EMBEDDING_DIMENSION,
    )
    index_store.reset()
    print("Created an empty FAISS index.")

    result = embed_missing_chunks(args.database, args.index_directory)
    print(f"Re-embedded and indexed {result.embedded_chunk_count} chunks.")


if __name__ == "__main__":
    main()
