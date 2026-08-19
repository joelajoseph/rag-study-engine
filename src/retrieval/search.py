"""Search the shared FAISS index and resolve results to study-material metadata."""

from contextlib import closing
from pathlib import Path

import numpy as np

from src.db.db import get_chunk_by_faiss_id, get_connection
from src.ingest.embed import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    load_embedding_model,
)
from src.retrieval.index_store import IndexStore


DEFAULT_RESULT_COUNT = 5
DEFAULT_CANDIDATE_COUNT = 20


def search_chunks(
    query: str,
    course_code: str,
    database_path: str | Path = "data/db/study_engine.db",
    index_directory: str | Path = "data/db/faiss_index",
    *,
    k: int = DEFAULT_RESULT_COUNT,
    candidate_k: int = DEFAULT_CANDIDATE_COUNT,
) -> list[dict[str, str | int | float]]:
    """Return the nearest indexed chunks that belong to ``course_code``.

    FAISS is searched globally for extra candidates. SQLite then resolves each
    FAISS position and filters course membership, preserving FAISS's original
    nearest-first order.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    if candidate_k < k:
        raise ValueError("candidate_k must be at least k")

    model = load_embedding_model()
    query_vector = model.encode(
        [query], convert_to_numpy=True, show_progress_bar=False
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)

    index_store = IndexStore(
        index_directory,
        model_name=EMBEDDING_MODEL_NAME,
        dimension=EMBEDDING_DIMENSION,
    )
    index_store.load()
    distances, faiss_ids = index_store.search(query_vector, candidate_k)

    results: list[dict[str, str | int | float]] = []
    with closing(get_connection(database_path)) as connection:
        for distance, faiss_id in zip(distances[0], faiss_ids[0], strict=True):
            if faiss_id < 0:
                continue

            chunk = get_chunk_by_faiss_id(connection, int(faiss_id))
            if chunk is None or chunk["course_code"] != course_code:
                continue

            results.append(
                {
                    "chunk_text": str(chunk["chunk_text"]),
                    "filename": str(chunk["filename"]),
                    "page_start": int(chunk["page_start"]),
                    "page_end": int(chunk["page_end"]),
                    "source_type": str(chunk["source_type"]),
                    "distance": float(distance),
                }
            )
            if len(results) == k:
                break

    return results
