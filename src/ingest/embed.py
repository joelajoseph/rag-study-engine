"""Embed unindexed chunks and connect their vectors to SQLite through FAISS IDs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.db.db import (
    get_chunks_missing_faiss_id,
    get_connection,
    set_chunk_faiss_id,
)
from src.retrieval.index_store import IndexStore


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class EmbeddingResult:
    """A small summary for the ingestion script's embedding stage."""

    embedded_chunk_count: int


def embed_missing_chunks(
    database_path: str | Path,
    index_directory: str | Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingResult:
    """Embed every chunk whose SQLite ``faiss_id`` is still ``NULL``.

    Texts are encoded as one batch for speed. Each resulting vector is then
    added individually, followed immediately by its SQLite ``faiss_id``
    write-back. The index is saved before the database transaction commits,
    so a failed run can leave an extra vector but never a persisted pointer to
    a vector that was not saved.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    with get_connection(database_path) as connection:
        chunks = get_chunks_missing_faiss_id(connection)
        if not chunks:
            return EmbeddingResult(embedded_chunk_count=0)

        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except OSError as error:
            raise RuntimeError(
                f"Embedding model {EMBEDDING_MODEL_NAME!r} is not available locally. "
                "Download it once while online, then rerun ingestion."
            ) from error
        model_dimension = model.get_embedding_dimension()
        if model_dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"{EMBEDDING_MODEL_NAME} produced {model_dimension} dimensions; "
                f"expected {EMBEDDING_DIMENSION}."
            )

        vectors = model.encode(
            [chunk["chunk_text"] for chunk in chunks],
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        index_store = IndexStore(
            index_directory,
            model_name=EMBEDDING_MODEL_NAME,
            dimension=EMBEDDING_DIMENSION,
        )
        index_store.load()

        for chunk, vector in zip(chunks, vectors, strict=True):
            faiss_id = index_store.add(vector)[0]
            set_chunk_faiss_id(connection, int(chunk["chunk_id"]), faiss_id)

        index_store.save()

    return EmbeddingResult(embedded_chunk_count=len(chunks))
