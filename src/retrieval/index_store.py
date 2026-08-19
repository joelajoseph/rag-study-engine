"""The single place where the project creates, loads, and queries FAISS."""

import json
from pathlib import Path

import faiss
import numpy as np


class IndexStore:
    """Persist an exact L2 FAISS index together with its embedding contract."""

    def __init__(self, index_directory: str | Path, model_name: str, dimension: int):
        self.index_directory = Path(index_directory)
        self.index_path = self.index_directory / "index.faiss"
        self.config_path = self.index_directory / "config.json"
        self.model_name = model_name
        self.dimension = dimension
        self.index: faiss.Index | None = None

    def load(self) -> None:
        """Load an existing index or create an empty compatible one.

        A persisted index must have matching model and dimension metadata. A
        mismatch raises instead of returning meaningless similarity results.
        """
        self.index_directory.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and not self.config_path.exists():
            raise RuntimeError(
                f"FAISS index exists at {self.index_path}, but config.json is missing. "
                "Rebuild the index rather than guessing its embedding model."
            )

        if self.config_path.exists():
            self._validate_config()
        else:
            self._write_config()

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if self.index.d != self.dimension:
                raise RuntimeError(
                    f"FAISS index dimension is {self.index.d}, but the requested model "
                    f"uses {self.dimension}. Rebuild the index."
                )
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def save(self) -> None:
        """Persist the in-memory index and its checked embedding metadata."""
        index = self._require_loaded_index()
        self._write_config()
        faiss.write_index(index, str(self.index_path))

    def reset(self) -> None:
        """Replace any existing vectors with an empty compatible L2 index."""
        self.index_directory.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.save()

    def add(self, vectors: np.ndarray) -> list[int]:
        """Add float32 vectors and return their assigned FAISS positions."""
        index = self._require_loaded_index()
        prepared_vectors = self._prepare_vectors(vectors)
        start_id = index.ntotal
        index.add(prepared_vectors)
        return list(range(start_id, start_id + len(prepared_vectors)))

    def search(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return L2 distances and FAISS positions for the nearest ``k`` vectors."""
        if k <= 0:
            raise ValueError("k must be positive")
        index = self._require_loaded_index()
        prepared_query = self._prepare_vectors(query_vector)
        if len(prepared_query) != 1:
            raise ValueError("search expects exactly one query vector")
        return index.search(prepared_query, k)

    def _validate_config(self) -> None:
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid FAISS config file: {self.config_path}") from error

        configured_model = config.get("embedding_model")
        configured_dimension = config.get("dimension")
        if configured_model != self.model_name or configured_dimension != self.dimension:
            raise RuntimeError(
                "FAISS index embedding configuration does not match the requested model. "
                f"Configured: {configured_model!r}, {configured_dimension!r} dimensions; "
                f"requested: {self.model_name!r}, {self.dimension} dimensions. "
                "Rebuild the index before continuing."
            )

    def _write_config(self) -> None:
        config = {"embedding_model": self.model_name, "dimension": self.dimension}
        self.config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def _require_loaded_index(self) -> faiss.Index:
        if self.index is None:
            raise RuntimeError("Call load() before using the FAISS index.")
        return self.index

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        prepared_vectors = np.asarray(vectors, dtype=np.float32)
        if prepared_vectors.ndim == 1:
            prepared_vectors = prepared_vectors.reshape(1, -1)
        if prepared_vectors.ndim != 2 or prepared_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors shaped (n, {self.dimension}), got {prepared_vectors.shape}."
            )
        return np.ascontiguousarray(prepared_vectors)
