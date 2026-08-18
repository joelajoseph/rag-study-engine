"""Split extracted PDF pages into overlapping, citation-friendly chunks."""

from dataclasses import asdict, dataclass
from math import ceil
from typing import Sequence


WORDS_PER_TOKEN_ESTIMATE = 1 / 1.3


@dataclass(frozen=True)
class Chunk:
    """One retrieval-sized passage, with enough metadata for citations."""

    chunk_text: str
    page_start: int
    page_end: int
    chunk_index: int
    token_count: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-ready representation for the inspection checkpoint."""
        return asdict(self)


def estimate_token_count(text: str) -> int:
    """Approximate token count without adding a model-specific tokenizer."""
    word_count = len(text.split())
    return ceil(word_count * 1.3)


def chunk_pages(
    pages: Sequence[tuple[int, str]],
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Create fixed-size overlapping chunks from ``(page_number, text)`` pairs.

    The target sizes are token estimates. Internally the function uses words,
    so it stays lightweight until the embedding model is introduced.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be at least 0 and smaller than max_tokens")

    words_per_chunk = max(1, int(max_tokens * WORDS_PER_TOKEN_ESTIMATE))
    overlap_words = int(overlap_tokens * WORDS_PER_TOKEN_ESTIMATE)
    step_size = words_per_chunk - overlap_words

    words_with_pages: list[tuple[int, str]] = []
    for page_number, page_text in pages:
        if page_number < 1:
            raise ValueError("page numbers must be 1-based positive integers")
        words_with_pages.extend((page_number, word) for word in page_text.split())

    chunks: list[Chunk] = []
    for start in range(0, len(words_with_pages), step_size):
        window = words_with_pages[start : start + words_per_chunk]
        if not window:
            break

        chunk_text = " ".join(word for _, word in window)
        chunks.append(
            Chunk(
                chunk_text=chunk_text,
                page_start=window[0][0],
                page_end=window[-1][0],
                chunk_index=len(chunks),
                token_count=estimate_token_count(chunk_text),
            )
        )

        if start + words_per_chunk >= len(words_with_pages):
            break

    return chunks
