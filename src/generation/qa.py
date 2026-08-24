"""Prompt construction and QA orchestration for course material RAG."""

from pathlib import Path
from typing import Any

from src.generation.ollama_client import OllamaClient
from src.retrieval.search import DEFAULT_RESULT_COUNT, search_chunks


DEFAULT_DISTANCE_THRESHOLD = 1.5


def format_source_label(filename: str, page_start: int, page_end: int) -> str:
    """Format a human-readable page/source tag."""
    if page_start == page_end:
        return f"{filename}, page {page_start}"
    return f"{filename}, pages {page_start}-{page_end}"


def format_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format structured source metadata matching the blueprint response shape."""
    sources: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "label": f"Source {i}",
                "filename": str(chunk.get("filename", "")),
                "page_start": int(chunk.get("page_start", 0)),
                "page_end": int(chunk.get("page_end", 0)),
            }
        )
    return sources


def build_qa_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Assemble a grounded QA prompt with labeled source context and citation rules.

    Args:
        question: The user's question.
        chunks: List of retrieved chunk dictionaries.

    Returns:
        The formatted prompt string for the LLM.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        filename = str(chunk.get("filename", "unknown"))
        p_start = int(chunk.get("page_start", 0))
        p_end = int(chunk.get("page_end", 0))
        label_info = format_source_label(filename, p_start, p_end)
        chunk_text = str(chunk.get("chunk_text", "")).strip()
        context_blocks.append(f"[Source {i}: {label_info}]\n{chunk_text}")

    context_str = "\n\n".join(context_blocks)

    return (
        "You are a helpful study assistant. Answer the user's question using ONLY the provided course material excerpts below.\n\n"
        "Guidelines:\n"
        "- Base your answer strictly on the provided context excerpts.\n"
        "- Cite sources in your answer using the source tags (e.g. [Source 1], [Source 2]).\n"
        "- If the provided context does not contain enough information to answer the question, explicitly state: "
        '"I do not have enough information in the provided course materials to answer this question."\n'
        "- Do not extrapolate or guess beyond what is directly stated in the context.\n\n"
        f"--- Context Excerpts ---\n{context_str}\n\n"
        f"--- Question ---\n{question.strip()}\n\n"
        "Answer:"
    )


def answer_question(
    question: str,
    course_code: str,
    database_path: str | Path = "data/db/study_engine.db",
    index_directory: str | Path = "data/db/faiss_index",
    *,
    chapter: str | None = None,
    k: int = DEFAULT_RESULT_COUNT,
    distance_threshold: float | None = DEFAULT_DISTANCE_THRESHOLD,
    client: OllamaClient | None = None,
    model: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve relevant chunks and generate a grounded, cited answer.

    Args:
        question: User query.
        course_code: Course code identifier (e.g. 'CSCA48').
        database_path: Path to SQLite database.
        index_directory: Path to FAISS index directory.
        chapter: Optional chapter identifier to scope search results.
        k: Maximum number of chunks to retrieve.
        distance_threshold: Maximum FAISS L2 distance allowed for a chunk to be considered relevant.
            If all chunks exceed this distance or if no chunks are found, the LLM call is skipped.
        client: Optional configured OllamaClient instance.
        model: Optional model name override.
        options: Optional generation options (e.g. {"temperature": 0.0}).

    Returns:
        Structured dictionary containing:
        {
            "answer": str,
            "sources": [
                {"label": "Source 1", "filename": "...", "page_start": 1, "page_end": 1},
                ...
            ]
        }
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if not course_code.strip():
        raise ValueError("course_code must not be empty")

    retrieved_chunks = search_chunks(
        query=question,
        course_code=course_code,
        database_path=database_path,
        index_directory=index_directory,
        chapter=chapter,
        k=k,
    )


    # Filter by distance threshold if provided
    if distance_threshold is not None:
        relevant_chunks = [
            c
            for c in retrieved_chunks
            if float(c.get("distance", float("inf"))) <= distance_threshold
        ]
    else:
        relevant_chunks = retrieved_chunks

    # Guardrail: skip LLM call if no relevant context exists
    if not relevant_chunks:
        return {
            "answer": "I do not have enough information in the provided course materials to answer this question.",
            "sources": [],
        }

    prompt = build_qa_prompt(question, relevant_chunks)
    sources = format_sources(relevant_chunks)

    ollama_client = client or OllamaClient()
    # Default to temperature 0.0 for deterministic, grounded answers if not specified
    gen_options = options if options is not None else {"temperature": 0.0}

    answer_text = ollama_client.generate(
        prompt=prompt,
        model=model,
        options=gen_options,
    )

    return {
        "answer": answer_text,
        "sources": sources,
    }
