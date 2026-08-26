"""Quiz generation from chapter-scoped course materials."""

from contextlib import closing
import json
from pathlib import Path
import re
from typing import Any

from src.db.db import get_chunks_by_chapter, get_connection
from src.generation.ollama_client import OllamaClient


# TODO: Tune threshold against real chapter token sizes once tested empirically.
# Placeholder context budget to fit comfortably within local LLM (e.g. qwen2.5:3b-instruct) context windows.
DEFAULT_MAX_QUIZ_TOKENS = 3000
DEFAULT_MAX_QUIZ_CHUNKS = 8


def format_chunk_page_range(chunk: dict[str, Any]) -> str:
    """Return a human-readable page string for a chunk."""
    p_start = chunk.get("page_start", 0)
    p_end = chunk.get("page_end", 0)
    if p_start == p_end:
        return f"page {p_start}"
    return f"pages {p_start}-{p_end}"


def select_quiz_chunks(
    chunks: list[dict[str, Any]],
    *,
    num_questions: int | None = None,
    max_tokens: int = DEFAULT_MAX_QUIZ_TOKENS,
    max_chunks: int = DEFAULT_MAX_QUIZ_CHUNKS,
) -> list[dict[str, Any]]:
    """Select representative chunks from a chapter without exceeding context budget.

    If num_questions is provided, targets min(num_questions, max_chunks, len(chunks))
    evenly sampled across the chapter to ensure each question is paired with a distinct page span.
    """
    if not chunks:
        return []

    total_tokens = sum(int(c.get("token_count") or 500) for c in chunks)
    total_count = len(chunks)

    # Determine desired chunk count
    if num_questions is not None and num_questions > 0:
        target_count = min(num_questions, max_chunks, total_count)
    else:
        target_count = min(max_chunks, total_count)

    if total_tokens <= max_tokens and total_count <= target_count:
        selected = chunks
    else:
        # Uniformly stride-sample across the chapter
        step = total_count / target_count
        selected_indices = [int(i * step) for i in range(target_count)]
        selected = [chunks[i] for i in selected_indices]

    selected_tokens = sum(int(c.get("token_count") or 500) for c in selected)
    included_excerpts = [
        f"{c.get('filename', 'doc')} ({format_chunk_page_range(c)})" for c in selected
    ]

    print(
        f"[Info] Chapter context: using {len(selected)}/{total_count} chunks "
        f"(~{selected_tokens}/~{total_tokens} estimated tokens)."
    )
    print(f"[Info] Included chapter excerpts: {', '.join(included_excerpts)}")

    return selected


def build_quiz_prompt(chunks: list[dict[str, Any]], num_questions: int = 5) -> str:
    """Build a prompt instructing the LLM to generate grounded quiz questions."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        filename = str(chunk.get("filename", "unknown"))
        page_str = format_chunk_page_range(chunk)
        chunk_text = str(chunk.get("chunk_text", "")).strip()
        context_blocks.append(f"[Excerpt {i}: {filename}, {page_str}]\n{chunk_text}")

    context_str = "\n\n".join(context_blocks)

    return (
        "You are an expert tutor creating a practice study quiz for a student.\n"
        f"Based ONLY on the provided course material excerpts below, generate exactly {len(chunks)} "
        "practice questions (one question per excerpt) that test core concepts, definitions, and mechanisms.\n\n"
        "Rules:\n"
        "- Every question and answer must be strictly grounded in its corresponding excerpt.\n"
        "- Do NOT introduce external facts or guess beyond what is explicitly stated.\n"
        "- Provide clear, concise answers.\n"
        "- Output your response as a single valid JSON array of objects with keys 'excerpt', 'question', and 'answer'.\n\n"
        "Required JSON Output Format:\n"
        "[\n"
        "  {\n"
        '    "excerpt": 1,\n'
        '    "question": "What is ...?",\n'
        '    "answer": "It is ..."\n'
        "  },\n"
        "  {\n"
        '    "excerpt": 2,\n'
        '    "question": "How do you ...?",\n'
        '    "answer": "By ..."\n'
        "  }\n"
        "]\n\n"
        f"--- Chapter Excerpts ---\n{context_str}\n\n"
        "JSON Response:"
    )


def extract_excerpt_index(item: dict[str, Any]) -> int | None:
    """Extract a 1-based excerpt index from a quiz item if specified by the LLM."""
    raw = item.get("excerpt")
    if raw is None and "source" in item:
        raw = item.get("source")

    if isinstance(raw, int):
        return raw

    if isinstance(raw, str):
        match = re.search(r"(?:excerpt\s*)?(\d+)", raw, re.IGNORECASE)
        if match:
            return int(match.group(1))

    if isinstance(raw, dict):
        if "excerpt" in raw and isinstance(raw["excerpt"], int):
            return raw["excerpt"]

    return None


def parse_quiz_response(
    raw_response: str,
    chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Parse and validate JSON response from the LLM into structured quiz items."""
    text = raw_response.strip()

    # Extract JSON if wrapped in markdown code fence
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Look for raw JSON array
        match_arr = re.search(r"(\[.*?\])", text, re.DOTALL)
        if match_arr:
            text = match_arr.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM quiz output as JSON: {exc}\nRaw output: {raw_response}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of quiz items, got: {type(data)}")

    chunk_list = chunks or []
    quiz_items = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if not question or not answer:
            continue

        # Deterministic source resolution via Python:
        # 1. Check if model provided a valid excerpt index
        excerpt_idx = extract_excerpt_index(item)
        if excerpt_idx is not None and 1 <= excerpt_idx <= len(chunk_list):
            target_chunk = chunk_list[excerpt_idx - 1]
        elif chunk_list:
            # 2. Positional fallback: question i maps to chunk i
            target_chunk = chunk_list[i % len(chunk_list)]
        else:
            target_chunk = {}

        source = {
            "filename": str(target_chunk.get("filename", "")),
            "page_start": int(target_chunk.get("page_start", 0)),
            "page_end": int(target_chunk.get("page_end", 0)),
        }

        quiz_items.append(
            {
                "question": question,
                "answer": answer,
                "source": source,
            }
        )

    return quiz_items


def generate_quiz(
    course_code: str,
    chapter: str,
    *,
    num_questions: int = 5,
    database_path: str | Path = "data/db/study_engine.db",
    client: OllamaClient | None = None,
    model: str | None = None,
    options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate a self-assessment quiz for a specific course chapter.

    Args:
        course_code: Course identifier (e.g. 'CSCA48').
        chapter: Chapter identifier (e.g. '1').
        num_questions: Number of questions to generate (default: 5).
        database_path: Path to SQLite database.
        client: Optional OllamaClient instance.
        model: Optional model override.
        options: Optional generation parameters.

    Returns:
        List of structured quiz items matching:
        [
            {
                "question": str,
                "answer": str,
                "source": {"filename": str, "page_start": int, "page_end": int}
            },
            ...
        ]
    """
    if not course_code.strip():
        raise ValueError("course_code must not be empty")
    if not str(chapter).strip():
        raise ValueError("chapter must not be empty")
    if num_questions <= 0:
        raise ValueError("num_questions must be positive")

    with closing(get_connection(database_path)) as connection:
        rows = get_chunks_by_chapter(connection, course_code, str(chapter))

    if not rows:
        raise ValueError(
            f"No chunks found for course '{course_code}', chapter '{chapter}' in {database_path}."
        )

    chunks = [dict(row) for row in rows]
    selected_chunks = select_quiz_chunks(chunks, num_questions=num_questions)

    prompt = build_quiz_prompt(selected_chunks, num_questions=num_questions)
    ollama_client = client or OllamaClient()

    gen_options = options if options is not None else {"temperature": 0.2}
    raw_response = ollama_client.generate(
        prompt=prompt,
        model=model,
        options=gen_options,
    )

    return parse_quiz_response(raw_response, chunks=selected_chunks)


