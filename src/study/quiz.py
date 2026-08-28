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


# Stopwords for topic keyword-grounding check
STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves",
}

# Explicit placeholder topic strings that should never be saved
REJECTED_TOPIC_PLACEHOLDERS: set[str] = {
    "<short topic label>",
    "short topic label",
    "concept name",
    "another concept",
    "topic label",
    "topic",
    "none",
    "n/a",
    "null",
}


def is_topic_grounded(topic: str, excerpt_text: str) -> bool:
    """Check whether a topic has meaningful keyword overlap with excerpt text (case-insensitive)."""
    if not topic or not excerpt_text:
        return False

    cleaned_topic = topic.strip().lower()
    if cleaned_topic in REJECTED_TOPIC_PLACEHOLDERS or (
        cleaned_topic.startswith("<") and cleaned_topic.endswith(">")
    ):
        return False

    # Extract alphanumeric tokens of length >= 2
    topic_tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", cleaned_topic)
    content_tokens = [t for t in topic_tokens if t not in STOPWORDS and len(t) >= 2]
    if not content_tokens:
        return False

    excerpt_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", excerpt_text.lower()))

    for token in content_tokens:
        if token in excerpt_tokens:
            return True
        # Match singular/plural or stem variations (e.g. pointer vs pointers)
        if len(token) >= 3:
            stem = token.rstrip("s")
            if any(et.startswith(stem) or stem in et for et in excerpt_tokens if len(et) >= 3):
                return True

    return False


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
        "- Provide a short, informal topic label (1-4 words) describing the concept tested.\n"
        "- The topic must name only what this specific excerpt discusses. Do not use general CS curriculum terms (e.g. common textbook topics) unless the underlying concept is actually present in the excerpt text.\n"
        "- Do NOT copy or reuse placeholder text from the format example (do not output '<short topic label>' or 'Concept Name').\n"
        "- Every question must be completely self-contained. If asking about a code snippet, table, or specific syntax from the excerpt, reproduce that code snippet or text directly inside the question string. Do NOT refer to 'the code above', 'the code below', or 'the excerpt' without including the actual code or content in the question.\n"
        "- Provide clear, concise answers.\n"
        "- Output your response as a single valid JSON array of objects with keys 'excerpt', 'topic', 'question', and 'answer'.\n\n"
        "Required JSON Output Format:\n"
        "[\n"
        "  {\n"
        '    "excerpt": 1,\n'
        '    "topic": "<short topic label>",\n'
        '    "question": "What is ...?",\n'
        '    "answer": "It is ..."\n'
        "  },\n"
        "  {\n"
        '    "excerpt": 2,\n'
        '    "topic": "<short topic label>",\n'
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

        # Informal topic parsing (soft-failure: keep question if topic is missing, placeholder, or invalid)
        raw_topic = item.get("topic")
        topic = raw_topic.strip() if isinstance(raw_topic, str) and raw_topic.strip() else None

        if topic:
            cleaned_topic = topic.lower()
            if cleaned_topic in REJECTED_TOPIC_PLACEHOLDERS or (
                cleaned_topic.startswith("<") and cleaned_topic.endswith(">")
            ):
                topic = None

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

        # Keyword-overlap backstop: verify topic is grounded in the source chunk text
        chunk_text = str(target_chunk.get("chunk_text", "")).strip()
        if topic and chunk_text and not is_topic_grounded(topic, chunk_text):
            topic = None

        source = {
            "document_id": target_chunk.get("document_id"),
            "filename": str(target_chunk.get("filename", "")),
            "page_start": int(target_chunk.get("page_start", 0)),
            "page_end": int(target_chunk.get("page_end", 0)),
        }

        quiz_items.append(
            {
                "question": question,
                "answer": answer,
                "topic": topic,
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


