"""The single entry point for SQLite access in the study engine."""

from pathlib import Path
import sqlite3
from typing import Iterable

from src.ingest.chunk import Chunk


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with foreign-key enforcement enabled."""
    connection = sqlite3.connect(Path(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path) -> None:
    """Create the database and its tables if they do not already exist."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(path) as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def get_or_create_course(
    connection: sqlite3.Connection,
    course_code: str,
    course_name: str | None = None,
) -> int:
    """Return the ID for a course, creating it when it is first seen."""
    connection.execute(
        """
        INSERT INTO courses (course_code, course_name)
        VALUES (?, ?)
        ON CONFLICT(course_code) DO NOTHING
        """,
        (course_code, course_name),
    )
    row = connection.execute(
        "SELECT course_id FROM courses WHERE course_code = ?", (course_code,)
    ).fetchone()
    assert row is not None
    return int(row["course_id"])


def get_document(
    connection: sqlite3.Connection, course_id: int, filename: str
) -> sqlite3.Row | None:
    """Look up a document by the idempotency key from the schema."""
    return connection.execute(
        "SELECT * FROM documents WHERE course_id = ? AND filename = ?",
        (course_id, filename),
    ).fetchone()


def create_document(
    connection: sqlite3.Connection,
    *,
    course_id: int,
    filename: str,
    source_type: str,
    week: int | None = None,
    chapter: str | None = None,
) -> int:
    """Return one PDF's ID without duplicating an existing document row."""
    document_id, _ = get_or_create_document(
        connection,
        course_id=course_id,
        filename=filename,
        source_type=source_type,
        week=week,
        chapter=chapter,
    )
    return document_id


def get_or_create_document(
    connection: sqlite3.Connection,
    *,
    course_id: int,
    filename: str,
    source_type: str,
    week: int | None = None,
    chapter: str | None = None,
) -> tuple[int, bool]:
    """Return a document ID and whether this call created it.

    The ``(course_id, filename)`` unique constraint is the ingestion
    idempotency key. Existing metadata is intentionally left unchanged:
    re-ingesting a PDF means "skip" rather than silently overwriting it.
    """
    cursor = connection.execute(
        """
        INSERT INTO documents (course_id, filename, source_type, week, chapter)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(course_id, filename) DO NOTHING
        """,
        (course_id, filename, source_type, week, chapter),
    )
    row = get_document(connection, course_id, filename)
    assert row is not None
    return int(row["document_id"]), cursor.rowcount == 1


def insert_chunks(
    connection: sqlite3.Connection,
    document_id: int,
    chunks: Iterable[Chunk],
) -> int:
    """Store unembedded chunks once, returning how many rows were inserted.

    A document's chunks are treated as one immutable ingestion result. The
    orchestration script only calls this for a new document; this guard also
    prevents accidental duplicate chunk rows if the helper is called again.
    """
    existing_chunk = connection.execute(
        "SELECT 1 FROM chunks WHERE document_id = ? LIMIT 1", (document_id,)
    ).fetchone()
    if existing_chunk is not None:
        return 0

    rows = [
        (
            document_id,
            chunk.chunk_text,
            chunk.page_start,
            chunk.page_end,
            chunk.chunk_index,
            chunk.token_count,
        )
        for chunk in chunks
    ]
    connection.executemany(
        """
        INSERT INTO chunks
            (document_id, chunk_text, page_start, page_end, chunk_index, token_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def get_chunks_missing_faiss_id(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return every chunk that has not yet been added to the FAISS index."""
    return connection.execute(
        """
        SELECT chunk_id, chunk_text
        FROM chunks
        WHERE faiss_id IS NULL
        ORDER BY chunk_id
        """
    ).fetchall()


def set_chunk_faiss_id(
    connection: sqlite3.Connection, chunk_id: int, faiss_id: int
) -> None:
    """Link an already-added FAISS vector position back to its chunk row."""
    cursor = connection.execute(
        """
        UPDATE chunks
        SET faiss_id = ?
        WHERE chunk_id = ? AND faiss_id IS NULL
        """,
        (faiss_id, chunk_id),
    )
    if cursor.rowcount != 1:
        raise RuntimeError(
            f"Could not assign FAISS ID {faiss_id} to unembedded chunk {chunk_id}."
        )


def clear_all_faiss_ids(connection: sqlite3.Connection) -> int:
    """Remove every SQLite-to-FAISS link before rebuilding the whole index."""
    cursor = connection.execute("UPDATE chunks SET faiss_id = NULL WHERE faiss_id IS NOT NULL")
    return cursor.rowcount


def get_chunk_by_faiss_id(
    connection: sqlite3.Connection, faiss_id: int
) -> sqlite3.Row | None:
    """Resolve one FAISS result position to the chunk and document metadata."""
    return connection.execute(
        """
        SELECT
            chunks.chunk_text,
            chunks.page_start,
            chunks.page_end,
            documents.filename,
            documents.source_type,
            documents.chapter,
            courses.course_code
        FROM chunks
        JOIN documents ON documents.document_id = chunks.document_id
        JOIN courses ON courses.course_id = documents.course_id
        WHERE chunks.faiss_id = ?
        """,
        (faiss_id,),
    ).fetchone()


def get_chunks_by_chapter(
    connection: sqlite3.Connection, course_code: str, chapter: str
) -> list[sqlite3.Row]:
    """Return all chunks belonging to a specific course and chapter, ordered sequentially."""
    return connection.execute(
        """
        SELECT
            chunks.chunk_id,
            chunks.chunk_text,
            chunks.page_start,
            chunks.page_end,
            chunks.chunk_index,
            chunks.token_count,
            documents.document_id,
            documents.filename,
            documents.source_type,
            documents.chapter,
            courses.course_code
        FROM chunks
        JOIN documents ON documents.document_id = chunks.document_id
        JOIN courses ON courses.course_id = documents.course_id
        WHERE courses.course_code = ? AND documents.chapter = ?
        ORDER BY documents.document_id, chunks.chunk_index
        """,
        (course_code, str(chapter)),
    ).fetchall()


def get_chapters_for_course(
    connection: sqlite3.Connection, course_code: str
) -> list[str]:
    """Return all distinct non-null chapter names available for a course."""
    rows = connection.execute(
        """
        SELECT DISTINCT documents.chapter
        FROM documents
        JOIN courses ON courses.course_id = documents.course_id
        WHERE courses.course_code = ? AND documents.chapter IS NOT NULL
        ORDER BY documents.chapter
        """,
        (course_code,),
    ).fetchall()
    return [str(row["chapter"]) for row in rows]


def record_quiz_attempt(
    connection: sqlite3.Connection,
    *,
    document_id: int,
    chapter: str,
    question_text: str,
    model_answer: str,
    self_correct: str,
    confidence: int,
    topic: str | None = None,
    source_filename: str | None = None,
    source_page_start: int | None = None,
    source_page_end: int | None = None,
    attempted_at: str | None = None,
) -> int:
    """Insert a single practice quiz attempt and return its attempt_id."""
    valid_correctness = {"correct", "partial", "incorrect"}
    if self_correct not in valid_correctness:
        raise ValueError(
            f"Invalid self_correct value '{self_correct}'. Must be one of {valid_correctness}."
        )
    if not (1 <= confidence <= 5):
        raise ValueError(f"Confidence rating must be between 1 and 5 (got {confidence}).")

    if attempted_at is not None:
        cursor = connection.execute(
            """
            INSERT INTO quiz_attempts (
                document_id, chapter, topic, question_text, model_answer,
                self_correct, confidence, source_filename, source_page_start,
                source_page_end, attempted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                str(chapter),
                topic,
                question_text,
                model_answer,
                self_correct,
                confidence,
                source_filename,
                source_page_start,
                source_page_end,
                attempted_at,
            ),
        )
    else:
        cursor = connection.execute(
            """
            INSERT INTO quiz_attempts (
                document_id, chapter, topic, question_text, model_answer,
                self_correct, confidence, source_filename, source_page_start,
                source_page_end
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                str(chapter),
                topic,
                question_text,
                model_answer,
                self_correct,
                confidence,
                source_filename,
                source_page_start,
                source_page_end,
            ),
        )
    return int(cursor.lastrowid)


def get_attempts_by_chapter(
    connection: sqlite3.Connection, course_code: str, chapter: str
) -> list[sqlite3.Row]:
    """Return all quiz attempts for a course and chapter, ordered chronologically."""
    return connection.execute(
        """
        SELECT
            quiz_attempts.attempt_id,
            quiz_attempts.document_id,
            quiz_attempts.chapter,
            quiz_attempts.topic,
            quiz_attempts.question_text,
            quiz_attempts.model_answer,
            quiz_attempts.self_correct,
            quiz_attempts.confidence,
            quiz_attempts.source_filename,
            quiz_attempts.source_page_start,
            quiz_attempts.source_page_end,
            quiz_attempts.attempted_at,
            courses.course_code
        FROM quiz_attempts
        JOIN documents ON documents.document_id = quiz_attempts.document_id
        JOIN courses ON courses.course_id = documents.course_id
        WHERE courses.course_code = ? AND quiz_attempts.chapter = ?
        ORDER BY quiz_attempts.attempted_at ASC, quiz_attempts.attempt_id ASC
        """,
        (course_code, str(chapter)),
    ).fetchall()


def get_attempts_summary(
    connection: sqlite3.Connection, course_code: str
) -> list[sqlite3.Row]:
    """Return all quiz attempts for a course across chapters, ordered newest first."""
    return connection.execute(
        """
        SELECT
            quiz_attempts.attempt_id,
            quiz_attempts.document_id,
            quiz_attempts.chapter,
            quiz_attempts.topic,
            quiz_attempts.question_text,
            quiz_attempts.model_answer,
            quiz_attempts.self_correct,
            quiz_attempts.confidence,
            quiz_attempts.source_filename,
            quiz_attempts.source_page_start,
            quiz_attempts.source_page_end,
            quiz_attempts.attempted_at,
            courses.course_code
        FROM quiz_attempts
        JOIN documents ON documents.document_id = quiz_attempts.document_id
        JOIN courses ON courses.course_id = documents.course_id
        WHERE courses.course_code = ?
        ORDER BY quiz_attempts.attempted_at DESC, quiz_attempts.attempt_id DESC
        """,
        (course_code,),
    ).fetchall()

