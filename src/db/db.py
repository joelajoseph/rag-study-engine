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
            courses.course_code
        FROM chunks
        JOIN documents ON documents.document_id = chunks.document_id
        JOIN courses ON courses.course_id = documents.course_id
        WHERE chunks.faiss_id = ?
        """,
        (faiss_id,),
    ).fetchone()
