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
    """Insert one PDF's metadata and return its database ID."""
    cursor = connection.execute(
        """
        INSERT INTO documents (course_id, filename, source_type, week, chapter)
        VALUES (?, ?, ?, ?, ?)
        """,
        (course_id, filename, source_type, week, chapter),
    )
    return int(cursor.lastrowid)


def insert_chunks(
    connection: sqlite3.Connection,
    document_id: int,
    chunks: Iterable[Chunk],
) -> None:
    """Store unembedded chunks. ``faiss_id`` stays NULL until embedding."""
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
