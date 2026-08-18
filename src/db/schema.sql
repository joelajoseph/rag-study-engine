-- A course the student is taking (e.g. "CS 2110")
CREATE TABLE IF NOT EXISTS courses (
    course_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT NOT NULL UNIQUE,     -- e.g. "CS2110"
    course_name TEXT,                     -- e.g. "Data Structures"
    created_at  TEXT DEFAULT (datetime('now'))
);

-- A source document (one PDF)
CREATE TABLE IF NOT EXISTS documents (
    document_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id     INTEGER NOT NULL REFERENCES courses(course_id),
    filename      TEXT NOT NULL,
    source_type   TEXT NOT NULL,          -- 'lecture' | 'textbook' | 'tutorial' | 'notes'
    week          INTEGER,                -- nullable, not all docs map to a week
    chapter       TEXT,                   -- nullable, free text e.g. "5" or "5.2"
    ingested_at   TEXT DEFAULT (datetime('now')),
    UNIQUE(course_id, filename)
);

-- A chunk of text extracted from a document
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(document_id),
    chunk_text   TEXT NOT NULL,
    page_start   INTEGER,
    page_end     INTEGER,
    chunk_index  INTEGER NOT NULL,        -- order within the document, 0-based
    token_count  INTEGER,
    -- NULL until Stage 1's embedding step assigns a position in FAISS.
    -- UNIQUE still prevents two embedded chunks from pointing at one vector.
    faiss_id     INTEGER UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id);
