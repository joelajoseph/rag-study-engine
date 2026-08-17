-- A course the student is taking (e.g. "CS 2110")
CREATE TABLE courses (
    course_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT NOT NULL UNIQUE,     -- e.g. "CS2110"
    course_name TEXT,                     -- e.g. "Data Structures"
    created_at  TEXT DEFAULT (datetime('now'))
);

-- A source document (one PDF)
CREATE TABLE documents (
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
CREATE TABLE chunks (
    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(document_id),
    chunk_text   TEXT NOT NULL,
    page_start   INTEGER,
    page_end     INTEGER,
    chunk_index  INTEGER NOT NULL,        -- order within the document, 0-based
    token_count  INTEGER,
    faiss_id     INTEGER NOT NULL UNIQUE  -- position of this chunk's vector in the FAISS index
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_faiss_id ON chunks(faiss_id);