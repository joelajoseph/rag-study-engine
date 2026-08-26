# Blueprint — Local RAG Study Engine

## Purpose
Local-first, adaptive study tool built on retrieval-augmented generation
(RAG) over the user's own course materials (PDFs). No paid APIs — all
embedding and LLM inference run locally. This document is the source of
truth for architecture, stack, and milestones. See GEMINI.md for how
agents should work with the user (teaching style, scope discipline).

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Familiar syntax, best ecosystem for this |
| Dependency mgmt | venv + requirements.txt | Simple, no extra tooling needed |
| PDF extraction | PyMuPDF (fitz) | Handles multi-column slides/textbooks well |
| Chunking | Custom fixed-size w/ overlap | Hand-rolled for learning; ~500 tokens, ~50 overlap |
| Embeddings | Sentence-Transformers, all-MiniLM-L6-v2 | Small (~80MB), fast on CPU, 384-dim vectors |
| Vector search | FAISS (IndexFlatL2) | Exact search, simple, fine at this data scale |
| Metadata / structured data | SQLite | File-based, built into Python, doubles as tracking DB later |
| Local LLM runtime | Ollama | Simple local API for quantized models |
| Local LLM | qwen2.5:3b-instruct (Q4) | Fits in 16GB alongside embeddings + OS |
| Interface | CLI (v1) | Forces understanding of the pipeline before hiding it behind a UI |

Explicitly avoiding LangChain/LlamaIndex for now — core RAG components
(extraction, chunking, embedding, retrieval, prompting) are implemented
directly so the underlying concepts are learned, not abstracted away.
Frameworks may be reconsidered later if they solve a concrete problem.

## Data model

SQLite is the source of truth for all metadata. FAISS stores vectors
only. The two are linked by a single field: `faiss_id`.

### Schema (`src/db/schema.sql`)

```sql
CREATE TABLE courses (
    course_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT NOT NULL UNIQUE,
    course_name TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE documents (
    document_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id     INTEGER NOT NULL REFERENCES courses(course_id),
    filename      TEXT NOT NULL,
    source_type   TEXT NOT NULL,          -- 'lecture' | 'textbook' | 'tutorial' | 'notes'
    week          INTEGER,
    chapter       TEXT,
    ingested_at   TEXT DEFAULT (datetime('now')),
    UNIQUE(course_id, filename)
);

CREATE TABLE chunks (
    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(document_id),
    chunk_text   TEXT NOT NULL,
    page_start   INTEGER,
    page_end     INTEGER,
    chunk_index  INTEGER NOT NULL,
    token_count  INTEGER,
    faiss_id     INTEGER NOT NULL UNIQUE
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_faiss_id ON chunks(faiss_id);
```

### Why this shape
- `courses` → `documents` → `chunks` mirrors the real hierarchy and
  enables course filtering via SQL, without touching FAISS.
- `faiss_id` is the link: FAISS returns a position, SQLite resolves it
  to real text/filename/page. FAISS never stores metadata.
- `chunk_index` supports future citation quality improvements
  (surrounding context, ordering).

## Repo conventions

- Raw PDFs: `data/raw/<course_code>/<filename>.pdf`
- Processed chunks (inspectable, pre-embedding): `data/processed/<course_code>/<filename>.json`
- FAISS index: `data/db/faiss_index/index.faiss`, with a sidecar
  `data/db/faiss_index/config.json` recording the embedding model +
  dimension it was built with (fail loudly on mismatch, don't silently
  return bad results).
- All DB access goes through `src/db/db.py` — no scattered `sqlite3.connect()` calls.
- All FAISS access goes through `src/retrieval/index_store.py` — no
  direct FAISS calls elsewhere.
- Data directories (`data/raw`, `data/processed`, `data/db`) are
  gitignored — personal, large, regeneratable.

## Stage 1: PDF ingestion & chunking

### Pipeline
1. **Extraction** (`src/ingest/extract.py`): PyMuPDF opens a PDF,
   returns text per page as `[(page_number, page_text), ...]`. Page
   boundaries are preserved for citation purposes.
2. **Chunking** (`src/ingest/chunk.py`): fixed-size chunks (~500 tokens,
   ~50 token overlap). Overlap prevents concepts from being destroyed
   at chunk boundaries. Token count approximated via word count × ~1.3
   (no tokenizer dependency needed yet). Each chunk carries text, page
   range, and `chunk_index`.
3. **Inspection point**: chunks are written to
   `data/processed/<course_code>/<filename>.json` before touching
   SQLite/FAISS — a deliberate manual sanity-check step, since PDF
   extraction quality varies (typed slides extract cleanly; scanned
   pages/heavy math notation may not — OCR is out of scope for v1).
4. **Orchestration** (`scripts/ingest_course.py`): per PDF — insert/find
   `documents` row, extract, chunk, write inspectable JSON, insert
   `chunks` rows (no `faiss_id` yet).

### Design principle
Ingestion is idempotent per document — re-running on an already-ingested
PDF skips or explicitly re-processes, never silently duplicates.
Enforced via `UNIQUE(course_id, filename)`.

## Stage 1 (cont.): Embedding & FAISS indexing

### Pipeline (`src/ingest/embed.py`)
1. Load `all-MiniLM-L6-v2` (384-dim output).
2. Batch-embed all chunks missing a `faiss_id` (batching is
   meaningfully faster than one-at-a-time).
3. For each chunk: add vector to FAISS (`IndexFlatL2`) → get assigned
   position → immediately write that position back to SQLite as
   `faiss_id`. Vector first, then pointer — an orphaned vector is
   harmless; an orphaned pointer is not.
4. Persist the index to `data/db/faiss_index/index.faiss` after
   changes; write/check `config.json` (model name + dimension) on
   every load.

### Rebuild path (`scripts/rebuild_index.py`)
`IndexFlatL2` doesn't support easy deletion. To re-chunk or remove a
document: clear affected `faiss_id`s, rebuild the index from scratch
from what's currently in SQLite. Not efficient at scale, but fast
enough at this project's size, and simpler to reason about than
in-place deletion.

### Milestone 1 — acceptance criteria
Given a course and a question, `search.py` (see Stage 2) returns the
top-k chunks with correct filenames/pages. Using hand-written questions
in `tests/eval_questions.json`, most return a chunk that actually
contains the right answer (judged by manual review).

## Stage 2: Retrieval flow

### Pipeline (`src/retrieval/search.py`)
1. Embed the query with the **same** model used at ingestion
   (`all-MiniLM-L6-v2`) — non-negotiable, this is what `config.json`
   guards against getting wrong.
2. `index_store.search(query_vector, k)` → FAISS returns `faiss_id`s +
   L2 distances (lower = more similar).
3. Join back to SQLite per result to get chunk text, filename, page
   range, source type.
4. **Course filtering**: search with a larger `k` (e.g. 20), filter to
   the requested course after the SQLite join, keep the top 5 matching.
   (Chosen over per-course FAISS indexes — unnecessary complexity at
   this data scale; "start small, earn complexity.")

### Output shape
```python
[
    {
        "chunk_text": "...",
        "filename": "Lecture5.pdf",
        "page_start": 12,
        "page_end": 12,
        "source_type": "lecture",
        "distance": 0.34
    },
    ...
]
```

## Stage 2 (cont.): Ollama integration & RAG Q&A

### Setup (one-time, outside codebase)
```bash
ollama pull qwen2.5:3b-instruct
```
Ollama runs as a local background service, exposing an HTTP API
(default `http://localhost:11434`).

### `src/generation/ollama_client.py`
Thin wrapper: prompt string in, response text out, via Ollama's
generate/chat endpoint. No prompt construction logic here — kept
model-agnostic. Blocking responses for v1 (not streaming) for
simplicity.

### `src/generation/qa.py` — prompt construction
Given a question + retrieved chunks, builds a prompt with:
1. Instructions to answer **only** from provided context, and to say
   so explicitly if the context doesn't contain the answer
   (anti-hallucination guardrail).
2. Retrieved chunks, each labeled `[Source N: filename, page]`.
3. The question, with an instruction to cite sources like `[Source N]`.

If retrieved results are all high-distance (low relevance), skip the
LLM call or explicitly flag "no relevant context found" rather than
forcing an answer. Exact distance threshold tuned empirically once real
results are visible.

### Response shape
```python
{
    "answer": "...",
    "sources": [
        {"label": "Source 1", "filename": "Lecture5.pdf", "page_start": 12, "page_end": 12},
        ...
    ]
}
```
Kept structured (not raw string) so sources can be displayed separately,
and so this shape can be reused by Stage 3 (study mode) later.

### Milestone 2 — acceptance criteria
Given a course and a question, the system returns a grounded answer
with visible citations mapping to real filenames/pages, and explicitly
says so when asked something not covered by ingested material.

## Stage 3: Study mode

Study mode adds two interaction types on top of the existing retrieval and
generation layers: **scoped Q&A** (structured, chapter-narrowed questions)
and **quiz generation** (LLM-generated questions from a chapter's content,
self-assessed against LLM-provided answers). Both are ephemeral — no
sessions, questions, or answers are persisted. Persistence and performance
tracking are explicitly out of scope here; see Stages 4-5.

### Prerequisite: verify chapter metadata

`documents.chapter` already exists in the schema but may not currently be
populated by `scripts/ingest_course.py`. Before building on it, confirm
whether ingested documents have a non-null `chapter` value. If not,
`ingest_course.py` needs a small update (or existing rows need a manual
backfill) before chapter-scoped features can work.

### Known scope limit: chapter-level only

The data model has no concept of *section* boundaries within a chapter —
chunks carry `page_start`/`page_end` but nothing tying them to a heading or
subsection. This is the same structural gap identified in Stage 2 (broad,
section-title-level queries don't retrieve well). Stage 3 scopes both
features to **chapter level only**. Section-level scoping remains a
documented future item, dependent on the same heading-aware chunking work
flagged in Stage 2.

### Pipeline

1. **Chapter-scoped retrieval** (`src/retrieval/search.py` or a new
   `src/retrieval/fetch.py`):
   - `get_chunks_by_chapter(course_code, chapter)`: direct SQL fetch of
     *all* chunks for a given chapter — no vector search involved. This is
     a different retrieval mode from Stage 2's semantic search: quiz
     generation needs comprehensive chapter content, not top-k similarity
     matches.
   - Scoped Q&A reuses Stage 2's `search_chunks`, extended with an optional
     `chapter` filter, following the same "search wide, filter after the
     SQLite join" pattern already used for course filtering.

2. **Scoped Q&A** (`src/generation/qa.py`, extended):
   - `answer_question(question, course_code, chapter=None, ...)` — when
     `chapter` is provided, retrieval is narrowed to that chapter before
     the existing prompt-construction and distance-guardrail logic runs
     unchanged.

3. **Quiz generation** (new: `src/study/quiz.py`):
   - `build_quiz_prompt(chunks, num_questions)`: constructs a prompt
     instructing the model to generate `num_questions` questions strictly
     grounded in the provided chapter content, each paired with its
     answer and a source page reference. Same anti-hallucination framing
     as Stage 2's `qa.py` — grounded only in provided context.
   - `generate_quiz(course_code, chapter, num_questions=5)`: calls
     `get_chunks_by_chapter`, builds the prompt, invokes
     `ollama_client.generate`, parses the model's output into a structured
     list.
   - **Design tradeoff — context window**: concatenating an entire
     chapter's chunks may exceed the model's usable context window,
     depending on chapter length and Ollama's configured context size.
     Approach TBD once tested against real chapter sizes — likely either
     batching questions across sub-groups of chunks, or capping how much
     chapter content is fed per generation call. Exact behavior tuned
     empirically once real chapter sizes are tested, same approach as the
     Stage 2 distance threshold.
   - **Design tradeoff — grading**: no exact-match answer checking. The
     quiz shows the model-generated answer after the user responds; the
     user self-assesses. This is a deliberate simplicity choice for v1,
     not an oversight — LLM-graded free-text comparison was considered and
     deferred as unnecessary complexity at this stage.

   Response shape:
   ```python
   [
       {
           "question": "...",
           "answer": "...",
           "source": {"filename": "...", "page_start": 4, "page_end": 4}
       },
       ...
   ]
   ```

4. **CLI entrypoint** (new: `scripts/study.py`, following the pattern of
   `scripts/ask.py`):
   - Prompt for course code and chapter.
   - Offer a choice: scoped Q&A loop, or quiz mode.
   - Quiz mode: present one question at a time, wait for the user to
     respond (any input), then reveal the model-provided answer and
     source citation before moving to the next question.
   - No persistence — closing the session discards everything.

### Milestone 3 — acceptance criteria

Given a course and a chapter, the user can either (a) ask questions
answered only from that chapter's content with correct citations, or
(b) receive a generated quiz grounded in that chapter's content, with
answers and source pages revealed on request. Both paths correctly
decline or flag when asked something outside the chapter's actual
content, consistent with Stage 2's anti-hallucination behavior.

## Stage 4: Performance tracking

Stage 4 adds durable performance tracking on top of Stage 3's quiz mode.
Quiz *attempts* become persistent; scoped Q&A sessions remain untracked,
since there's no fixed answer to self-assess against in a free-form Q&A
exchange the way there is in a quiz question.

### Known scope note: topic granularity

The data model has no concept of section/topic below the chapter level
(same structural gap flagged in Stages 2-3). Rather than waiting on
heading-aware chunking, quiz generation is extended to have the LLM emit
an informal `topic` label per question, drawn from the same chapter
content it already processes. This is explicitly free-text and
inconsistent across generations, not a controlled vocabulary — useful
for surfacing patterns, not for precise reporting.

### Schema addition (`src/db/schema.sql`)

```sql
CREATE TABLE quiz_attempts (
    attempt_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id        INTEGER NOT NULL REFERENCES documents(document_id),
    chapter            TEXT NOT NULL,        -- denormalized, insulates history
                                              -- from a document's chapter tag changing
    topic              TEXT,                 -- LLM-generated, informal, nullable
    question_text      TEXT NOT NULL,
    model_answer        TEXT NOT NULL,
    self_correct        TEXT NOT NULL,        -- 'correct' | 'partial' | 'incorrect'
    confidence          INTEGER NOT NULL,     -- 1-5, captured AFTER reveal
    source_filename     TEXT,
    source_page_start    INTEGER,
    source_page_end      INTEGER,
    attempted_at        TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_attempts_chapter ON quiz_attempts(document_id, chapter);
CREATE INDEX idx_attempts_topic ON quiz_attempts(topic);
```

### Pipeline

1. **Quiz generation (`src/study/quiz.py`, extended):** `build_quiz_prompt`
   extended so the model returns `{excerpt, question, answer, topic}` — a
   short informal topic label per question, generated from the same
   chapter content it already sees. `parse_quiz_response` treats a
   missing/malformed `topic` as a soft failure: null it out and keep the
   question, rather than dropping the whole item — consistent with Stage
   3's "don't silently over-penalize" pattern, but scoped to just this
   one field.

2. **CLI capture (`scripts/study.py`, extended):** after the model's
   answer is revealed, prompt once for self-assessed correctness
   (correct/partial/incorrect) and once for retrospective confidence
   (1-5), then write the attempt. Confidence is captured *after* the
   reveal rather than before — this trades away a calibration signal
   (can't detect "confidently wrong" patterns as directly) in exchange
   for a single CLI interruption per question instead of two.

3. **DB layer (`src/db/db.py`, extended):** `record_quiz_attempt(...)`,
   `get_attempts_by_chapter(course_code, chapter)`,
   `get_attempts_summary(course_code)`. Raw reads only — no scoring logic
   here, per the existing "all DB access through db.py" convention.

4. **Scoring (`src/study/scoring.py`, new):** deterministic
   `chapter_mastery_score(attempts)` — a weighted combination of
   correctness rate (correct=1, partial=0.5, incorrect=0), mean
   confidence, and recency decay. Computed on-the-fly from raw rows, not
   cached, to avoid a second source of truth going stale. Weights are
   placeholder constants pending empirical tuning, same pattern as the
   Stage 2 distance threshold and Stage 3 quiz-context budget.

5. **Out of scope for v1:** response-time/behavioral pattern tracking
   from the original note — no CLI instrumentation for this yet.

### Milestone 4 — acceptance criteria

After completing a quiz, self-assessed correctness (3-way), retrospective
confidence, and an informal topic tag are persisted per question and
queryable by chapter and topic. A chapter-level mastery score can be
computed from stored attempts via a documented, deterministic weighted
formula — no ML, no hidden heuristics.