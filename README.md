Local-first, adaptive study tool built on retrieval-augmented generation (RAG) over my own course materials (PDFs). No paid APIs — all
embedding and LLM inference run locally.

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

Explicitly avoiding LangChain/LlamaIndex for now — core RAG components (extraction, chunking, embedding, retrieval, prompting) are implemented directly

## Data model

SQLite is the source of truth for all metadata. FAISS stores vectors
only. The two are linked by a single field: `faiss_id`.

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