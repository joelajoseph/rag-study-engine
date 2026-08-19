# Agent Instructions — RAG Study Engine

## Who you're working with
The user is an early CS student, comfortable with Python syntax, but new to:
Next.js/React (used in a separate project), embeddings, vector search, RAG
pipelines, and local LLM tooling (Ollama). This project is explicitly a
learning project, not just a deliverable.

## How to work
- You may write the actual code. The user does not need to type
  every line themselves.
- Before or alongside writing code for a step, explain: what this step
  does, why it's needed at this point in the pipeline, and how it connects
  to the steps before/after it.
- Do NOT explain every individual line of code. Assume Python syntax
  fluency. Focus explanations on architecture, data flow, and the "why,"
  not syntax.
- After completing a step, summarize what changed: which files were
  created/modified, and what new capability now exists.
- Take one milestone/stage at a time (see docs/BLUEPRINT.md for stage
  breakdown). Do not jump ahead to later stages or add functionality
  not yet scoped, even if it seems like a natural next step.
- If a design decision has real tradeoffs (e.g. chunk size, similarity
  metric, model choice), briefly explain the tradeoff rather than
  silently picking one.
- Prefer explicit, readable code over clever/compressed code, since
  this is a learning project.

## Ground truth
- docs/BLUEPRINT.md is the source of truth for architecture, stack, and
  milestones. If a request conflicts with it, flag the conflict rather
  than silently deviating.
- Do not introduce new major dependencies or frameworks (e.g. LangChain,
  LlamaIndex) without explicit user approval — see BLUEPRINT.md design
  principles for why.

## Environment
- Python 3.11+, virtual environment at venv/ (already created).
- Local-only stack: no paid APIs. All LLM inference via Ollama, running
  locally.
- Data in data/ (raw PDFs, processed chunks, sqlite + FAISS index) is
  gitignored — never assume it's empty or ask the user to commit it.