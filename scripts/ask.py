"""Interactive CLI for Stage 2 RAG Q&A."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.ollama_client import OllamaConnectionError, OllamaError
from src.generation.qa import DEFAULT_DISTANCE_THRESHOLD, answer_question
from src.retrieval.search import DEFAULT_RESULT_COUNT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions about course materials via RAG.")
    parser.add_argument(
        "--course",
        "-c",
        dest="course_code",
        help="Course code to query (e.g. CSCA48). If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/db/study_engine.db"),
        help="SQLite database path (default: data/db/study_engine.db).",
    )
    parser.add_argument(
        "--index-directory",
        type=Path,
        default=Path("data/db/faiss_index"),
        help="FAISS index directory (default: data/db/faiss_index).",
    )
    parser.add_argument(
        "--model",
        help="Ollama model name override (default: qwen2.5:3b-instruct).",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_RESULT_COUNT,
        help=f"Number of retrieved chunks (default: {DEFAULT_RESULT_COUNT}).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help=f"Maximum FAISS L2 distance allowed (default: {DEFAULT_DISTANCE_THRESHOLD}).",
    )
    return parser


def format_sources_display(sources: list[dict[str, str | int]]) -> str:
    """Format structured source list for terminal display."""
    if not sources:
        return "  (No sources used)"

    lines = []
    for s in sources:
        label = s["label"]
        filename = s["filename"]
        p_start = s["page_start"]
        p_end = s["page_end"]
        page_str = f"page {p_start}" if p_start == p_end else f"pages {p_start}-{p_end}"
        lines.append(f"  - [{label}] {filename} ({page_str})")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()

    course_code = args.course_code
    if not course_code:
        try:
            course_code = input("Enter course code (e.g. CSCA48): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not course_code or course_code.lower() in ("quit", "exit", "q"):
        print("Exiting.")
        sys.exit(0)

    print("=" * 60)
    print(f"RAG Study Engine — Q&A Mode ({course_code})")
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        try:
            result = answer_question(
                question=question,
                course_code=course_code,
                database_path=args.database,
                index_directory=args.index_directory,
                k=args.k,
                distance_threshold=args.distance_threshold,
                model=args.model,
            )

            print("\nAnswer:")
            print(result["answer"])
            print("\nSources:")
            print(format_sources_display(result["sources"]))

        except OllamaConnectionError as exc:
            print(
                f"\n[Connection Error] Could not connect to Ollama.\n"
                f"Make sure the Ollama service is running locally (e.g., run `ollama serve`).\n"
                f"Details: {exc}"
            )
        except OllamaError as exc:
            print(f"\n[Ollama Error] Generation request failed: {exc}")
        except Exception as exc:
            print(f"\n[Error] {exc}")


if __name__ == "__main__":
    main()
