"""Interactive CLI for Stage 3 Study Mode (Scoped Q&A and Practice Quizzes)."""

import argparse
from contextlib import closing
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.db import get_chapters_for_course, get_connection
from src.generation.ollama_client import OllamaConnectionError, OllamaError
from src.generation.qa import DEFAULT_DISTANCE_THRESHOLD, answer_question
from src.retrieval.search import DEFAULT_RESULT_COUNT
from src.study.quiz import generate_quiz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Study course materials via scoped Q&A or practice quizzes.")
    parser.add_argument(
        "--course",
        "-c",
        dest="course_code",
        help="Course code to study (e.g. CSCA48). If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--chapter",
        help="Chapter identifier (e.g. 1). If omitted, you will select from available chapters.",
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
        help=f"Number of retrieved chunks for Q&A (default: {DEFAULT_RESULT_COUNT}).",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help=f"Maximum FAISS L2 distance allowed for Q&A (default: {DEFAULT_DISTANCE_THRESHOLD}).",
    )
    return parser


def format_sources_display(sources: list[dict[str, Any]]) -> str:
    """Format structured source list for terminal display."""
    if not sources:
        return "  (No sources used)"

    lines = []
    for s in sources:
        label = s.get("label", "Source")
        filename = s.get("filename", "unknown")
        p_start = s.get("page_start", 0)
        p_end = s.get("page_end", 0)
        page_str = f"page {p_start}" if p_start == p_end else f"pages {p_start}-{p_end}"
        lines.append(f"  - [{label}] {filename} ({page_str})")
    return "\n".join(lines)


def select_chapter_interactive(database_path: Path, course_code: str) -> str | None:
    """Prompt the user to select from available chapters for a course."""
    with closing(get_connection(database_path)) as connection:
        chapters = get_chapters_for_course(connection, course_code)

    if not chapters:
        print(f"[!] No chapters found for course '{course_code}' in {database_path}.")
        return None

    if len(chapters) == 1:
        print(f"Auto-selected only available chapter: Chapter '{chapters[0]}'")
        return chapters[0]

    print(f"\nAvailable Chapters for {course_code}:")
    for idx, ch in enumerate(chapters, start=1):
        print(f"  [{idx}] Chapter {ch}")

    while True:
        try:
            choice = input(f"\nSelect a chapter (1-{len(chapters)}, or 'q' to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            return None

        if choice.lower() in ("q", "quit", "exit"):
            return None

        if choice.isdigit():
            val = int(choice)
            if 1 <= val <= len(chapters):
                return chapters[val - 1]

        print(f"Invalid selection. Please enter a number between 1 and {len(chapters)}.")


def run_scoped_qa_loop(
    course_code: str,
    chapter: str,
    args: argparse.Namespace,
) -> None:
    """Interactive loop for asking questions scoped to a chapter."""
    print("\n" + "=" * 60)
    print(f"Scoped Q&A Mode — {course_code} (Chapter {chapter})")
    print("Type 'back' to return to mode menu, or 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nReturning to menu.")
            break

        if not question:
            continue

        if question.lower() == "back":
            break
        if question.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            sys.exit(0)

        try:
            result = answer_question(
                question=question,
                course_code=course_code,
                chapter=chapter,
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
                f"Make sure the Ollama service is running locally (e.g., `ollama serve`).\n"
                f"Details: {exc}"
            )
        except OllamaError as exc:
            print(f"\n[Ollama Error] Generation request failed: {exc}")
        except Exception as exc:
            print(f"\n[Error] {exc}")


def run_quiz_mode(
    course_code: str,
    chapter: str,
    args: argparse.Namespace,
) -> None:
    """Generate and run a self-assessed practice quiz."""
    print("\n" + "=" * 60)
    print(f"Practice Quiz Mode — {course_code} (Chapter {chapter})")
    print("Generating grounded quiz questions from chapter material...")
    print("=" * 60)

    try:
        quiz_items = generate_quiz(
            course_code=course_code,
            chapter=chapter,
            num_questions=5,
            database_path=args.database,
            model=args.model,
        )
    except OllamaConnectionError as exc:
        print(
            f"\n[Connection Error] Could not connect to Ollama.\n"
            f"Make sure the Ollama service is running locally (e.g., `ollama serve`).\n"
            f"Details: {exc}"
        )
        return
    except Exception as exc:
        print(f"\n[Error] Failed to generate quiz: {exc}")
        return

    if not quiz_items:
        print("[!] No quiz questions could be generated.")
        return

    print(f"\nGenerated {len(quiz_items)} practice questions. Let's begin!\n")

    for i, item in enumerate(quiz_items, start=1):
        print("-" * 60)
        print(f"Question {i}/{len(quiz_items)}:")
        print(item["question"])
        print("-" * 60)

        try:
            input("\nYour Answer (type response or press Enter to reveal): ")
        except (KeyboardInterrupt, EOFError):
            print("\nQuiz stopped.")
            break

        print("\n--- Answer Key ---")
        print(item["answer"])

        source = item.get("source", {})
        filename = source.get("filename") or "doc"
        p_start = int(source.get("page_start") or 0)
        p_end = int(source.get("page_end") or 0)
        if p_start > 0:
            page_str = f"page {p_start}" if p_start == p_end else f"pages {p_start}-{p_end}"
            print(f"\nSource Reference: {filename} ({page_str})")
        else:
            print(f"\nSource Reference: {filename}")


        if i < len(quiz_items):
            try:
                input("\n[Press Enter for next question...]")
            except (KeyboardInterrupt, EOFError):
                print("\nQuiz finished early.")
                break

    print("\n" + "=" * 60)
    print("Quiz Complete! (Scores and answers are ephemeral and not saved)")
    print("=" * 60)


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

    chapter = args.chapter
    if not chapter:
        chapter = select_chapter_interactive(args.database, course_code)
        if not chapter:
            print("No chapter selected. Exiting.")
            sys.exit(0)

    while True:
        print("\n" + "=" * 60)
        print(f"RAG Study Engine — Study Mode: {course_code} (Chapter {chapter})")
        print("=" * 60)
        print("Select Mode:")
        print("  [1] Scoped Q&A (Ask questions about this chapter)")
        print("  [2] Practice Quiz (Self-assessed chapter quiz)")
        print("  [3] Change Chapter")
        print("  [q] Quit")

        try:
            choice = input("\nEnter choice [1-3, q]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if choice in ("1", "qa", "q&a"):
            run_scoped_qa_loop(course_code, chapter, args)
        elif choice in ("2", "quiz"):
            run_quiz_mode(course_code, chapter, args)
        elif choice == "3":
            new_chapter = select_chapter_interactive(args.database, course_code)
            if new_chapter:
                chapter = new_chapter
        elif choice.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or q.")


if __name__ == "__main__":
    main()
