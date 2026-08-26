"""Interactive CLI for Study Mode (Scoped Q&A, Practice Quizzes, and Performance Tracking)."""

import argparse
from contextlib import closing
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.db import (
    get_attempts_by_chapter,
    get_chapters_for_course,
    get_connection,
    initialize_database,
    record_quiz_attempt,
)
from src.generation.ollama_client import OllamaConnectionError, OllamaError
from src.generation.qa import DEFAULT_DISTANCE_THRESHOLD, answer_question
from src.retrieval.search import DEFAULT_RESULT_COUNT
from src.study.quiz import generate_quiz
from src.study.scoring import chapter_mastery_score


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


def prompt_self_correctness() -> str:
    """Prompt student to self-assess their answer correctness."""
    mapping = {
        "c": "correct",
        "correct": "correct",
        "1": "correct",
        "p": "partial",
        "partial": "partial",
        "2": "partial",
        "i": "incorrect",
        "incorrect": "incorrect",
        "3": "incorrect",
    }
    while True:
        try:
            val = input("Self-assessment ([c]orrect / [p]artial / [i]ncorrect): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            raise
        if val in mapping:
            return mapping[val]
        print("Invalid input. Please enter 'c' (correct), 'p' (partial), or 'i' (incorrect).")


def prompt_confidence() -> int:
    """Prompt student for retrospective confidence rating (1-5)."""
    while True:
        try:
            val = input("Confidence rating (1-5, where 5 is highest): ").strip()
        except (KeyboardInterrupt, EOFError):
            raise
        if val.isdigit() and 1 <= int(val) <= 5:
            return int(val)
        print("Invalid input. Please enter an integer between 1 and 5.")


def resolve_document_id(
    database_path: Path,
    course_code: str,
    source: dict[str, Any],
) -> int:
    """Resolve document_id from source metadata or database lookup."""
    doc_id = source.get("document_id")
    if doc_id is not None and isinstance(doc_id, int):
        return doc_id

    filename = source.get("filename", "")
    with closing(get_connection(database_path)) as connection:
        row = connection.execute(
            """
            SELECT documents.document_id
            FROM documents
            JOIN courses ON courses.course_id = documents.course_id
            WHERE courses.course_code = ? AND documents.filename = ?
            LIMIT 1
            """,
            (course_code, filename),
        ).fetchone()

        if row is not None:
            return int(row["document_id"])

        # Fallback: first document matching course
        fallback_row = connection.execute(
            """
            SELECT documents.document_id
            FROM documents
            JOIN courses ON courses.course_id = documents.course_id
            WHERE courses.course_code = ?
            LIMIT 1
            """,
            (course_code,),
        ).fetchone()
        if fallback_row is not None:
            return int(fallback_row["document_id"])

    raise RuntimeError(f"Could not resolve document_id for course '{course_code}' and file '{filename}'.")


def display_chapter_mastery(database_path: Path, course_code: str, chapter: str) -> None:
    """Fetch attempts and display formatted chapter mastery progress."""
    with closing(get_connection(database_path)) as connection:
        attempts = get_attempts_by_chapter(connection, course_code, chapter)

    print("\n" + "=" * 60)
    print(f"Chapter Mastery Progress — {course_code} (Chapter {chapter})")
    print("=" * 60)

    score_data = chapter_mastery_score(attempts)
    if score_data is None:
        print("No practice attempts recorded yet for this chapter.")
        print("Complete a practice quiz to generate performance tracking metrics.")
        return

    mastery = score_data["mastery_score"]
    total = score_data["total_attempts"]
    raw_corr_pct = score_data["correctness_rate"] * 100.0
    conf_norm = score_data["mean_confidence"]
    conf_1_to_5 = 1.0 + (conf_norm * 4.0)
    recency_corr_pct = score_data["recency_weighted_correctness"] * 100.0

    print(f"Overall Mastery Score:             {mastery:.1f}%")
    print(f"Total Questions Practiced:         {total}")
    print(f"Overall Correctness Rate:          {raw_corr_pct:.1f}%")
    print(f"Average Confidence:                {conf_1_to_5:.1f} / 5.0")
    print(f"Recency-Weighted Correctness:      {recency_corr_pct:.1f}%")

    # Show recent attempts breakdown
    recent_attempts = attempts[-5:]
    print("\nRecent Practice History:")
    for a in recent_attempts:
        topic_str = f" [{a['topic']}]" if a["topic"] else ""
        date_str = str(a["attempted_at"])[:16]
        print(f"  • {date_str} - {a['self_correct'].upper()} (conf: {a['confidence']}/5){topic_str}")


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
    """Generate and run a self-assessed practice quiz with attempt tracking."""
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

    recorded_count = 0
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

        if item.get("topic"):
            print(f"Topic: {item['topic']}")

        print("\n--- Record Attempt ---")
        try:
            self_correct = prompt_self_correctness()
            confidence = prompt_confidence()
        except (KeyboardInterrupt, EOFError):
            print("\nQuiz stopped before saving this question.")
            break

        try:
            doc_id = resolve_document_id(args.database, course_code, source)
            with closing(get_connection(args.database)) as connection:
                with connection:
                    record_quiz_attempt(
                        connection,
                        document_id=doc_id,
                        chapter=chapter,
                        topic=item.get("topic"),
                        question_text=item["question"],
                        model_answer=item["answer"],
                        self_correct=self_correct,
                        confidence=confidence,
                        source_filename=filename,
                        source_page_start=p_start if p_start > 0 else None,
                        source_page_end=p_end if p_end > 0 else None,
                    )
            recorded_count += 1
            print(f"[Saved] Attempt recorded ({self_correct}, confidence {confidence}/5).")
        except Exception as exc:
            print(f"[!] Warning: Could not record attempt to database: {exc}")

        if i < len(quiz_items):
            try:
                input("\n[Press Enter for next question...]")
            except (KeyboardInterrupt, EOFError):
                print("\nQuiz finished early.")
                break

    print("\n" + "=" * 60)
    print(f"Quiz Complete! Recorded {recorded_count} question attempt(s).")
    print("=" * 60)

    if recorded_count > 0:
        display_chapter_mastery(args.database, course_code, chapter)


def main() -> None:
    args = build_parser().parse_args()

    # Ensure database tables exist (idempotent)
    initialize_database(args.database)

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
        print("  [3] View Chapter Mastery / Progress")
        print("  [4] Change Chapter")
        print("  [q] Quit")

        try:
            choice = input("\nEnter choice [1-4, q]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if choice in ("1", "qa", "q&a"):
            run_scoped_qa_loop(course_code, chapter, args)
        elif choice in ("2", "quiz"):
            run_quiz_mode(course_code, chapter, args)
        elif choice in ("3", "mastery", "progress"):
            display_chapter_mastery(args.database, course_code, chapter)
        elif choice == "4":
            new_chapter = select_chapter_interactive(args.database, course_code)
            if new_chapter:
                chapter = new_chapter
        elif choice.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, 4, or q.")


if __name__ == "__main__":
    main()
