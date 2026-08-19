"""Evaluation and unit tests for search and retrieval."""

import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.search import search_chunks

EVAL_QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"


class TestRetrieval(unittest.TestCase):
    def test_search_chunks_validation(self) -> None:
        """Test parameter boundary conditions and validation errors."""
        with self.assertRaises(ValueError):
            search_chunks("", "CSCA48")
        with self.assertRaises(ValueError):
            search_chunks("   ", "CSCA48")
        with self.assertRaises(ValueError):
            search_chunks("test", "CSCA48", k=0)
        with self.assertRaises(ValueError):
            search_chunks("test", "CSCA48", k=10, candidate_k=5)

    def test_search_chunks_unknown_course(self) -> None:
        """Searching for a course code with no documents should return empty results."""
        results = search_chunks("programming", "NONEXISTENT_COURSE_123")
        self.assertEqual(results, [])

    def test_eval_questions_retrieval(self) -> None:
        """Evaluate retrieval against curated evaluation questions (Milestone 1)."""
        self.assertTrue(
            EVAL_QUESTIONS_PATH.exists(),
            f"Eval questions file not found at {EVAL_QUESTIONS_PATH}",
        )

        with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        self.assertGreater(len(eval_data), 0, "No evaluation questions found.")

        passed_count = 0
        total_count = len(eval_data)

        for item in eval_data:
            q_id = item["id"]
            question = item["question"]
            course = item["course"]
            expected = item["expected_source"]
            expected_filename = expected["filename"]
            expected_page = expected["page"]

            results = search_chunks(question, course, k=5)
            self.assertGreater(
                len(results),
                0,
                f"[{q_id}] Expected retrieval results for query '{question}', got none.",
            )

            # Check if expected source is present in top-k results
            matched_indices = []
            for rank, result in enumerate(results):
                if (
                    result["filename"] == expected_filename
                    and result["page_start"] <= expected_page <= result["page_end"]
                ):
                    matched_indices.append(rank)

            is_hit = len(matched_indices) > 0
            if is_hit:
                passed_count += 1

            self.assertTrue(
                is_hit,
                f"[{q_id}] Query: '{question}'\n"
                f"Expected source: {expected_filename} (page {expected_page})\n"
                f"Retrieved top sources: {[(r['filename'], r['page_start'], r['page_end']) for r in results]}",
            )

        print(f"\n[Retrieval Eval] Hit@5 Success: {passed_count}/{total_count} ({passed_count / total_count * 100:.1f}%)")


def run_eval_report() -> None:
    """Print an inspectable, human-readable summary of retrieval results for each eval question."""
    with open(EVAL_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print("=" * 80)
    print("STAGE 1 RETRIEVAL EVALUATION REPORT")
    print("=" * 80)

    for item in eval_data:
        q_id = item["id"]
        question = item["question"]
        course = item["course"]
        expected = item["expected_source"]
        notes = item.get("notes", "")

        results = search_chunks(question, course, k=5)

        print(f"\nQuestion ID : {q_id}")
        print(f"Course      : {course}")
        print(f"Question    : {question}")
        print(f"Expected    : {expected['filename']} (page {expected['page']})")
        if notes:
            print(f"Notes       : {notes}")
        print("-" * 60)

        if not results:
            print("  [!] No results returned.")
            continue

        for rank, res in enumerate(results, start=1):
            is_match = (
                res["filename"] == expected["filename"]
                and res["page_start"] <= expected["page"] <= res["page_end"]
            )
            match_flag = "✓ MATCH" if is_match else "       "
            snippet = res["chunk_text"].replace("\n", " ")[:120] + "..."
            print(
                f"  Rank #{rank} [{match_flag}] "
                f"L2 Distance: {res['distance']:.4f} | "
                f"Source: {res['filename']} (pages {res['page_start']}-{res['page_end']})"
            )
            print(f"    Snippet: {snippet}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_eval_report()
    unittest.main()
