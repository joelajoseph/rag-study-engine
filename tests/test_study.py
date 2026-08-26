"""Unit tests for Stage 3 Study Mode (scoped Q&A, quiz generation, and chapter metadata)."""

import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_course import infer_chapter_from_filename
from src.db.db import (
    get_chapters_for_course,
    get_chunks_by_chapter,
    initialize_database,
)
from src.generation.qa import answer_question
from src.study.quiz import (
    build_quiz_prompt,
    extract_excerpt_index,
    format_chunk_page_range,
    generate_quiz,
    parse_quiz_response,
    select_quiz_chunks,
)




class TestChapterInference(unittest.TestCase):
    def test_infer_chapter_from_filename(self) -> None:
        self.assertEqual(infer_chapter_from_filename("chapter_1.pdf"), "1")
        self.assertEqual(infer_chapter_from_filename("chapter-2.pdf"), "2")
        self.assertEqual(infer_chapter_from_filename("chapter 3.pdf"), "3")
        self.assertEqual(infer_chapter_from_filename("ch4.pdf"), "4")
        self.assertEqual(infer_chapter_from_filename("chapter_1_2.pdf"), "1_2")
        self.assertIsNone(infer_chapter_from_filename("lecture_slides.pdf"))


class TestDatabaseChapterQueries(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        schema_path = PROJECT_ROOT / "src" / "db" / "schema.sql"
        self.connection.executescript(schema_path.read_text(encoding="utf-8"))

        # Seed test data
        self.connection.execute("INSERT INTO courses (course_code, course_name) VALUES ('CSCA48', 'Intro CS')")
        self.connection.execute(
            "INSERT INTO documents (course_id, filename, source_type, chapter) VALUES (1, 'chapter_1.pdf', 'textbook', '1')"
        )
        self.connection.execute(
            "INSERT INTO documents (course_id, filename, source_type, chapter) VALUES (1, 'chapter_2.pdf', 'textbook', '2')"
        )
        self.connection.execute(
            """INSERT INTO chunks (document_id, chunk_text, page_start, page_end, chunk_index, token_count, faiss_id)
               VALUES (1, 'Chapter 1 Chunk 0 text', 1, 2, 0, 100, 0)"""
        )
        self.connection.execute(
            """INSERT INTO chunks (document_id, chunk_text, page_start, page_end, chunk_index, token_count, faiss_id)
               VALUES (1, 'Chapter 1 Chunk 1 text', 3, 3, 1, 120, 1)"""
        )
        self.connection.execute(
            """INSERT INTO chunks (document_id, chunk_text, page_start, page_end, chunk_index, token_count, faiss_id)
               VALUES (2, 'Chapter 2 Chunk 0 text', 1, 2, 0, 110, 2)"""
        )
        self.connection.commit()

    def tearDown(self) -> None:
        self.connection.close()

    def test_get_chapters_for_course(self) -> None:
        chapters = get_chapters_for_course(self.connection, "CSCA48")
        self.assertEqual(chapters, ["1", "2"])

        unknown = get_chapters_for_course(self.connection, "NONEXISTENT")
        self.assertEqual(unknown, [])

    def test_get_chunks_by_chapter(self) -> None:
        ch1_chunks = get_chunks_by_chapter(self.connection, "CSCA48", "1")
        self.assertEqual(len(ch1_chunks), 2)
        self.assertEqual(ch1_chunks[0]["chunk_text"], "Chapter 1 Chunk 0 text")
        self.assertEqual(ch1_chunks[1]["chunk_text"], "Chapter 1 Chunk 1 text")

        ch2_chunks = get_chunks_by_chapter(self.connection, "CSCA48", "2")
        self.assertEqual(len(ch2_chunks), 1)
        self.assertEqual(ch2_chunks[0]["chunk_text"], "Chapter 2 Chunk 0 text")

        ch_none = get_chunks_by_chapter(self.connection, "CSCA48", "999")
        self.assertEqual(ch_none, [])


class TestQuizGeneration(unittest.TestCase):
    def test_format_chunk_page_range(self) -> None:
        self.assertEqual(format_chunk_page_range({"page_start": 3, "page_end": 3}), "page 3")
        self.assertEqual(format_chunk_page_range({"page_start": 3, "page_end": 5}), "pages 3-5")

    def test_select_quiz_chunks_under_budget(self) -> None:
        chunks = [
            {"filename": "ch1.pdf", "page_start": 1, "page_end": 2, "token_count": 200, "chunk_text": "text"}
            for _ in range(4)
        ]
        selected = select_quiz_chunks(chunks, max_tokens=3000, max_chunks=8)
        self.assertEqual(len(selected), 4)

    def test_select_quiz_chunks_stride_sampling(self) -> None:
        chunks = [
            {"filename": "ch1.pdf", "page_start": i, "page_end": i, "token_count": 500, "chunk_text": f"text {i}"}
            for i in range(1, 21)
        ]
        # Should sample 5 chunks evenly across the 20 chunks
        selected = select_quiz_chunks(chunks, max_tokens=2500, max_chunks=5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected[0]["page_start"], 1)
        self.assertEqual(selected[-1]["page_start"], 17)


    def test_build_quiz_prompt(self) -> None:
        chunks = [
            {"filename": "ch1.pdf", "page_start": 2, "page_end": 3, "chunk_text": "Pointers hold addresses."},
            {"filename": "ch1.pdf", "page_start": 4, "page_end": 5, "chunk_text": "Malloc allocates memory."},
            {"filename": "ch1.pdf", "page_start": 6, "page_end": 7, "chunk_text": "Free releases memory."},
        ]
        prompt = build_quiz_prompt(chunks, num_questions=3)
        self.assertIn("exactly 3 practice questions", prompt)
        self.assertIn("[Excerpt 1: ch1.pdf, pages 2-3]", prompt)
        self.assertIn("[Excerpt 2: ch1.pdf, pages 4-5]", prompt)
        self.assertIn("Pointers hold addresses.", prompt)
        self.assertIn("Required JSON Output Format", prompt)


    def test_extract_excerpt_index(self) -> None:
        self.assertEqual(extract_excerpt_index({"excerpt": 2}), 2)
        self.assertEqual(extract_excerpt_index({"excerpt": "Excerpt 3"}), 3)
        self.assertEqual(extract_excerpt_index({"source": "4"}), 4)
        self.assertEqual(extract_excerpt_index({"source": {"excerpt": 5}}), 5)
        self.assertIsNone(extract_excerpt_index({"other": "field"}))

    def test_parse_quiz_response_with_excerpts(self) -> None:
        chunks = [
            {"filename": "chapter_1.pdf", "page_start": 1, "page_end": 2},
            {"filename": "chapter_1.pdf", "page_start": 4, "page_end": 5},
            {"filename": "chapter_1.pdf", "page_start": 8, "page_end": 9},
        ]
        raw = json.dumps(
            [
                {
                    "excerpt": 1,
                    "question": "What is a pointer?",
                    "answer": "A variable storing an address.",
                },
                {
                    "excerpt": 2,
                    "question": "What is malloc?",
                    "answer": "Dynamic memory allocation.",
                },
                {
                    "excerpt": 3,
                    "question": "How to free memory?",
                    "answer": "Using the free() function.",
                },
            ]
        )
        items = parse_quiz_response(raw, chunks=chunks)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["source"], {"filename": "chapter_1.pdf", "page_start": 1, "page_end": 2})
        self.assertEqual(items[1]["source"], {"filename": "chapter_1.pdf", "page_start": 4, "page_end": 5})
        self.assertEqual(items[2]["source"], {"filename": "chapter_1.pdf", "page_start": 8, "page_end": 9})

    def test_parse_quiz_response_positional_fallback(self) -> None:
        chunks = [
            {"filename": "chapter_1.pdf", "page_start": 1, "page_end": 2},
            {"filename": "chapter_1.pdf", "page_start": 6, "page_end": 7},
        ]
        # LLM omitted 'excerpt' key entirely
        raw = json.dumps(
            [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ]
        )
        items = parse_quiz_response(raw, chunks=chunks)
        self.assertEqual(len(items), 2)
        # First question gets first chunk, second gets second chunk
        self.assertEqual(items[0]["source"], {"filename": "chapter_1.pdf", "page_start": 1, "page_end": 2})
        self.assertEqual(items[1]["source"], {"filename": "chapter_1.pdf", "page_start": 6, "page_end": 7})

    def test_parse_quiz_response_code_fence(self) -> None:
        chunks = [{"filename": "ch1.pdf", "page_start": 2, "page_end": 2}]
        raw = (
            "Here is the quiz:\n"
            "```json\n"
            "[\n"
            "  {\n"
            '    "excerpt": 1,\n'
            '    "question": "What is a pointer?",\n'
            '    "answer": "A variable that holds an address."\n'
            "  }\n"
            "]\n"
            "```"
        )
        items = parse_quiz_response(raw, chunks=chunks)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["question"], "What is a pointer?")
        self.assertEqual(items[0]["source"]["filename"], "ch1.pdf")
        self.assertEqual(items[0]["source"]["page_start"], 2)

    def test_parse_quiz_response_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_quiz_response("Not valid JSON output")



    @patch("src.study.quiz.get_chunks_by_chapter")
    def test_generate_quiz_end_to_end(self, mock_get_chunks: MagicMock) -> None:
        mock_get_chunks.return_value = [
            {
                "chunk_text": "Pointers store memory addresses.",
                "page_start": 4,
                "page_end": 4,
                "token_count": 100,
                "filename": "chapter_1.pdf",
            }
        ]
        mock_client = MagicMock()
        mock_client.generate.return_value = json.dumps(
            [
                {
                    "excerpt": 1,
                    "question": "What do pointers store?",
                    "answer": "Memory addresses.",
                }
            ]
        )


        quiz = generate_quiz(
            course_code="CSCA48",
            chapter="1",
            num_questions=1,
            client=mock_client,
        )

        self.assertEqual(len(quiz), 1)
        self.assertEqual(quiz[0]["question"], "What do pointers store?")
        self.assertEqual(quiz[0]["answer"], "Memory addresses.")
        self.assertEqual(quiz[0]["source"]["filename"], "chapter_1.pdf")
        mock_client.generate.assert_called_once()


class TestScopedQA(unittest.TestCase):
    @patch("src.generation.qa.search_chunks")
    def test_answer_question_with_chapter(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {
                "chunk_text": "Pointers point to memory.",
                "filename": "chapter_1.pdf",
                "page_start": 2,
                "page_end": 2,
                "chapter": "1",
                "distance": 0.4,
            }
        ]
        mock_client = MagicMock()
        mock_client.generate.return_value = "Pointers point to memory [Source 1]."

        result = answer_question(
            "What is a pointer?",
            "CSCA48",
            chapter="1",
            client=mock_client,
        )

        mock_search.assert_called_once()
        self.assertEqual(mock_search.call_args[1]["chapter"], "1")
        self.assertEqual(result["answer"], "Pointers point to memory [Source 1].")
        self.assertEqual(len(result["sources"]), 1)


if __name__ == "__main__":
    unittest.main()
