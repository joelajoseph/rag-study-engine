"""Unit tests for generation layer (ollama_client and qa)."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    generate,
)
from src.generation.qa import (
    answer_question,
    build_qa_prompt,
    format_source_label,
    format_sources,
)


class TestOllamaClient(unittest.TestCase):
    def test_empty_prompt_raises_value_error(self) -> None:
        client = OllamaClient()
        with self.assertRaises(ValueError):
            client.generate("")
        with self.assertRaises(ValueError):
            client.generate("   ")

    @patch("urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"response": "Linked lists use pointers.", "done": True}
        ).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = OllamaClient(host="http://localhost:11434", default_model="qwen2.5:3b-instruct")
        result = client.generate("Explain linked lists", options={"temperature": 0.0})

        self.assertEqual(result, "Linked lists use pointers.")
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/generate")
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen2.5:3b-instruct")
        self.assertEqual(payload["prompt"], "Explain linked lists")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"], {"temperature": 0.0})

    @patch("urllib.request.urlopen")
    def test_generate_connection_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        client = OllamaClient()
        with self.assertRaises(OllamaConnectionError):
            client.generate("Hello")

    @patch("urllib.request.urlopen")
    def test_generate_api_error(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"error": "model 'nonexistent' not found"}
        ).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        client = OllamaClient()
        with self.assertRaises(OllamaError) as ctx:
            client.generate("Hello")
        self.assertIn("model 'nonexistent' not found", str(ctx.exception))


class TestQA(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_chunks = [
            {
                "chunk_text": "Pointers store memory addresses in C.",
                "filename": "chapter_1.pdf",
                "page_start": 4,
                "page_end": 4,
                "source_type": "textbook",
                "distance": 0.25,
            },
            {
                "chunk_text": "Dynamic memory allocation is performed using malloc.",
                "filename": "chapter_1_2.pdf",
                "page_start": 10,
                "page_end": 12,
                "source_type": "textbook",
                "distance": 0.45,
            },
        ]

    def test_format_source_label(self) -> None:
        self.assertEqual(format_source_label("doc.pdf", 5, 5), "doc.pdf, page 5")
        self.assertEqual(format_source_label("doc.pdf", 5, 8), "doc.pdf, pages 5-8")

    def test_format_sources(self) -> None:
        sources = format_sources(self.mock_chunks)
        self.assertEqual(len(sources), 2)
        self.assertEqual(
            sources[0],
            {"label": "Source 1", "filename": "chapter_1.pdf", "page_start": 4, "page_end": 4},
        )
        self.assertEqual(
            sources[1],
            {"label": "Source 2", "filename": "chapter_1_2.pdf", "page_start": 10, "page_end": 12},
        )

    def test_build_qa_prompt(self) -> None:
        prompt = build_qa_prompt("What is malloc?", self.mock_chunks)
        self.assertIn("[Source 1: chapter_1.pdf, page 4]", prompt)
        self.assertIn("[Source 2: chapter_1_2.pdf, pages 10-12]", prompt)
        self.assertIn("Pointers store memory addresses", prompt)
        self.assertIn("What is malloc?", prompt)
        self.assertIn("Cite sources in your answer using the source tags", prompt)
        self.assertIn("I do not have enough information in the provided course materials", prompt)

    @patch("src.generation.qa.search_chunks")
    def test_answer_question_empty_results_skips_llm(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        mock_client = MagicMock()

        result = answer_question("What is Python?", "CSCA48", client=mock_client)
        self.assertIn("I do not have enough information", result["answer"])
        self.assertEqual(result["sources"], [])
        mock_client.generate.assert_not_called()

    @patch("src.generation.qa.search_chunks")
    def test_answer_question_high_distance_skips_llm(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {
                "chunk_text": "Completely unrelated content.",
                "filename": "unrelated.pdf",
                "page_start": 1,
                "page_end": 1,
                "source_type": "textbook",
                "distance": 1.75,
            }
        ]
        mock_client = MagicMock()

        # Tests default distance_threshold (1.5)
        result = answer_question(
            "What is a quantum computer?",
            "CSCA48",
            client=mock_client,
        )
        self.assertIn("I do not have enough information", result["answer"])
        self.assertEqual(result["sources"], [])
        mock_client.generate.assert_not_called()

    @patch("src.generation.qa.search_chunks")
    def test_answer_question_filters_distant_chunks(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {
                "chunk_text": "Pointers store memory addresses.",
                "filename": "pointers.pdf",
                "page_start": 1,
                "page_end": 1,
                "source_type": "textbook",
                "distance": 0.3,
            },
            {
                "chunk_text": "Irrelevant distant chunk.",
                "filename": "distant.pdf",
                "page_start": 5,
                "page_end": 5,
                "source_type": "textbook",
                "distance": 1.8,
            },
        ]
        mock_client = MagicMock()
        mock_client.generate.return_value = "Pointers hold addresses [Source 1]."

        result = answer_question(
            "What is a pointer?",
            "CSCA48",
            distance_threshold=1.5,
            client=mock_client,
        )

        self.assertEqual(result["answer"], "Pointers hold addresses [Source 1].")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["filename"], "pointers.pdf")
        mock_client.generate.assert_called_once()
        # Verify prompt passed to generate only contains the relevant chunk
        passed_prompt = mock_client.generate.call_args[1]["prompt"]
        self.assertIn("pointers.pdf", passed_prompt)
        self.assertNotIn("distant.pdf", passed_prompt)



if __name__ == "__main__":
    unittest.main()
