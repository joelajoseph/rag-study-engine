"""Unit tests for PDF chunking logic."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.chunk import Chunk, chunk_pages, estimate_token_count
from src.ingest.extract import normalize_text


class TestChunking(unittest.TestCase):
    def test_normalize_text_ligatures(self) -> None:
        raw_text = "e\ufb00ect and \ufb01le and \ufb02ow and o\ufb03ce and ba\ufb04e"
        normalized = normalize_text(raw_text)
        self.assertEqual(normalized, "effect and file and flow and office and baffle")

    def test_estimate_token_count(self) -> None:
        text = "Hello world this is a test"
        tokens = estimate_token_count(text)
        self.assertGreater(tokens, 0)
        self.assertEqual(tokens, 8)  # 6 words * 1.3 ceil = 8

    def test_chunk_pages_basic(self) -> None:
        pages = [
            (1, "word " * 400),
            (2, "word " * 400),
        ]
        chunks = chunk_pages(pages, max_tokens=500, overlap_tokens=50)
        self.assertGreater(len(chunks), 1)
        for i, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, Chunk)
            self.assertEqual(chunk.chunk_index, i)
            self.assertGreaterEqual(chunk.page_start, 1)
            self.assertLessEqual(chunk.page_end, 2)

    def test_chunk_pages_invalid_arguments(self) -> None:
        pages = [(1, "some text")]
        with self.assertRaises(ValueError):
            chunk_pages(pages, max_tokens=0)
        with self.assertRaises(ValueError):
            chunk_pages(pages, max_tokens=100, overlap_tokens=100)
        with self.assertRaises(ValueError):
            chunk_pages(pages, max_tokens=100, overlap_tokens=-1)
        with self.assertRaises(ValueError):
            chunk_pages([(0, "invalid page number")])


if __name__ == "__main__":
    unittest.main()

