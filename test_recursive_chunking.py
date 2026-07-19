import unittest

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from recursive_chunking import (
    RECURSIVE_CHUNK_OVERLAP,
    RECURSIVE_CHUNK_SIZE,
    RECURSIVE_STRATEGY,
    build_recursive_chunker,
)

class RecursiveChunkingTests(unittest.TestCase):
    def test_recursive_strategy_name(self):
        self.assertEqual(RECURSIVE_STRATEGY, "recursive")

    def test_recursive_chunker_uses_expected_configuration(self):
        splitter = build_recursive_chunker()

        self.assertIsInstance(splitter, RecursiveCharacterTextSplitter)
        self.assertEqual(splitter._chunk_size, RECURSIVE_CHUNK_SIZE)
        self.assertEqual(splitter._chunk_overlap, RECURSIVE_CHUNK_OVERLAP)
        self.assertEqual(RECURSIVE_CHUNK_SIZE, 700)
        self.assertEqual(RECURSIVE_CHUNK_OVERLAP, 100)

    def test_recursive_chunker_splits_long_documents(self):
        splitter = build_recursive_chunker()
        document = Document(page_content=("A complete sentence. " * 100).strip())

        chunks = splitter.split_documents([document])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.page_content) <= RECURSIVE_CHUNK_SIZE for chunk in chunks))


if __name__ == "__main__":
    unittest.main()