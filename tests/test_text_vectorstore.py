import unittest

from langchain_core.documents import Document

from ingestion.text_vectorstore import (
    _stable_qdrant_id,
)


class TextVectorStoreTests(unittest.TestCase):
    def test_qdrant_ids_are_stable_for_chunk_id(self):
        document = Document(
            page_content="Text",
            metadata={"chunk_id": "data/a.txt::chunk-0"},
        )

        self.assertEqual(_stable_qdrant_id(document), _stable_qdrant_id(document))

    def test_qdrant_ids_require_chunk_id(self):
        with self.assertRaisesRegex(ValueError, "chunk_id"):
            _stable_qdrant_id(Document(page_content="Text"))


if __name__ == "__main__":
    unittest.main()
