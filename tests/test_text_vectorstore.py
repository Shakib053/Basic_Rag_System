import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from text_vectorstore import (
    PROVIDER_CHROMA,
    PROVIDER_QDRANT,
    _stable_qdrant_id,
    get_vector_store_provider,
)


class TextVectorStoreProviderTests(unittest.TestCase):
    @patch.dict("text_vectorstore.os.environ", {}, clear=True)
    def test_chroma_is_default_provider(self):
        self.assertEqual(get_vector_store_provider(), PROVIDER_CHROMA)

    @patch.dict(
        "text_vectorstore.os.environ",
        {"VECTOR_STORE_PROVIDER": "qdrant"},
        clear=True,
    )
    def test_qdrant_provider_can_be_selected_from_environment(self):
        self.assertEqual(get_vector_store_provider(), PROVIDER_QDRANT)

    def test_invalid_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported VECTOR_STORE_PROVIDER"):
            get_vector_store_provider("sqlite")

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
