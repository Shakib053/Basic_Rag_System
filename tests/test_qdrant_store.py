import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from vectorstore.qdrant_store import _document_ids_filter, upsert_document_chunks


class QdrantStoreTests(unittest.TestCase):
    def test_document_filter_targets_only_selected_metadata_ids(self):
        search_filter = _document_ids_filter(["doc-a", "doc-b"])

        self.assertEqual(search_filter.must[0].key, "metadata.document_id")
        self.assertEqual(search_filter.must[0].match.any, ["doc-a", "doc-b"])

    def test_unchanged_document_skips_embedding_and_upsert(self):
        chunk = Document(
            page_content="same",
            metadata={
                "document_id": "doc-a",
                "content_hash": "hash-a",
                "chunk_id": "doc-a::chunk-0",
            },
        )
        client = Mock()
        client.collection_exists.return_value = True
        client.get_collection.return_value.payload_schema = {}
        existing = SimpleNamespace(payload={"metadata": {"content_hash": "hash-a"}})
        with (
            patch("vectorstore.qdrant_store.get_qdrant_client", return_value=client),
            patch("vectorstore.qdrant_store._scroll_document_points", return_value=[existing]),
            patch("vectorstore.qdrant_store.load_text_vectorstore") as load_store,
        ):
            changed = upsert_document_chunks([chunk], Mock())

        self.assertFalse(changed)
        load_store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
