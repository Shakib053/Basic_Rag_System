import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from image_retrieval import (
    format_image_context,
    format_image_references,
    get_image_docs,
    get_image_docs_with_scores,
    image_path_for_document,
    load_image_vectorstore,
)


class ImageVectorStoreLoadingTests(unittest.TestCase):
    @patch("image_retrieval.CLIPEmbeddings")
    @patch("image_retrieval.Chroma")
    def test_loads_image_chroma_store_with_clip_embeddings(self, chroma, clip_embeddings):
        embedding_model = Mock()
        clip_embeddings.return_value = embedding_model

        with patch.object(Path, "exists", return_value=True):
            result = load_image_vectorstore("image_chroma_db")

        chroma.assert_called_once_with(
            persist_directory="image_chroma_db",
            embedding_function=embedding_model,
        )
        self.assertIs(result, chroma.return_value)

    @patch("image_retrieval.Chroma")
    def test_returns_none_when_image_store_is_missing(self, chroma):
        with patch.object(Path, "exists", return_value=False):
            result = load_image_vectorstore("missing_image_chroma_db")

        self.assertIsNone(result)
        chroma.assert_not_called()


class ImageRetrievalTests(unittest.TestCase):
    def test_get_image_docs_queries_image_vectorstore(self):
        vectorstore = Mock()
        image_doc = Document(
            page_content="data/extracted_images/sample/page_1_img_1.png",
            metadata={"file_type": "image"},
        )
        vectorstore.similarity_search.return_value = [image_doc]

        results = get_image_docs("diagram in the PDF", vectorstore, k=2)

        vectorstore.similarity_search.assert_called_once_with("diagram in the PDF", k=2)
        self.assertEqual(results, [image_doc])

    def test_get_image_docs_with_scores_queries_image_vectorstore(self):
        vectorstore = Mock()
        image_doc = Document(
            page_content="data/extracted_images/sample/page_1_img_1.png",
            metadata={"file_type": "image"},
        )
        vectorstore.similarity_search_with_score.return_value = [(image_doc, 0.42)]

        results = get_image_docs_with_scores("architecture image", vectorstore, k=1)

        vectorstore.similarity_search_with_score.assert_called_once_with(
            "architecture image",
            k=1,
        )
        self.assertEqual(results, [(image_doc, 0.42)])

    def test_missing_image_vectorstore_returns_no_results(self):
        self.assertEqual(get_image_docs("anything", None), [])
        self.assertEqual(get_image_docs_with_scores("anything", None), [])

    def test_image_path_uses_metadata_when_present(self):
        document = Document(
            page_content="fallback/path.png",
            metadata={"image_path": "metadata/path.png"},
        )

        self.assertEqual(image_path_for_document(document), "metadata/path.png")

    def test_formats_image_references_with_scores(self):
        document = Document(
            page_content="data/extracted_images/sample/page_2_img_1.png",
            metadata={
                "source": "data/sample.pdf",
                "page": 2,
                "image_index": 1,
                "file_type": "image",
            },
        )

        references = format_image_references([(document, 0.25)])

        self.assertEqual(
            references,
            [
                {
                    "image_path": "data/extracted_images/sample/page_2_img_1.png",
                    "source": "data/sample.pdf",
                    "page": 2,
                    "image_index": 1,
                    "score": 0.25,
                }
            ],
        )

    def test_formats_image_context_with_score(self):
        document = Document(
            page_content="data/extracted_images/sample/page_2_img_1.png",
            metadata={
                "source": "data/sample.pdf",
                "page": 2,
                "image_index": 1,
                "file_type": "image",
            },
        )

        context = format_image_context([(document, 0.25)])

        self.assertEqual(
            context,
            "1. source: data/sample.pdf | page: 2 | image index: 1 | "
            "path: data/extracted_images/sample/page_2_img_1.png | distance: 0.2500",
        )

    def test_formats_image_context_without_score_and_uses_fallback_path(self):
        document = Document(
            page_content="fallback/path.png",
            metadata={"source": "data/sample.pdf"},
        )

        context = format_image_context([document])

        self.assertEqual(
            context,
            "1. source: data/sample.pdf | path: fallback/path.png",
        )


if __name__ == "__main__":
    unittest.main()
