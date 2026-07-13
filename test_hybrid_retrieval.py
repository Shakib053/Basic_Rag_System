import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_retrieval import load_documents, split_documents_with_ids


class DocumentLoadingTests(unittest.TestCase):
    def test_loads_txt_with_expected_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("Useful text", encoding="utf-8")

            documents = load_documents(directory)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "Useful text")
        self.assertEqual(documents[0].metadata["file_name"], "notes.txt")
        self.assertEqual(documents[0].metadata["file_type"], "txt")

    @patch("hybrid_retrieval.PyPDFLoader")
    def test_loads_only_nonempty_pdf_pages(self, loader_class):
        loader_class.return_value.load.return_value = [
            Document(page_content="First page", metadata={"page": 0}),
            Document(page_content="   ", metadata={"page": 1}),
            Document(page_content="Third page", metadata={"page": 2}),
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.pdf"
            path.touch()
            documents = load_documents(directory)

        self.assertEqual([doc.metadata["page"] for doc in documents], [0, 2])
        self.assertTrue(all(doc.metadata["file_name"] == "guide.pdf" for doc in documents))
        self.assertTrue(all(doc.metadata["file_type"] == "pdf" for doc in documents))

    @patch("hybrid_retrieval.warnings.warn")
    @patch("hybrid_retrieval.PyPDFLoader")
    def test_warns_and_continues_for_unreadable_pdf(self, loader_class, warn):
        loader_class.return_value.load.side_effect = ValueError("broken")

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory)
            (data_path / "broken.pdf").touch()
            (data_path / "valid.txt").write_text("Still loaded", encoding="utf-8")
            documents = load_documents(directory)

        self.assertEqual([doc.metadata["file_name"] for doc in documents], ["valid.txt"])
        self.assertIn("Skipping unreadable PDF", warn.call_args.args[0])

    def test_chunk_ids_are_unique_and_stable_per_page(self):
        documents = [
            Document(page_content="A" * 30, metadata={"source": "data/a.pdf", "page": 0}),
            Document(page_content="B" * 30, metadata={"source": "data/a.pdf", "page": 1}),
        ]
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)

        chunks = split_documents_with_ids(documents, splitter)
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(ids[0].startswith("data/a.pdf::page-0::chunk-"))
        self.assertTrue(ids[-1].startswith("data/a.pdf::page-1::chunk-"))

    def test_chunk_metadata_is_preserved_and_strategy_is_recorded(self):
        documents = [
            Document(
                page_content="A complete sentence about one topic.",
                metadata={
                    "source": "data/guide.pdf",
                    "file_name": "guide.pdf",
                    "file_type": "pdf",
                    "page": 2,
                },
            )
        ]
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

        chunks = split_documents_with_ids(
            documents,
            splitter,
            chunking_strategy="semantic",
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["source"], "data/guide.pdf")
        self.assertEqual(chunks[0].metadata["file_name"], "guide.pdf")
        self.assertEqual(chunks[0].metadata["file_type"], "pdf")
        self.assertEqual(chunks[0].metadata["page"], 2)
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[0].metadata["chunking_strategy"], "semantic")

    def test_semantic_splitter_topic_boundaries_become_separate_chunks(self):
        class DeterministicSemanticSplitter:
            def split_documents(self, documents):
                source = documents[0]
                return [
                    Document(page_content=text.strip(), metadata=dict(source.metadata))
                    for text in source.page_content.split("<topic-change>")
                ]

        documents = [
            Document(
                page_content=(
                    "Python is used for the backend."
                    "<topic-change>Dhaka was the destination of the trip."
                ),
                metadata={"source": "data/profile.txt"},
            )
        ]

        chunks = split_documents_with_ids(
            documents,
            DeterministicSemanticSplitter(),
            chunking_strategy="semantic",
        )

        self.assertEqual(
            [chunk.page_content for chunk in chunks],
            [
                "Python is used for the backend.",
                "Dhaka was the destination of the trip.",
            ],
        )
        self.assertEqual([chunk.metadata["chunk_index"] for chunk in chunks], [0, 1])


if __name__ == "__main__":
    unittest.main()
