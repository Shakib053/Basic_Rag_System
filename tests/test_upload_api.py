import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from app.services import upload_service
from ingestion.document_loader import DocumentLoadError
from ingestion.models import DocumentRecord, IngestionResult, IngestionStatus


class UploadEndpointTests(unittest.TestCase):
    def setUp(self):
        # Save uploads into a throwaway folder instead of data/uploads.
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.upload_dir = Path(temp_dir.name)

        dir_patch = patch.object(upload_service, "UPLOAD_DIR", self.upload_dir)
        dir_patch.start()
        self.addCleanup(dir_patch.stop)

        reset_patch = patch.object(upload_service, "reset_retrieval_components")
        self.reset_retrieval_components = reset_patch.start()
        self.addCleanup(reset_patch.stop)

        self.client = TestClient(main.app)

    def test_upload_saves_file_indexes_it_and_returns_summary(self):
        result = IngestionResult(
            document_id="doc-1",
            file_name="notes.txt",
            status=IngestionStatus.INDEXED,
            chunk_count=3,
            warnings=["page 2 skipped"],
        )
        with patch.object(upload_service, "ingest_file", return_value=result) as ingest_file:
            response = self.client.post(
                "/upload",
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "document_id": "doc-1",
            "file_name": "notes.txt",
            "status": "indexed",
            "chunk_count": 3,
            "warnings": ["page 2 skipped"],
        })
        saved_path = self.upload_dir / "notes.txt"
        self.assertEqual(saved_path.read_bytes(), b"hello")
        ingest_file.assert_called_once_with(saved_path)
        self.reset_retrieval_components.assert_called_once()

    def test_upload_rejects_bad_file_and_removes_it(self):
        error = DocumentLoadError("Unsupported file type '.zip'.")
        with patch.object(upload_service, "ingest_file", side_effect=error):
            response = self.client.post(
                "/upload",
                files={"file": ("archive.zip", b"PK", "application/zip")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])
        self.assertFalse((self.upload_dir / "archive.zip").exists())
        self.reset_retrieval_components.assert_not_called()

    def test_upload_keeps_file_inside_upload_folder(self):
        result = IngestionResult(
            document_id="doc-2",
            file_name="evil.txt",
            status=IngestionStatus.INDEXED,
            chunk_count=1,
        )
        with patch.object(upload_service, "ingest_file", return_value=result):
            response = self.client.post(
                "/upload",
                files={"file": ("../../evil.txt", b"x", "text/plain")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue((self.upload_dir / "evil.txt").exists())

    def test_upload_without_file_field_returns_422(self):
        with patch.object(upload_service, "ingest_file") as ingest_file:
            response = self.client.post("/upload")

        self.assertEqual(response.status_code, 422)
        ingest_file.assert_not_called()


class DocumentsEndpointTests(unittest.TestCase):
    def test_documents_lists_indexed_files(self):
        records = [
            DocumentRecord(
                document_id="doc-1",
                file_name="cv.pdf",
                file_type="pdf",
                content_hash="abc",
                ingested_at="2026-09-26T10:00:00+00:00",
                chunk_count=12,
            ),
            DocumentRecord(
                document_id="doc-2",
                file_name="notes.txt",
                file_type="txt",
                content_hash="def",
                ingested_at="2026-09-26T11:00:00+00:00",
                chunk_count=2,
            ),
        ]
        with patch.object(main, "list_document_records", return_value=records):
            response = TestClient(main.app).get("/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [
            {
                "document_id": "doc-1",
                "file_name": "cv.pdf",
                "file_type": "pdf",
                "chunk_count": 12,
                "ingested_at": "2026-09-26T10:00:00+00:00",
            },
            {
                "document_id": "doc-2",
                "file_name": "notes.txt",
                "file_type": "txt",
                "chunk_count": 2,
                "ingested_at": "2026-09-26T11:00:00+00:00",
            },
        ])

    def test_documents_returns_empty_list_when_nothing_is_indexed(self):
        with patch.object(main, "list_document_records", return_value=[]):
            response = TestClient(main.app).get("/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])


if __name__ == "__main__":
    unittest.main()
