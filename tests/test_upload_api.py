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

        # By default nothing with the same content is indexed yet.
        find_patch = patch.object(upload_service, "find_document_by_content_hash", return_value=None)
        self.find_document_by_content_hash = find_patch.start()
        self.addCleanup(find_patch.stop)

        self.client = TestClient(main.app)

    def saved_files(self):
        # Every upload lives in its own folder: <upload_dir>/<random id>/<name>
        return sorted(self.upload_dir.glob("*/*"))

    def indexed_result(self, document_id, file_name):
        return IngestionResult(
            document_id=document_id,
            file_name=file_name,
            status=IngestionStatus.INDEXED,
            chunk_count=1,
        )

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
        [saved_path] = self.saved_files()
        self.assertEqual(saved_path.name, "notes.txt")
        self.assertEqual(saved_path.read_bytes(), b"hello")
        ingest_file.assert_called_once_with(saved_path)
        self.reset_retrieval_components.assert_called_once()

    def test_same_name_different_content_keeps_both_documents(self):
        results = [self.indexed_result("doc-1", "notes.txt"), self.indexed_result("doc-2", "notes.txt")]
        with patch.object(upload_service, "ingest_file", side_effect=results) as ingest_file:
            first = self.client.post("/upload", files={"file": ("notes.txt", b"apple", "text/plain")})
            second = self.client.post("/upload", files={"file": ("notes.txt", b"banana", "text/plain")})

        self.assertEqual(first.json()["document_id"], "doc-1")
        self.assertEqual(second.json()["document_id"], "doc-2")
        saved = self.saved_files()
        self.assertEqual(len(saved), 2)
        self.assertNotEqual(saved[0].parent, saved[1].parent)
        self.assertEqual({path.read_bytes() for path in saved}, {b"apple", b"banana"})
        self.assertEqual(ingest_file.call_count, 2)

    def test_same_content_returns_existing_document_as_duplicate(self):
        existing = DocumentRecord(
            document_id="doc-1",
            file_name="notes.txt",
            file_type="txt",
            content_hash="abc",
            ingested_at="2026-09-27T10:00:00+00:00",
            chunk_count=4,
        )
        self.find_document_by_content_hash.return_value = existing
        with patch.object(upload_service, "ingest_file") as ingest_file:
            response = self.client.post("/upload", files={"file": ("copy.txt", b"apple", "text/plain")})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "document_id": "doc-1",
            "file_name": "notes.txt",
            "status": "duplicate",
            "chunk_count": 4,
            "warnings": [],
        })
        ingest_file.assert_not_called()
        self.assertEqual(self.saved_files(), [])
        self.reset_retrieval_components.assert_not_called()

    def test_failed_upload_removes_only_its_own_file(self):
        with patch.object(upload_service, "ingest_file", return_value=self.indexed_result("doc-1", "notes.txt")):
            self.client.post("/upload", files={"file": ("notes.txt", b"good", "text/plain")})

        error = DocumentLoadError("The file is empty.")
        with patch.object(upload_service, "ingest_file", side_effect=error):
            response = self.client.post("/upload", files={"file": ("notes.txt", b"bad", "text/plain")})

        self.assertEqual(response.status_code, 400)
        self.assertIn("The file is empty.", response.json()["detail"])
        [remaining] = self.saved_files()
        self.assertEqual(remaining.read_bytes(), b"good")

    def test_server_error_during_duplicate_check_removes_the_upload(self):
        self.find_document_by_content_hash.side_effect = RuntimeError("Qdrant is down")
        client = TestClient(main.app, raise_server_exceptions=False)
        response = client.post("/upload", files={"file": ("notes.txt", b"apple", "text/plain")})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.saved_files(), [])

    def test_upload_keeps_file_inside_upload_folder(self):
        with patch.object(upload_service, "ingest_file", return_value=self.indexed_result("doc-2", "evil.txt")):
            response = self.client.post(
                "/upload",
                files={"file": ("../../evil.txt", b"x", "text/plain")},
            )

        self.assertEqual(response.status_code, 200)
        [saved_path] = self.saved_files()
        self.assertEqual(saved_path.name, "evil.txt")
        self.assertEqual(saved_path.parent.parent, self.upload_dir)

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
