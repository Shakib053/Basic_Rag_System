"""Save an uploaded file to disk and add it to the search index."""

import shutil
import threading
import uuid
from pathlib import Path

from app.services.rag_service import reset_retrieval_components
from ingestion.document_loader import file_content_hash
from ingestion.text_pipeline import ingest_file
from vectorstore.qdrant_store import delete_document, find_document_by_content_hash, get_document_source

# Every upload gets its own folder, e.g. data/uploads/3f2a9c.../report.pdf.
# The document id comes from the file path, so two files with the same name
# become two separate documents instead of one overwriting the other.
UPLOAD_DIR = Path("data/uploads")

# Only one upload is checked and indexed at a time, so two identical files
# uploaded at the same moment can't both pass the duplicate check.
_upload_lock = threading.Lock()


def save_uploaded_file(file_name: str, file_object) -> Path:
    # Keep only the name part, e.g. "../../x.pdf" becomes "x.pdf".
    safe_name = Path(file_name).name

    # A new random folder for this upload.
    upload_folder = UPLOAD_DIR / uuid.uuid4().hex
    upload_folder.mkdir(parents=True)
    saved_path = upload_folder / safe_name

    # Copy the uploaded bytes into a real file on disk.
    with open(saved_path, "wb") as output_file:
        shutil.copyfileobj(file_object, output_file)

    return saved_path


def upload_document(file_name: str, file_object) -> dict:
    """Save the file, index it, and return a summary for the API response."""
    saved_path = save_uploaded_file(file_name, file_object)

    with _upload_lock:
        try:
            # If exactly the same content is already indexed (under any name),
            # return that document instead of indexing a second copy.
            content_hash = file_content_hash(saved_path)
            existing = find_document_by_content_hash(content_hash)
            if existing is not None:
                shutil.rmtree(saved_path.parent)
                return {
                    "document_id": existing.document_id,
                    "file_name": existing.file_name,
                    "status": "duplicate",
                    "chunk_count": existing.chunk_count,
                    "warnings": [],
                }

            result = ingest_file(saved_path)
        except Exception:
            # Something failed. Remove only this upload's folder; older uploads are untouched.
            shutil.rmtree(saved_path.parent, ignore_errors=True)
            raise

    # The chat retriever caches the index; clear it so new chunks are searchable.
    reset_retrieval_components()

    return {
        "document_id": result.document_id,
        "file_name": result.file_name,
        "status": result.status.value,  # "indexed"
        "chunk_count": result.chunk_count,
        "warnings": result.warnings,
    }


def delete_uploaded_document(document_id: str) -> bool:
    """Remove a document from the index, and its file if it was uploaded. False if not found."""
    with _upload_lock:
        source = get_document_source(document_id)
        if not delete_document(document_id):
            return False

        # Only delete the folder of a web upload (<UPLOAD_DIR>/<id>/<name>);
        # documents ingested from data/ keep their file.
        if source is not None:
            upload_folder = Path(source).resolve().parent
            if upload_folder.parent == UPLOAD_DIR.resolve():
                shutil.rmtree(upload_folder, ignore_errors=True)

    reset_retrieval_components()
    return True
