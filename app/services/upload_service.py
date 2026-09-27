"""Save an uploaded file to disk and add it to the search index."""

import shutil
from pathlib import Path

from app.services.rag_service import reset_retrieval_components
from ingestion.text_pipeline import ingest_file

# Uploaded files are kept here. The same file name always lands on the same
# path, so re-uploading a file replaces its old version in the index.
UPLOAD_DIR = Path("data/uploads")


def save_uploaded_file(file_name: str, file_object) -> Path:
    # Keep only the name part, e.g. "../../x.pdf" becomes "x.pdf".
    safe_name = Path(file_name).name

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOAD_DIR / safe_name

    # Copy the uploaded bytes into a real file on disk.
    with open(saved_path, "wb") as output_file:
        shutil.copyfileobj(file_object, output_file)

    return saved_path


def upload_document(file_name: str, file_object) -> dict:
    """Save the file, index it, and return a summary for the API response."""
    saved_path = save_uploaded_file(file_name, file_object)

    try:
        result = ingest_file(saved_path)
    except Exception:
        # Indexing failed, so don't keep the broken file around.
        saved_path.unlink(missing_ok=True)
        raise

    # The chat retriever caches the index; clear it so new chunks are searchable.
    reset_retrieval_components()

    return {
        "document_id": result.document_id,
        "file_name": result.file_name,
        "status": result.status.value,  # "indexed" or "unchanged"
        "chunk_count": result.chunk_count,
        "warnings": result.warnings,
    }
