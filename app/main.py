import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, StringConstraints

from app.services.rag_service import ask_question
from app.services.upload_service import delete_uploaded_document, upload_document
from ingestion.document_loader import DocumentLoadError
from vectorstore.qdrant_store import get_document_source, list_document_records

# Only files inside this folder are ever served back to the browser.
DATA_DIR = Path("data")


logging.basicConfig(format="%(levelname)s:     %(message)s")
logging.getLogger("app.services.rag_service").setLevel(logging.INFO)

app = FastAPI(
    title="Basic RAG API",
    version="0.1.0"
)

# Let the React dev server (a different port) call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    # Strip surrounding spaces, then require at least 1 character (else FastAPI returns 422).
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Source(BaseModel):
    document: str
    page: int | None = None  # 1-based; only PDFs have pages
    score: float | None = None  # raw cross-encoder rerank score, higher is more relevant


class Citation(BaseModel):
    id: str  # matches the [S1] markers in the answer text
    document_id: str
    document: str
    locator: str  # e.g. "page 4", "slide 2", "Sheet1 rows 2-101"
    page: int | None = None  # 1-based; only PDFs have pages


class ChatResponse(BaseModel):
    answer: str
    mode: str  # "grounded", "general", or "error"
    citations: list[Citation] = []
    sources: list[Source] = []


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    status: str  # "indexed" (new document) or "duplicate" (same content already indexed)
    chunk_count: int
    warnings: list[str] = []


class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    file_type: str  # e.g. "pdf", "docx"
    chunk_count: int
    ingested_at: str  # ISO date/time string


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = ask_question(
        request.query
    )

    return result


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)):
    # "file" is the form field name the client must use.
    if not file.filename:
        raise HTTPException(status_code=400, detail="The uploaded file has no name.")

    try:
        result = upload_document(file.filename, file.file)
    except DocumentLoadError as error:
        # Wrong file type, empty file, too large, or unreadable content.
        raise HTTPException(status_code=400, detail=str(error))

    return result


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    records = list_document_records()

    result = []
    for record in records:
        result.append({
            "document_id": record.document_id,
            "file_name": record.file_name,
            "file_type": record.file_type,
            "chunk_count": record.chunk_count,
            "ingested_at": record.ingested_at,
        })
    return result


@app.get("/documents/{document_id}/file")
def document_file(document_id: str):
    source = get_document_source(document_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    path = Path(source).resolve()
    # Never serve anything outside data/, and the file may be gone (e.g. after a server restart).
    if not path.is_relative_to(DATA_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="The original file is no longer available.")

    # "inline" lets the browser show the file instead of downloading it.
    return FileResponse(path, filename=path.name, content_disposition_type="inline")


@app.delete("/documents/{document_id}", status_code=204)
def remove_document(document_id: str):
    if not delete_uploaded_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")

