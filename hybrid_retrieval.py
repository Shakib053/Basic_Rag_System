from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence
import warnings

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SUPPORTED_FILE_TYPES = {".txt", ".pdf"}

DEFAULT_SUBJECT = "Kazi Tanjim Shakib"

def _clean_pdf_headers(pages: list[Document]) -> list[Document]:
    """Detect and strip repeating header lines across multi-page PDFs."""
    if len(pages) <= 1:
        return pages

    header_candidates: dict[str, int] = {}
    for page in pages:
        lines = [line.strip() for line in page.page_content.splitlines() if line.strip()]
        if lines:
            for line in lines[:2]:
                header_candidates[line] = header_candidates.get(line, 0) + 1

    threshold = max(1, len(pages) // 2)
    common_headers = {line for line, count in header_candidates.items() if count > threshold}

    if not common_headers:
        return pages

    cleaned_pages = []
    for page in pages:
        lines = page.page_content.splitlines()
        cleaned_lines = [line for line in lines if line.strip() not in common_headers]
        cleaned_content = "\n".join(cleaned_lines).strip()
        if cleaned_content:
            page.page_content = cleaned_content
            cleaned_pages.append(page)

    return cleaned_pages

def load_documents(data_dir: str | Path) -> list[Document]:
    data_path = Path(data_dir)
    documents: list[Document] = []

    supported_files = sorted(
        (path for path in data_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_TYPES),
        key=lambda path: path.name.lower(),
    )

    for file_path in supported_files:
        file_type = file_path.suffix.lower().lstrip(".")

        if file_type == "txt":
            text = file_path.read_text(encoding="utf-8")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "file_type": file_type,
                        },
                    )
                )
            continue

        try:
            pdf_pages = PyMuPDFLoader(
                str(file_path),
                mode="page",
                extract_tables="markdown",
            ).load()
        except Exception as exc:
            warnings.warn(
                f"Skipping unreadable PDF '{file_path}': {exc}",
                stacklevel=2,
            )
            continue

        pdf_pages = _clean_pdf_headers(pdf_pages)

        readable_pages = 0
        for page in pdf_pages:
            if not page.page_content.strip():
                continue

            page.metadata.update(
                {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_type,
                }
            )
            documents.append(page)
            readable_pages += 1

        if readable_pages == 0:
            warnings.warn(
                f"Skipping PDF with no extractable text: '{file_path}'",
                stacklevel=2,
            )

    return documents

def split_documents_with_ids(
    documents: Sequence[Document],
    splitter,
    *,
    chunking_strategy: str | None = None,
    subject: str = DEFAULT_SUBJECT,
) -> list[Document]:
    chunks: list[Document] = []
    location_counters: dict[str, int] = {}

    for document in documents:
        document_chunks = splitter.split_documents([document])
        source = document.metadata.get("source", "unknown")
        file_name = document.metadata.get("file_name", Path(source).name)
        page = document.metadata.get("page")
        location = f"{source}::page-{page}" if page is not None else source

        for chunk in document_chunks:
            index = location_counters.get(location, 0)
            location_counters[location] = index + 1

            prefix = f"[Subject: {subject} | Source: {file_name}]\n\n"
            if not chunk.page_content.startswith("[Subject:"):
                chunk.page_content = prefix + chunk.page_content.strip()

            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{location}::chunk-{index}"
            chunk.metadata["subject"] = subject
            if chunking_strategy is not None:
                chunk.metadata["chunking_strategy"] = chunking_strategy
            chunks.append(chunk)

    return chunks

def _build_bm25_retriever_from_vectorstore(vectorstore: Chroma, *, k: int) -> BM25Retriever:
    stored = vectorstore.get()
    documents = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(stored.get("documents", []), stored.get("metadatas", []))
    ]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    return bm25_retriever

def build_hybrid_retriever(
    vectorstore: Chroma,
    *,
    semantic_k: int = 12,
    keyword_k: int = 12,
    semantic_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> EnsembleRetriever:
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": semantic_k,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        }
    )
    
    keyword_retriever = _build_bm25_retriever_from_vectorstore(vectorstore, k=keyword_k)
    return EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[semantic_weight, keyword_weight],
    )

@lru_cache(maxsize=1)
def _load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    return CrossEncoder(model_name)

def rerank_documents(
    query: str,
    documents: Sequence[Document],
    *,
    top_k: int = 5,
    model_name: str = RERANKER_MODEL_NAME,
) -> list[Document]:
    if not documents:
        return []

    reranker = _load_reranker(model_name)
    pairs = [(query, document.page_content) for document in documents]
    scores = reranker.predict(pairs)

    scored_documents = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    reranked_documents: list[Document] = []
    for rank, (document, score) in enumerate(scored_documents[:top_k], start=1):
        document.metadata["rerank_score"] = float(score)
        document.metadata["rerank_rank"] = rank
        reranked_documents.append(document)

    return reranked_documents
