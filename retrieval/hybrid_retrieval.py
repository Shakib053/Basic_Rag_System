from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ingestion.document_loader import load_documents_from_directory


RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

def load_documents(data_dir: str | Path) -> list[Document]:
    documents, warnings = load_documents_from_directory(data_dir)
    for warning in warnings:
        print(f"Warning: {warning}")
    return documents

def split_documents_with_ids(
    documents: Sequence[Document],
    splitter,
    *,
    chunking_strategy: str | None = None,
) -> list[Document]:
    chunks: list[Document] = []
    location_counters: dict[str, int] = {}

    for document in documents:
        document_chunks = splitter.split_documents([document])
        document_id = document.metadata.get("document_id", "unknown")
        source_locator = document.metadata.get("source_locator", "document")
        location = f"{document_id}::{source_locator}"

        for chunk in document_chunks:
            index = location_counters.get(location, 0)
            location_counters[location] = index + 1

            chunk.page_content = chunk.page_content.strip()
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{location}::chunk-{index}"
            if chunking_strategy is not None:
                chunk.metadata["chunking_strategy"] = chunking_strategy
            chunks.append(chunk)

    return chunks

def _metadata_search_terms(document: Document) -> str:
    metadata = document.metadata
    values = [
        metadata.get("file_name"),
        metadata.get("source"),
        metadata.get("file_type"),
        metadata.get("source_locator"),
        metadata.get("sheet_name"),
    ]
    terms = []
    for value in values:
        if not value:
            continue
        text = str(value)
        terms.append(text)
        terms.append(Path(text).stem.replace("_", " "))
    return " ".join(terms)


def build_hybrid_retriever(
    vectorstore,
    *,
    k: int = 20,
):
    """Build a retriever over a Qdrant collection configured for hybrid RRF."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

@lru_cache(maxsize=1)
def _load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    offline = os.getenv("HF_HUB_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}
    return CrossEncoder(model_name, local_files_only=offline)

def rerank_documents(
    query: str,
    documents: Sequence[Document],
    *,
    top_k: int = 5,
    model_name: str = RERANKER_MODEL_NAME,
) -> list[Document]:
    if not documents:
        return []

    unique_documents: list[Document] = []
    seen_documents = set()
    for document in documents:
        identity = _document_identity(document)
        if identity == (None, None, None):
            identity = (None, None, hash(document.page_content))
        if identity in seen_documents:
            continue
        seen_documents.add(identity)
        unique_documents.append(document)

    reranker = _load_reranker(model_name)
    pairs = [
        (query, f"{_metadata_search_terms(document)}\n\n{document.page_content}".strip())
        for document in unique_documents
    ]
    scores = reranker.predict(pairs)

    scored_documents = sorted(
        zip(unique_documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    reranked_documents: list[Document] = []
    for rank, (document, score) in enumerate(scored_documents[:top_k], start=1):
        document.metadata["rerank_score"] = float(score)
        document.metadata["rerank_rank"] = rank
        reranked_documents.append(document)

    return reranked_documents


def _document_identity(document: Document) -> tuple[str | None, str | None, int | None]:
    metadata = document.metadata
    return (
        metadata.get("chunk_id"),
        metadata.get("source"),
        metadata.get("chunk_index"),
    )


def select_final_context_documents(
    query: str,
    candidate_documents: Sequence[Document],
    *,
    rerank_top_k: int = 5,
) -> list[Document]:
    """Return only cross-encoder-reranked documents."""
    return rerank_documents(
        query,
        candidate_documents,
        top_k=rerank_top_k,
    )
