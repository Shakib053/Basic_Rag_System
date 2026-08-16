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

from query_enhancement import is_travel_query

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SUPPORTED_FILE_TYPES = {".txt", ".pdf"}
TRAVEL_FILE_NAME = "Travel_history.txt"

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
) -> list[Document]:
    chunks: list[Document] = []
    location_counters: dict[str, int] = {}

    for document in documents:
        document_chunks = splitter.split_documents([document])
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        location = f"{source}::page-{page}" if page is not None else source

        for chunk in document_chunks:
            index = location_counters.get(location, 0)
            location_counters[location] = index + 1
            
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{location}::chunk-{index}"
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


def _is_travel_document(document: Document) -> bool:
    return document.metadata.get("file_name") == TRAVEL_FILE_NAME


def _refresh_rerank_ranks(documents: Sequence[Document]) -> None:
    for rank, document in enumerate(documents, start=1):
        document.metadata["rerank_rank"] = rank


def select_final_documents(
    query: str,
    documents: Sequence[Document],
    *,
    top_k: int = 5,
    model_name: str = RERANKER_MODEL_NAME,
) -> list[Document]:
    """Rerank documents and keep travel evidence for travel-intent queries."""
    if not documents:
        return []

    reranked_documents = rerank_documents(
        query,
        documents,
        top_k=len(documents),
        model_name=model_name,
    )
    final_documents = list(reranked_documents[:top_k])

    if (
        not is_travel_query(query)
        or any(_is_travel_document(document) for document in final_documents)
    ):
        _refresh_rerank_ranks(final_documents)
        return final_documents

    best_travel_document = next(
        (
            document for document in reranked_documents
            if _is_travel_document(document)
        ),
        None,
    )
    if best_travel_document is None:
        _refresh_rerank_ranks(final_documents)
        return final_documents

    replacement_index = next(
        (
            index for index in range(len(final_documents) - 1, -1, -1)
            if not _is_travel_document(final_documents[index])
        ),
        None,
    )
    if replacement_index is None:
        _refresh_rerank_ranks(final_documents)
        return final_documents

    final_documents[replacement_index] = best_travel_document
    final_documents.sort(
        key=lambda document: float(document.metadata.get("rerank_score", 0.0)),
        reverse=True,
    )
    _refresh_rerank_ranks(final_documents)
    return final_documents
