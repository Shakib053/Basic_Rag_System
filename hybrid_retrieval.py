from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_txt_documents(data_dir: str | Path) -> list[Document]:
    data_path = Path(data_dir)
    documents: list[Document] = []

    for file_path in sorted(data_path.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                },
            )
        )

    return documents

def split_documents_with_ids(documents: Sequence[Document], splitter) -> list[Document]:
    chunks = splitter.split_documents(list(documents))

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{source}::chunk-{index}"

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
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": semantic_k})
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
