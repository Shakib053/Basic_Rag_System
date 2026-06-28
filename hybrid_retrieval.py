from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

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
    semantic_k: int = 6,
    keyword_k: int = 6,
    semantic_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> EnsembleRetriever:
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": semantic_k})
    keyword_retriever = _build_bm25_retriever_from_vectorstore(vectorstore, k=keyword_k)
    return EnsembleRetriever(
        retrievers=[semantic_retriever, keyword_retriever],
        weights=[semantic_weight, keyword_weight],
    )
