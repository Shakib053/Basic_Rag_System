from __future__ import annotations

import os
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_TEXT_COLLECTION = "rag_text"


def get_qdrant_collection_name() -> str:
    return os.getenv("QDRANT_TEXT_COLLECTION", DEFAULT_TEXT_COLLECTION).strip()


def _get_qdrant_connection() -> tuple[str, str, str]:
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = get_qdrant_collection_name()

    if not qdrant_url:
        raise ValueError("QDRANT_URL is required for Qdrant text vector storage.")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY is required for Qdrant text vector storage.")
    if not collection_name:
        raise ValueError("QDRANT_TEXT_COLLECTION cannot be empty.")

    return qdrant_url, qdrant_api_key, collection_name


def load_text_vectorstore(
    embedding_model: HuggingFaceEmbeddings,
):
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    qdrant_url, qdrant_api_key, collection_name = _get_qdrant_connection()
    return QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embedding_model,
        url=qdrant_url,
        api_key=qdrant_api_key,
        retrieval_mode=RetrievalMode.DENSE,
    )


def _stable_qdrant_id(document: Document) -> str:
    chunk_id = document.metadata.get("chunk_id")
    if not chunk_id:
        raise ValueError("Every document must have metadata['chunk_id'] before Qdrant ingestion.")
    return str(uuid5(NAMESPACE_URL, str(chunk_id)))


def rebuild_text_vectorstore(
    chunks: Sequence[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> None:
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    qdrant_url, qdrant_api_key, collection_name = _get_qdrant_connection()
    QdrantVectorStore.from_documents(
        documents=list(chunks),
        embedding=embedding_model,
        ids=[_stable_qdrant_id(document) for document in chunks],
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.DENSE,
        force_recreate=True,
    )
