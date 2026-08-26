from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_TEXT_COLLECTION = "rag_text"
DEFAULT_TEXT_PERSIST_DIR = Path("chroma_db")
PROVIDER_CHROMA = "chroma"
PROVIDER_QDRANT = "qdrant"
SUPPORTED_PROVIDERS = {PROVIDER_CHROMA, PROVIDER_QDRANT}


def get_vector_store_provider(value: str | None = None) -> str:
    provider = value if value is not None else os.getenv("VECTOR_STORE_PROVIDER", PROVIDER_CHROMA)
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(
            f"Unsupported VECTOR_STORE_PROVIDER '{provider}'. Expected one of: {choices}."
        )
    return provider


def get_qdrant_collection_name() -> str:
    return os.getenv("QDRANT_TEXT_COLLECTION", DEFAULT_TEXT_COLLECTION).strip()


def _get_qdrant_connection() -> tuple[str, str, str]:
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = get_qdrant_collection_name()

    if not qdrant_url:
        raise ValueError("QDRANT_URL is required when VECTOR_STORE_PROVIDER=qdrant.")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY is required when VECTOR_STORE_PROVIDER=qdrant.")
    if not collection_name:
        raise ValueError("QDRANT_TEXT_COLLECTION cannot be empty.")

    return qdrant_url, qdrant_api_key, collection_name


def load_text_vectorstore(
    embedding_model: HuggingFaceEmbeddings,
    *,
    provider: str | None = None,
    persist_dir: Path = DEFAULT_TEXT_PERSIST_DIR,
):
    provider = get_vector_store_provider(provider)
    if provider == PROVIDER_QDRANT:
        from langchain_qdrant import QdrantVectorStore, RetrievalMode

        qdrant_url, qdrant_api_key, collection_name = _get_qdrant_connection()
        return QdrantVectorStore.from_existing_collection(
            collection_name=collection_name,
            embedding=embedding_model,
            url=qdrant_url,
            api_key=qdrant_api_key,
            retrieval_mode=RetrievalMode.DENSE,
        )

    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
    )


def _stable_qdrant_id(document: Document) -> str:
    chunk_id = document.metadata.get("chunk_id")
    if not chunk_id:
        raise ValueError("Every document must have metadata['chunk_id'] before Qdrant ingestion.")
    return str(uuid5(NAMESPACE_URL, str(chunk_id)))


def rebuild_qdrant_text_vectorstore(
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


def rebuild_text_vectorstore(
    chunks: Sequence[Document],
    embedding_model: HuggingFaceEmbeddings,
    *,
    provider: str | None = None,
    persist_dir: Path = DEFAULT_TEXT_PERSIST_DIR,
) -> None:
    provider = get_vector_store_provider(provider)
    if provider == PROVIDER_QDRANT:
        rebuild_qdrant_text_vectorstore(chunks, embedding_model)
        return

    from ingestion.store import rebuild_vector_store

    rebuild_vector_store(chunks, embedding_model, persist_dir)
