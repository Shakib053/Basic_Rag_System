from __future__ import annotations

import os
from collections import defaultdict
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.models import DocumentRecord


DEFAULT_TEXT_COLLECTION = "rag_text_v2"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


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


def get_qdrant_client():
    from qdrant_client import QdrantClient

    qdrant_url, qdrant_api_key, _ = _get_qdrant_connection()
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def text_collection_exists() -> bool:
    return get_qdrant_client().collection_exists(get_qdrant_collection_name())


def ensure_document_id_index(client=None) -> None:
    """Ensure filtered document operations work on existing collections too."""
    client = client or get_qdrant_client()
    collection_name = get_qdrant_collection_name()
    if not client.collection_exists(collection_name):
        return
    collection = client.get_collection(collection_name)
    payload_schema = getattr(collection, "payload_schema", {}) or {}
    if not isinstance(payload_schema, dict):
        payload_schema = {}
    if "metadata.document_id" in payload_schema:
        return

    from qdrant_client import models

    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )


def get_sparse_embedding_model():
    try:
        from langchain_qdrant import FastEmbedSparse
    except ImportError as exc:
        raise RuntimeError(
            "Hybrid retrieval requires fastembed. Install project dependencies first."
        ) from exc
    return FastEmbedSparse(model_name=os.getenv("SPARSE_EMBEDDING_MODEL", DEFAULT_SPARSE_MODEL))


def load_text_vectorstore(
    embedding_model: HuggingFaceEmbeddings,
):
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    qdrant_url, qdrant_api_key, collection_name = _get_qdrant_connection()
    ensure_document_id_index()
    return QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embedding_model,
        url=qdrant_url,
        api_key=qdrant_api_key,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=get_sparse_embedding_model(),
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
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_embedding=get_sparse_embedding_model(),
        force_recreate=True,
    )
    ensure_document_id_index()


def _document_filter(document_id: str):
    from qdrant_client import models

    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )


def _document_ids_filter(document_ids: Sequence[str]):
    from qdrant_client import models

    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.document_id",
                match=models.MatchAny(any=list(document_ids)),
            )
        ]
    )


def retrieve_text_documents(
    vectorstore,
    query: str,
    *,
    k: int = 20,
    document_ids: Sequence[str] | None = None,
) -> list[Document]:
    search_filter = _document_ids_filter(document_ids) if document_ids else None
    return vectorstore.similarity_search(query, k=k, filter=search_filter)


def _scroll_document_points(client, collection_name: str, document_id: str):
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=_document_filter(document_id),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            return points


def upsert_document_chunks(
    chunks: Sequence[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> bool:
    """Create or replace one document. Return False when its content is unchanged."""
    if not chunks:
        raise ValueError("Cannot index a document without chunks.")
    document_id = str(chunks[0].metadata["document_id"])
    content_hash = str(chunks[0].metadata["content_hash"])
    if any(str(chunk.metadata.get("document_id")) != document_id for chunk in chunks):
        raise ValueError("All chunks in one upsert must belong to the same document.")

    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    qdrant_url, qdrant_api_key, collection_name = _get_qdrant_connection()
    client = get_qdrant_client()
    collection_exists = client.collection_exists(collection_name)
    if collection_exists:
        ensure_document_id_index(client)
    existing_points = (
        _scroll_document_points(client, collection_name, document_id)
        if collection_exists
        else []
    )
    existing_hashes = {
        (point.payload or {}).get("metadata", {}).get("content_hash")
        for point in existing_points
    }
    if existing_points and existing_hashes == {content_hash}:
        return False

    ids = [_stable_qdrant_id(chunk) for chunk in chunks]
    if collection_exists:
        vectorstore = load_text_vectorstore(embedding_model)
        vectorstore.add_documents(list(chunks), ids=ids)
    else:
        vectorstore = QdrantVectorStore.from_documents(
            documents=list(chunks),
            embedding=embedding_model,
            ids=ids,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_embedding=get_sparse_embedding_model(),
        )

    stale_ids = [point.id for point in existing_points if point.id not in set(ids)]
    if stale_ids:
        vectorstore.delete(ids=stale_ids)
    return True


def delete_document(document_id: str) -> bool:
    from qdrant_client import models

    client = get_qdrant_client()
    collection_name = get_qdrant_collection_name()
    if not client.collection_exists(collection_name):
        return False
    ensure_document_id_index(client)
    points = _scroll_document_points(client, collection_name, document_id)
    if not points:
        return False
    result = client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=_document_filter(document_id)),
        wait=True,
    )
    return result.status == models.UpdateStatus.COMPLETED


def list_document_records() -> list[DocumentRecord]:
    client = get_qdrant_client()
    collection_name = get_qdrant_collection_name()
    if not client.collection_exists(collection_name):
        return []

    grouped: dict[str, dict] = defaultdict(dict)
    counts: dict[str, int] = defaultdict(int)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            metadata = (point.payload or {}).get("metadata", {})
            document_id = metadata.get("document_id")
            if not document_id:
                continue
            grouped[document_id] = metadata
            counts[document_id] += 1
        if offset is None:
            break

    return sorted(
        [
            DocumentRecord(
                document_id=document_id,
                file_name=str(metadata.get("file_name", "unknown")),
                file_type=str(metadata.get("file_type", "unknown")),
                content_hash=str(metadata.get("content_hash", "")),
                ingested_at=str(metadata.get("ingested_at", "")),
                chunk_count=counts[document_id],
            )
            for document_id, metadata in grouped.items()
        ],
        key=lambda record: record.file_name.casefold(),
    )
