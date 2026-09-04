"""
ingestion/text_pipeline.py

Text ingestion pipeline.

Loads .txt and .pdf files from the data directory, chunks them using either
semantic or recursive strategy, embeds with HuggingFace, and rebuilds the
configured text vector store.
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from retrieval.hybrid_retrieval import load_documents, split_documents_with_ids
from chunking.recursive_chunking import RECURSIVE_STRATEGY, build_recursive_chunker
from embeddings.text_embeddings import get_text_embedding_model
from vectorstore.qdrant_store import (
    get_qdrant_collection_name,
    rebuild_text_vectorstore,
)


DATA_DIR = Path("data")

DEFAULT_CHUNKING_STRATEGY = "semantic"
SUPPORTED_CHUNKING_STRATEGIES = {DEFAULT_CHUNKING_STRATEGY, RECURSIVE_STRATEGY}
SEMANTIC_BREAKPOINT_PERCENTILE = 90
SEMANTIC_MIN_CHUNK_SIZE = 200


def get_chunking_strategy(value: str | None = None) -> str:
    """
    Resolve the chunking strategy from an explicit *value*, or fall back to
    the ``CHUNKING_STRATEGY`` environment variable, or the default.
    """
    strategy = (
        value
        if value is not None
        else os.getenv("CHUNKING_STRATEGY", DEFAULT_CHUNKING_STRATEGY)
    )
    strategy = strategy.strip().lower()
    if strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        choices = ", ".join(sorted(SUPPORTED_CHUNKING_STRATEGIES))
        raise ValueError(
            f"Unsupported CHUNKING_STRATEGY '{strategy}'. Expected one of: {choices}."
        )
    return strategy


def _load_semantic_chunker_class():
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError as exc:
        raise RuntimeError(
            "Semantic chunking requires langchain-experimental. Install it with: "
            "python -m pip install langchain-experimental"
        ) from exc
    return SemanticChunker


def build_chunker(strategy: str, embedding_model: HuggingFaceEmbeddings):
    """Return the appropriate text splitter for *strategy*."""
    strategy = get_chunking_strategy(strategy)
    if strategy == RECURSIVE_STRATEGY:
        return build_recursive_chunker()

    semantic_chunker = _load_semantic_chunker_class()
    return semantic_chunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_PERCENTILE,
        min_chunk_size=SEMANTIC_MIN_CHUNK_SIZE,
    )


def run_text_pipeline(strategy: str | None = None) -> bool:
    """
    Build / rebuild the configured text vector store.

    Parameters
    ----------
    strategy:
        Chunking strategy override (``"semantic"`` or ``"recursive"``).
        If *None*, the value is read from the ``CHUNKING_STRATEGY`` env var,
        defaulting to ``"semantic"``.

    Returns
    -------
    bool
        ``True`` if the store was rebuilt, ``False`` if skipped because no
        documents were found.
    """
    strategy = get_chunking_strategy(strategy)
    source_documents = load_documents(DATA_DIR)

    if not source_documents:
        print("⚠  No .txt/.pdf documents found — skipping text pipeline.")
        return False

    # Pre-split long documents so the semantic chunker never creates
    # oversized chunks from very long pages.
    if strategy == "semantic":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
        )
        source_documents = pre_splitter.split_documents(source_documents)
        print(f"Loaded {len(source_documents)} source documents (after pre-splitting)")
    else:
        print(f"Loaded {len(source_documents)} source documents")

    embedding_model = get_text_embedding_model()
    splitter = build_chunker(strategy, embedding_model)
    chunks = split_documents_with_ids(
        source_documents,
        splitter,
        chunking_strategy=strategy,
    )
    print(f"Created {len(chunks)} chunks using {strategy} chunking")

    rebuild_text_vectorstore(chunks, embedding_model)

    print(f"Text data stored in Qdrant collection '{get_qdrant_collection_name()}'")
    return True
