from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from hybrid_retrieval import load_documents, split_documents_with_ids
from recursive_chunking import RECURSIVE_STRATEGY, build_recursive_chunker
from embeddings.text_embeddings import get_text_embedding_model

DATA_DIR = Path("data")
PERSIST_DIR = Path("chroma_db")
DEFAULT_CHUNKING_STRATEGY = "semantic"
SUPPORTED_CHUNKING_STRATEGIES = {
    DEFAULT_CHUNKING_STRATEGY,
    RECURSIVE_STRATEGY
}
SEMANTIC_BREAKPOINT_PERCENTILE = 90
SEMANTIC_MIN_CHUNK_SIZE = 200

def get_chunking_strategy(value: str | None = None) -> str:
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


def verify_chroma_native_bindings() -> None:
    try:
        import chromadb_rust_bindings.chromadb_rust_bindings  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Chroma's native bindings are missing. Reinstall the binary wheel with:\n"
            "python -m pip install --force-reinstall --no-cache-dir --no-deps "
            "--only-binary=:all: chromadb==1.5.9"
        ) from exc

def build_vector_store(
    chunks: Sequence[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: Path,
) -> None:
    vector_store = Chroma.from_documents(
        documents=list(chunks),
        embedding=embedding_model,
        persist_directory=str(persist_dir),
        ids=[doc.metadata["chunk_id"] for doc in chunks],
    )
    del vector_store
    gc.collect()

def _rename_path(source: Path, destination: Path) -> None:
    source.rename(destination)


def replace_vector_store(staging_dir: Path, persist_dir: Path) -> None:
    backup_dir = persist_dir.with_name(
        f".{persist_dir.name}.backup-{uuid4().hex}"
    )
    moved_live_store = False

    if persist_dir.exists():
        _rename_path(persist_dir, backup_dir)
        moved_live_store = True

    try:
        _rename_path(staging_dir, persist_dir)
    except Exception:
        if moved_live_store and backup_dir.exists() and not persist_dir.exists():
            _rename_path(backup_dir, persist_dir)
        raise

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

def rebuild_vector_store(
    chunks: Sequence[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_dir: Path,
    *,
    builder: Callable[[Sequence[Document], HuggingFaceEmbeddings, Path], None] = build_vector_store,
) -> None:
    staging_dir = persist_dir.with_name(
        f".{persist_dir.name}.staging-{uuid4().hex}"
    )

    try:
        builder(chunks, embedding_model, staging_dir)
        replace_vector_store(staging_dir, persist_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

def main() -> None:
    load_dotenv()
    verify_chroma_native_bindings()

    strategy = get_chunking_strategy()
    source_documents = load_documents(DATA_DIR)
    if not source_documents:
        raise RuntimeError(
            f"No usable .txt or .pdf documents found in {DATA_DIR.resolve()}"
        )

    # Pre-split long documents to ensure semantic chunker doesn't create oversized chunks
    if strategy == "semantic":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
        )
        source_documents = pre_splitter.split_documents(source_documents)

    embedding_model = get_text_embedding_model()
    splitter = build_chunker(strategy, embedding_model)
    chunks = split_documents_with_ids(
        source_documents,
        splitter,
        chunking_strategy=strategy,
    )
    print(f"Loaded {len(source_documents)} source documents (after pre-splitting)" if strategy == "semantic" else f"Loaded {len(source_documents)} source documents")
    print(f"Created {len(chunks)} chunks using {strategy} chunking")

    rebuild_vector_store(chunks, embedding_model, PERSIST_DIR)
    print(f"Data stored in ChromaDB at {PERSIST_DIR}")


if __name__ == "__main__":
    main()