"""
ingestion/store.py

Shared vector-store helpers used by both text_pipeline and image_pipeline.
These utilities handle building, staging, and atomically swapping Chroma
databases so that a failed rebuild never corrupts the live store.
"""
from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def verify_chroma_native_bindings() -> None:
    """Raise RuntimeError if Chroma's native Rust bindings are not installed."""
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
    """Create a Chroma vector store from *chunks* and persist it to *persist_dir*."""
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
    """
    Atomically promote *staging_dir* to *persist_dir*.

    If *persist_dir* already exists it is moved to a hidden backup first.
    If the promotion fails the backup is restored, leaving the live store intact.
    """
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
    """
    Build a new vector store in a staging directory and atomically swap it
    into place, so the live store is never left in a partial state.
    """
    staging_dir = persist_dir.with_name(
        f".{persist_dir.name}.staging-{uuid4().hex}"
    )

    try:
        builder(chunks, embedding_model, staging_dir)
        replace_vector_store(staging_dir, persist_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
