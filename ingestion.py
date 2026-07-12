from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hybrid_retrieval import load_documents, split_documents_with_ids

DATA_DIR = Path("data")
PERSIST_DIR = Path("chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )
    source_documents = load_documents(DATA_DIR)
    if not source_documents:
        raise RuntimeError(
            f"No usable .txt or .pdf documents found in {DATA_DIR.resolve()}"
        )

    chunks = split_documents_with_ids(source_documents, splitter)
    print(f"Loaded {len(source_documents)} source documents")
    print(f"Created {len(chunks)} chunks")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    rebuild_vector_store(chunks, embedding_model, PERSIST_DIR)
    print(f"Data stored in ChromaDB at {PERSIST_DIR}")


if __name__ == "__main__":
    main()
