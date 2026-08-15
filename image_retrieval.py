from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings.clip_embeddings import CLIPEmbeddings

IMAGE_PERSIST_DIR = Path("image_chroma_db")
DEFAULT_IMAGE_RESULTS = 3


def load_image_vectorstore(
    persist_dir: str | Path = IMAGE_PERSIST_DIR,
    *,
    embedding_model=None,
) -> Chroma | None:
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        return None

    return Chroma(
        persist_directory=str(persist_path),
        embedding_function=embedding_model or CLIPEmbeddings(),
    )


def get_image_docs(
    query: str,
    vectorstore: Chroma | None,
    *,
    k: int = DEFAULT_IMAGE_RESULTS,
) -> list[Document]:
    if vectorstore is None:
        return []

    return vectorstore.similarity_search(query, k=k)


def get_image_docs_with_scores(
    query: str,
    vectorstore: Chroma | None,
    *,
    k: int = DEFAULT_IMAGE_RESULTS,
) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []

    return vectorstore.similarity_search_with_score(query, k=k)


def image_path_for_document(document: Document) -> str:
    return str(document.metadata.get("image_path") or document.page_content)


def format_image_references(
    image_results: Sequence[Document] | Sequence[tuple[Document, float]],
) -> list[dict]:
    references: list[dict] = []

    for result in image_results:
        score = None
        document = result
        if isinstance(result, tuple):
            document, score = result

        references.append(
            {
                "image_path": image_path_for_document(document),
                "source": document.metadata.get("source", "unknown source"),
                "page": document.metadata.get("page"),
                "image_index": document.metadata.get("image_index"),
                "score": score,
            }
        )

    return references
