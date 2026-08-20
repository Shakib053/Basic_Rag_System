from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from image_retrieval import format_image_context


def _source_header(doc: Document) -> str:
    file_name = doc.metadata.get("file_name")
    if not file_name:
        source = doc.metadata.get("source")
        file_name = Path(source).name if source else None
    if not file_name:
        return ""

    parts = [f"Source: {file_name}"]
    page = doc.metadata.get("page")
    if isinstance(page, int):
        parts.append(f"page {page + 1}")

    return f"[{' | '.join(parts)}]"


def format_docs(docs: Sequence[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        header = _source_header(doc)
        formatted_chunks.append(f"{header}\n\n{doc.page_content}" if header else doc.page_content)
    return "\n\n".join(formatted_chunks)


def build_combined_context(
    text_docs: Sequence[Document],
    image_results: Sequence[Document] | Sequence[tuple[Document, float]],
) -> str:
    text_context = format_docs(text_docs)
    image_context = format_image_context(image_results)

    if not image_context:
        return text_context

    sections = []
    if text_context:
        sections.append(f"Text context:\n{text_context}")
    sections.append(f"Image references:\n{image_context}")

    return "\n\n".join(sections)
