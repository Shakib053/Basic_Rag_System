from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from image_retrieval import format_image_context


def format_docs(docs: Sequence[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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
