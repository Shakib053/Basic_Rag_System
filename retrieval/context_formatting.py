from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from .result import SourceCitation


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


def _source_locator(doc: Document) -> str:
    metadata = doc.metadata
    if metadata.get("source_locator"):
        return str(metadata["source_locator"])
    if isinstance(metadata.get("page"), int):
        return f"page {metadata['page'] + 1}"
    if metadata.get("slide") is not None:
        return f"slide {metadata['slide']}"
    if metadata.get("sheet_name"):
        return str(metadata["sheet_name"])
    return "document"


def format_docs(docs: Sequence[Document]) -> str:
    formatted_chunks = []
    for doc in docs:
        header = _source_header(doc)
        formatted_chunks.append(f"{header}\n\n{doc.page_content}" if header else doc.page_content)
    return "\n\n".join(formatted_chunks)


def build_cited_context(docs: Sequence[Document]) -> tuple[str, list[SourceCitation]]:
    """Format untrusted source text and assign stable IDs for this answer."""
    sections = []
    citations = []
    for index, doc in enumerate(docs, start=1):
        citation_id = f"S{index}"
        file_name = str(doc.metadata.get("file_name") or "unknown file")
        document_id = str(doc.metadata.get("document_id") or "")
        locator = _source_locator(doc)
        citations.append(
            SourceCitation(
                citation_id=citation_id,
                document_id=document_id,
                file_name=file_name,
                locator=locator,
            )
        )
        sections.append(
            f'<source id="{citation_id}" file="{escape(file_name)}" '
            f'location="{escape(locator)}">\n{escape(doc.page_content)}\n</source>'
        )
    return "\n\n".join(sections), citations


def build_combined_context(
    text_docs: Sequence[Document],
    image_results: Sequence[Document] | Sequence[tuple[Document, float]],
) -> str:
    """Backward-compatible formatter; image paths are not answer evidence."""
    text_context, _ = build_cited_context(text_docs)
    return text_context
