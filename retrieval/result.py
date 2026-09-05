from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class AnswerMode(str, Enum):
    GROUNDED = "grounded"
    GENERAL = "general"
    ERROR = "error"


@dataclass(frozen=True)
class SourceCitation:
    citation_id: str
    document_id: str
    file_name: str
    locator: str


@dataclass(frozen=True)
class AnswerResult:
    text: str
    mode: AnswerMode
    citations: list[SourceCitation] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    reason: str = ""


_CITATION_PATTERN = re.compile(r"\[(S\d+)(?:,[^\]]*)?\]")


def cited_ids(text: str) -> set[str]:
    return set(_CITATION_PATTERN.findall(text))


def remove_invalid_citations(text: str, citations: list[SourceCitation]) -> str:
    """Remove citation markers that were not present in the supplied context."""
    valid_ids = {citation.citation_id for citation in citations}
    return _CITATION_PATTERN.sub(
        lambda match: match.group(0) if match.group(1) in valid_ids else "",
        text,
    )

