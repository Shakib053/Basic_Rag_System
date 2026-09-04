from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryMode(str, Enum):
    DIRECT = "direct"
    RETRIEVE = "retrieve"


@dataclass(frozen=True)
class QueryRoute:
    mode: QueryMode
    reason: str


EXPLICIT_RETRIEVAL_PATTERN = re.compile(
    r"\b(according to|based on|uploaded|indexed)\b|"
    r"\b(my|our|this|that|the)\s+"
    r"(documents?|files?|books?|pdfs?|notes?|profiles?|resumes?|"
    r"reports?|papers?|articles?|data)\b",
    re.IGNORECASE,
)
DOCUMENT_REFERENCE_PATTERN = re.compile(
    r"\b(document|file|book|pdf|chapter|page|notes?|profile|resume|"
    r"report|paper|article|dataset)\b",
    re.IGNORECASE,
)
DIRECT_TASK_PATTERN = re.compile(
    r"^\s*(write|draft|generate|create|compose|code|implement|debug|"
    r"translate|rewrite|proofread|calculate|compute|solve|brainstorm|"
    r"explain|define|teach|summari[sz]e|format|convert|plan|suggest|list)\b|"
    r"^\s*(can|could|would)\s+you\s+"
    r"(write|draft|generate|create|compose|code|implement|debug|"
    r"translate|rewrite|proofread|calculate|compute|solve|brainstorm|"
    r"explain|define|teach|summari[sz]e|format|convert|plan|suggest|list)\b",
    re.IGNORECASE,
)
MATH_PATTERN = re.compile(
    r"^\s*(what is|calculate|compute|solve)\b.*\d.*[+\-*/=]",
    re.IGNORECASE,
)


def route_query(question: str) -> QueryRoute:
    """Choose whether a question should use the LLM directly or retrieve data.

    Retrieval is the conservative default because this application is primarily
    a document question-answering system. Explicit document references always
    take priority over direct-task wording.
    """
    if EXPLICIT_RETRIEVAL_PATTERN.search(question):
        return QueryRoute(QueryMode.RETRIEVE, "explicit document or data reference")

    if DOCUMENT_REFERENCE_PATTERN.search(question):
        return QueryRoute(QueryMode.RETRIEVE, "document-related wording")

    if DIRECT_TASK_PATTERN.search(question):
        return QueryRoute(QueryMode.DIRECT, "general-purpose task wording")

    if MATH_PATTERN.search(question):
        return QueryRoute(QueryMode.DIRECT, "arithmetic or mathematical question")

    return QueryRoute(QueryMode.RETRIEVE, "retrieval is the safe default")
