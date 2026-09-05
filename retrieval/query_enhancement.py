from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from prompts.query import QUERY_PLAN_SYSTEM_PROMPT


MAX_PLANNED_QUERIES = 3


@dataclass(frozen=True)
class QueryPlan:
    """Validated retrieval queries produced for one user request."""

    queries: list[str]

    def __post_init__(self) -> None:
        if not 1 <= len(self.queries) <= MAX_PLANNED_QUERIES:
            raise ValueError("QueryPlan must contain between one and three queries.")
        if any(not isinstance(query, str) or not query.strip() for query in self.queries):
            raise ValueError("QueryPlan queries must be non-empty strings.")


def build_query_plan_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", QUERY_PLAN_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


def _model_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts).strip()
    return str(content).strip()


def parse_query_plan(text: str, original_question: str) -> QueryPlan:
    """Parse strict JSON and always retain the original request as a query."""
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {"queries"}:
        raise ValueError("Query plan must be an object containing only 'queries'.")

    raw_queries = parsed["queries"]
    if not isinstance(raw_queries, list):
        raise ValueError("Query plan 'queries' must be a list.")

    original = original_question.strip()
    queries: list[str] = [original]
    seen = {original.casefold()}
    for value in raw_queries:
        if not isinstance(value, str):
            raise ValueError("Every planned query must be a string.")
        query = value.strip()
        if not query or query.casefold() in seen:
            continue
        queries.append(query)
        seen.add(query.casefold())
        if len(queries) == MAX_PLANNED_QUERIES:
            break

    return QueryPlan(queries=queries)


def plan_queries(question: str, chat_history, llm) -> QueryPlan:
    """Create a validated plan, falling back safely to the original question."""
    original = question.strip()
    if not original:
        raise ValueError("Question cannot be empty.")

    try:
        response = (build_query_plan_prompt() | llm).invoke(
            {"chat_history": chat_history or [], "question": original}
        )
        return parse_query_plan(_model_text(response), original)
    except (json.JSONDecodeError, TypeError, ValueError):
        return QueryPlan(queries=[original])


def rewrite_query(question: str, chat_history, llm) -> str:
    """Backward-compatible single-query wrapper used by evaluation code."""
    return plan_queries(question, chat_history, llm).queries[-1]


def unique_documents(document_groups: Sequence[Sequence[object]]) -> list[object]:
    """Merge retrieval results while preserving first-seen rank."""
    merged: list[object] = []
    seen: set[tuple[object, object, object]] = set()
    for documents in document_groups:
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            identity = (
                metadata.get("chunk_id"),
                metadata.get("document_id"),
                hash(getattr(document, "page_content", "")),
            )
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(document)
    return merged
