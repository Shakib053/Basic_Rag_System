"""Guardrails that run around the RAG pipeline.

check_input runs before the pipeline sees the user's query.
check_output runs after the pipeline, before the answer reaches the user.
Both currently pass text through unchanged; checks are added step by step.
"""


def check_input(query: str) -> str:
    """Return the query that is safe to send into the RAG pipeline."""
    return query


def check_output(answer: str) -> str:
    """Return the answer that is safe to show to the user."""
    return answer
