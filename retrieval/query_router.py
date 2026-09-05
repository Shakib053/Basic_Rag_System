from __future__ import annotations

import os
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence


class QueryMode(str, Enum):
    DIRECT = "direct"
    RETRIEVE = "retrieve"


@dataclass(frozen=True)
class QueryRoute:
    mode: QueryMode
    reason: str


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
THRESHOLD_CONFIG_PATH = Path(__file__).with_name("relevance_thresholds.json")


def get_relevance_threshold() -> float:
    """Return the threshold calibrated for the configured reranker model."""
    override = os.getenv("RERANK_RELEVANCE_THRESHOLD")
    if override is not None:
        return float(override)
    model_name = os.getenv("RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL)
    try:
        config = json.loads(THRESHOLD_CONFIG_PATH.read_text(encoding="utf-8"))
        return float(config["models"][model_name]["threshold"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"No calibrated relevance threshold is configured for reranker '{model_name}'."
        ) from exc


def route_query(question: str) -> QueryRoute:
    """Route every natural-language request through corpus retrieval first."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    return QueryRoute(QueryMode.RETRIEVE, "corpus-first routing")


def route_retrieval_result(
    documents: Sequence[object],
    *,
    threshold: float | None = None,
) -> QueryRoute:
    """Choose a final route after reranking retrieved documents.

    The configured boundary must be calibrated against the configured reranker.
    """
    scores = []
    for document in documents:
        metadata = getattr(document, "metadata", {})
        score = metadata.get("rerank_score") if isinstance(metadata, dict) else None
        if isinstance(score, (int, float)):
            scores.append(float(score))

    if not scores:
        return QueryRoute(QueryMode.DIRECT, "no reranked document match")

    best_score = max(scores)
    relevance_threshold = get_relevance_threshold() if threshold is None else threshold
    if best_score < relevance_threshold:
        return QueryRoute(
            QueryMode.DIRECT,
            f"no relevant document match (best rerank score: {best_score:.2f})",
        )

    return QueryRoute(
        QueryMode.RETRIEVE,
        f"relevant document match (best rerank score: {best_score:.2f})",
    )


def relevant_documents(
    documents: Sequence[object],
    *,
    threshold: float | None = None,
) -> list[object]:
    """Return only reranked documents that meet the relevance boundary."""
    relevance_threshold = get_relevance_threshold() if threshold is None else threshold
    return [
        document
        for document in documents
        if isinstance(getattr(document, "metadata", None), dict)
        and isinstance(document.metadata.get("rerank_score"), (int, float))
        and float(document.metadata["rerank_score"]) >= relevance_threshold
    ]
