"""One privacy-conscious completion event per user query."""

import json
import logging
import sys
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4


logger = logging.getLogger("app.query_events")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def start_query():
    return uuid4().hex, perf_counter()


def log_query(query_id, started, entry_point, query, mode, reason):
    # Reasons from the pipeline are free-form; only approved labels leave the process.
    allowed_reasons = {
        "document store is empty", "answer model unavailable",
        "retrieval system unavailable", "reranker unavailable",
    }
    event = {
        "event": "query_completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": query_id,
        "entry_point": entry_point,
        "query_length": len(query),
        "duration_ms": round((perf_counter() - started) * 1000, 2),
        "mode": mode,
        "reason": reason if reason in allowed_reasons else ("completed" if mode != "error" else "unexpected error"),
    }
    logger.info(json.dumps(event, separators=(",", ":")))
