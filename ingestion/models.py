from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IngestionStatus(str, Enum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    file_name: str
    status: IngestionStatus
    chunk_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    file_name: str
    file_type: str
    content_hash: str
    ingested_at: str
    chunk_count: int

