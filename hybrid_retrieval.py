from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document

TOKEN_PATTERN = re.compile(r"\b\w+\b")
DEFAULT_BM25_PATH = Path("./bm25_corpus.json")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_txt_documents(data_dir: str | Path) -> list[Document]:
    data_path = Path(data_dir)
    documents: list[Document] = []

    for file_path in sorted(data_path.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                },
            )
        )

    return documents


def split_documents_with_ids(documents: Sequence[Document], splitter) -> list[Document]:
    chunks = splitter.split_documents(list(documents))

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{source}::chunk-{index}"

    return chunks


@dataclass
class ScoredDocument:
    document: Document
    score: float
    rank: int


class BM25Index:
    def __init__(
        self,
        documents: Sequence[Document],
        tokenized_documents: Sequence[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        self.tokenized_documents = [list(tokens) for tokens in tokenized_documents]
        self.k1 = k1
        self.b = b
        self.doc_len = [len(tokens) for tokens in self.tokenized_documents]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.doc_freqs = self._build_doc_freqs()
        self.idf = self._build_idf()

    @classmethod
    def from_documents(cls, documents: Sequence[Document]) -> "BM25Index":
        tokenized_documents = [tokenize(doc.page_content) for doc in documents]
        return cls(documents, tokenized_documents)

    def _build_doc_freqs(self) -> list[dict[str, int]]:
        frequencies: list[dict[str, int]] = []
        for tokens in self.tokenized_documents:
            doc_freq: dict[str, int] = {}
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
            frequencies.append(doc_freq)
        return frequencies

    def _build_idf(self) -> dict[str, float]:
        document_count = len(self.tokenized_documents)
        doc_occurrences: dict[str, int] = {}

        for tokens in self.tokenized_documents:
            for token in set(tokens):
                doc_occurrences[token] = doc_occurrences.get(token, 0) + 1

        idf: dict[str, float] = {}
        for token, occurrences in doc_occurrences.items():
            idf[token] = math.log(1 + ((document_count - occurrences + 0.5) / (occurrences + 0.5)))
        return idf

    def score(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        scores = [0.0] * len(self.documents)

        if not query_tokens or not self.documents:
            return scores

        for idx, doc_freq in enumerate(self.doc_freqs):
            doc_length = self.doc_len[idx] or 1
            score = 0.0
            for token in query_tokens:
                freq = doc_freq.get(token)
                if not freq:
                    continue
                idf = self.idf.get(token, 0.0)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl if self.avgdl else 0.0))
                score += idf * (freq * (self.k1 + 1)) / denominator
            scores[idx] = score

        return scores

    def search(self, query: str, top_k: int = 4) -> list[ScoredDocument]:
        scores = self.score(query)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        results: list[ScoredDocument] = []

        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            if scores[idx] <= 0:
                continue
            results.append(ScoredDocument(document=self.documents[idx], score=scores[idx], rank=rank))

        return results

    def to_payload(self) -> dict:
        return {
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in self.documents
            ],
            "tokenized_documents": self.tokenized_documents,
            "k1": self.k1,
            "b": self.b,
        }

    def save(self, path: str | Path = DEFAULT_BM25_PATH) -> None:
        payload_path = Path(path)
        payload_path.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BM25_PATH) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        documents = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in payload["documents"]
        ]
        return cls(
            documents=documents,
            tokenized_documents=payload["tokenized_documents"],
            k1=payload.get("k1", 1.5),
            b=payload.get("b", 0.75),
        )


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[Document]],
    *,
    rrf_k: int = 60,
) -> list[Document]:
    scores: dict[str, float] = {}
    documents_by_id: dict[str, Document] = {}

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs, start=1):
            doc_id = str(doc.metadata.get("chunk_id") or doc.metadata.get("source") or doc.page_content[:80])
            documents_by_id.setdefault(doc_id, doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    ordered_ids = sorted(scores, key=scores.get, reverse=True)
    return [documents_by_id[doc_id] for doc_id in ordered_ids]


def build_bm25_from_documents(documents: Sequence[Document]) -> BM25Index:
    return BM25Index.from_documents(documents)

