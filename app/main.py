import logging
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints

from app.services.rag_service import ask_question


logging.basicConfig(format="%(levelname)s:     %(message)s")
logging.getLogger("app.services.rag_service").setLevel(logging.INFO)

app = FastAPI(
    title="Basic RAG API",
    version="0.1.0"
)


class ChatRequest(BaseModel):
    # Strip surrounding spaces, then require at least 1 character (else FastAPI returns 422).
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Source(BaseModel):
    document: str
    page: int | None = None  # 1-based; only PDFs have pages
    score: float | None = None  # raw cross-encoder rerank score, higher is more relevant


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = ask_question(
        request.query
    )

    return result