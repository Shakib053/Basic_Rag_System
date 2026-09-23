import logging

from fastapi import FastAPI
from pydantic import BaseModel

from app.services.rag_service import ask_question


logging.basicConfig(format="%(levelname)s:     %(message)s")
logging.getLogger("app.services.rag_service").setLevel(logging.INFO)

app = FastAPI(
    title="Basic RAG API",
    version="0.1.0"
)


class ChatRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = ask_question(
        request.query
    )

    return result