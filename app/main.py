from fastapi import FastAPI

app = FastAPI(
    title="Basic RAG API",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }