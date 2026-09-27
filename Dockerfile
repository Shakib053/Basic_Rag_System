# One container for the whole app: FastAPI serves the API and the built React UI.
# Made for a free Hugging Face Space (Docker SDK), which expects port 7860 and user id 1000.

# Stage 1: build the React UI into frontend/dist.
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: the Python backend.
FROM python:3.12-slim

# Tesseract OCRs image-only PDF pages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed
WORKDIR /home/user/app

# CPU-only torch first: the default wheel bundles CUDA and is several GB larger.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download the embedding, reranker and BM25 models now, so the first question isn't slow.
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
from fastembed import SparseTextEmbedding; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
SparseTextEmbedding('Qdrant/bm25')"

COPY --chown=user . .
COPY --from=frontend --chown=user /frontend/dist frontend/dist

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
