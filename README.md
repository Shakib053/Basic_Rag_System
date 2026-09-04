# Local Document RAG Assistant

A terminal-based Retrieval-Augmented Generation (RAG) assistant for querying local `.txt`, `.docx`, and text-extractable PDF files. It builds a text index in Qdrant, optionally extracts PDF images into a CLIP-powered Chroma index, and answers questions using either Ollama or OpenRouter-compatible chat models.

## Features

- Ingests local `.txt`, `.docx`, and `.pdf` files from `data/`
- Supports semantic chunking by default, with recursive chunking as an option
- Stores text embeddings in Qdrant using `sentence-transformers/all-MiniLM-L6-v2`
- Combines Qdrant MMR semantic search with BM25 keyword retrieval
- Rewrites follow-up questions conservatively and expands queries for stronger recall
- Reranks retrieved text with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Extracts embedded PDF images and retrieves image references with CLIP + Chroma
- Avoids guessing ambiguous pronouns such as "he" or "she"; when needed, answers by source document
- Includes RAGAS evaluation support with `eval_dataset.json`

## Project Structure

```text
.
├── chat.py                  # terminal chat entry point (kept at root)
├── chunking/
│   └── recursive_chunking.py
├── embeddings/
│   ├── clip_embeddings.py
│   └── text_embeddings.py
├── evaluation/
│   └── ragas_eval.py
├── ingestion/
│   ├── image_extractor.py
│   ├── image_pipeline.py
│   ├── run.py               # text/image ingestion entry point
│   ├── text_pipeline.py
│   └── __init__.py
├── prompts/
│   ├── answer.py
│   └── query.py
├── retrieval/
│   ├── context_formatting.py
│   ├── hybrid_retrieval.py
│   ├── image_retrieval.py
│   └── query_enhancement.py
├── scripts/                 # manual external-service smoke checks
│   ├── ollama_smoke.py
│   ├── openrouter_smoke.py
│   └── qdrant_smoke.py
├── vectorstore/
│   ├── chroma_store.py
│   └── qdrant_store.py
└── data/                    # source documents
```

## Setup

Requires Python 3.12, Qdrant, and either Ollama or an OpenRouter API key.

Create `.env`:

```text
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_TEXT_COLLECTION=rag_text

LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_BASE_URL=http://localhost:11434

# For OpenRouter instead:
# LLM_PROVIDER=openrouter
# OPENROUTER_API_KEY=your_openrouter_api_key
```

Install dependencies:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

Add documents to `data/`, then rebuild indexes:

```bash
python -m ingestion.run
```

Useful ingestion options:

```bash
python -m ingestion.run --text-only
python -m ingestion.run --images-only
python -m ingestion.run --strategy recursive
```

Start chat:

```bash
python chat.py
```

Run evaluation:

```bash
python -m evaluation.ragas_eval --mode full --dataset eval_dataset.json --output evaluation_results.json
```

Run external-service smoke checks when the corresponding service is configured:

```bash
python scripts/ollama_smoke.py
python scripts/openrouter_smoke.py
python scripts/qdrant_smoke.py
```

## Notes

- Re-run ingestion after adding or changing files in `data/`.
- Ambiguous pronouns are not mapped to a person unless chat history or retrieved context clearly identifies that subject.
- Image retrieval returns references to extracted images; final answers are grounded primarily in retrieved text context.
- Generated image data is stored in `data/extracted_images/` and `image_chroma_db/`.
