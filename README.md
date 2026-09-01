# Local Personal RAG Assistant

A terminal-based Retrieval-Augmented Generation (RAG) assistant for querying local `.txt` files and text-extractable PDFs. It builds a text index in Qdrant, optionally extracts PDF images into a CLIP-powered Chroma index, and answers questions using either Ollama or OpenRouter-compatible chat models.

## Features

- Ingests local `.txt` and `.pdf` files from `data/`
- Supports semantic chunking by default, with recursive chunking as an option
- Stores text embeddings in Qdrant using `sentence-transformers/all-MiniLM-L6-v2`
- Combines Qdrant MMR semantic search with BM25 keyword retrieval
- Rewrites follow-up questions and expands queries for stronger recall
- Reranks retrieved text with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Extracts embedded PDF images and retrieves image references with CLIP + Chroma
- Includes RAGAS evaluation support with `eval_dataset.json`

## Project Structure

```text
.
├── chat.py                  # terminal chat entry point
├── ingestion/
│   ├── run.py               # text/image ingestion entry point
│   ├── text_pipeline.py
│   ├── image_pipeline.py
│   └── text_vectorstore.py
├── hybrid_retrieval.py      # semantic + keyword retrieval and reranking
├── query_enhancement.py     # query rewrite and multi-query retrieval
├── context_formatting.py
├── image_retrieval.py
├── ragas_eval.py
├── tests/
└── data/                    # source documents
```

## Setup

Requires Python 3.10+, Qdrant, and either Ollama or an OpenRouter API key.

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
python -m venv venv
source venv/bin/activate
pip install langchain langchain-classic langchain-community langchain-chroma langchain-qdrant langchain-experimental langchain-huggingface langchain-openai langchain-ollama langchain-text-splitters python-dotenv chromadb qdrant-client sentence-transformers pypdf pymupdf pillow ragas langchain-google-vertexai
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
python ragas_eval.py --mode full --dataset eval_dataset.json --output evaluation_results.json
```

## Notes

- Re-run ingestion after adding or changing files in `data/`.
- Image retrieval returns references to extracted images; final answers are grounded primarily in retrieved text context.
- Generated image data is stored in `data/extracted_images/` and `image_chroma_db/`.
