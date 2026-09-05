# Local Document RAG Assistant

A terminal-based Retrieval-Augmented Generation (RAG) assistant for querying local, text-extractable documents. It incrementally builds a dense+sparse hybrid index in Qdrant and answers from cited document evidence, with an explicit general-knowledge fallback when the corpus has no relevant evidence.

## Features

- Ingests `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.html`, `.csv`, and `.xlsx`
- Supports semantic chunking by default, with recursive chunking as an option
- Stores text embeddings in Qdrant using `sentence-transformers/all-MiniLM-L6-v2`
- Combines dense and BM25 sparse vectors in Qdrant using reciprocal-rank fusion
- Produces validated 1–3 query plans while always retaining the original request
- Searches the corpus first, then chooses a grounded answer or a clearly labeled general fallback
- Reranks retrieved text with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Emits inline file/page/slide/sheet citations and rejects invented citation IDs
- Treats document text as untrusted data rather than model instructions
- Supports incremental upload, replacement, listing, deletion, and document-scoped search
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
QDRANT_TEXT_COLLECTION=rag_text_v2
SPARSE_EMBEDDING_MODEL=Qdrant/bm25
# Optional override for the checked-in calibrated model threshold:
# RERANK_RELEVANCE_THRESHOLD=-5.0740085
# Set to 1 when all Hugging Face models are already cached:
# HF_HUB_OFFLINE=1

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

The v2 collection has a different dense+sparse schema. Add documents to `data/`, then perform a one-time rebuild when migrating from the old `rag_text` collection:

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

Inside chat, documents can be managed incrementally without rebuilding the collection:

```text
/upload "/absolute/path/to/report.pdf"
/documents
/use <document_id> [document_id ...]
/use all
/delete <document_id>
```

Run evaluation:

```bash
python -m evaluation.ragas_eval --mode full --dataset eval_dataset.json --output evaluation_results.json
```

Recalculate a relevance threshold after changing the reranker or materially changing the corpus. The input contains labeled reranker scores; score collection should use the full query-planning and retrieval path:

```bash
python -m evaluation.relevance_calibration \
  evaluation/relevance_samples.json \
  retrieval/relevance_thresholds.json
```

Run external-service smoke checks when the corresponding service is configured:

```bash
python scripts/ollama_smoke.py
python scripts/openrouter_smoke.py
python scripts/qdrant_smoke.py
```

## Notes

- `/upload` replaces chunks when the same canonical local path changes and is a no-op when its content hash is unchanged.
- The default upload limit is 50 MiB and can be changed with `MAX_UPLOAD_BYTES`.
- Scanned/image-only files, OCR, audio/video, archives, and chart understanding are not supported.
- The existing image extraction pipeline remains optional, but image paths are never treated as textual answer evidence.
- The included relevance calibration is an initial 20-query local-corpus baseline; expand it with held-out genre-specific examples before treating its quality metrics as an SLA.
- Generated image data is stored in `data/extracted_images/` and `image_chroma_db/`.
