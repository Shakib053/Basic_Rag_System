# Local Document RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant for querying local, text-extractable documents. It incrementally builds a dense+sparse hybrid index in Qdrant and answers from cited document evidence, with an explicit general-knowledge fallback when the corpus has no relevant evidence.

It can be used three ways: a **web UI** (React), a **REST API** (FastAPI), or a **terminal chat**.

## Features

### Retrieval and answering

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
- OCRs image-only PDF pages (scans, phone photos) with Tesseract
- Includes RAGAS evaluation support with `eval_dataset.json`

### Web UI and API
- Upload documents from the browser and see them indexed immediately
- "Your documents" list showing every indexed document with its type, chunk count, and upload date
- Ask questions and get an answer with its sources (document, page, and relevance score)
- Each upload gets its own ID, so files with the same name are kept as separate documents
- Duplicate content is detected by file hash: uploading identical bytes (under any name) returns the existing document instead of indexing it again

## Web UI

The React app in `frontend/` talks to the FastAPI backend at `http://localhost:8000`. The page has three parts:

1. **Upload**: choose a file (`.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.html`, `.csv`, `.xlsx`) and click **Upload**. It shows `report.pdf indexed (12 chunks)` on success, `Already uploaded as report.pdf, nothing new was added.` for identical content, or the reason a file was rejected (wrong type, empty, too large).
2. **Your documents**: every document in the index, loaded when the page opens and refreshed after each upload. Same-name documents can be told apart by their upload date.
3. **Ask**: type a question to get the answer and its sources.

## API

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `POST` | `/chat` | Body `{"query": "..."}`. Returns `answer`, `mode` (`grounded`, `general`, or `error`), `citations`, and `sources` |
| `POST` | `/upload` | Multipart form with a `file` field. Saves and indexes the file; returns `document_id`, `file_name`, `status` (`indexed` or `duplicate`), `chunk_count`, and `warnings`. Invalid files return `400` with the reason |
| `GET` | `/documents` | Lists every indexed document: `document_id`, `file_name`, `file_type`, `chunk_count`, `ingested_at` |

Interactive API docs are available at http://localhost:8000/docs while the server runs.

## Project Structure

```text
.
├── app/
│   ├── main.py              # FastAPI app: /health, /chat, /upload, /documents
│   └── services/
│       ├── guardrails.py
│       ├── rag_service.py   # RAG answering pipeline
│       └── upload_service.py
├── frontend/                # React + Vite web UI
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
├── tests/
├── vectorstore/
│   ├── chroma_store.py
│   └── qdrant_store.py
└── data/                    # source documents; web uploads go to data/uploads/<id>/
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

Install the web UI dependencies (requires Node.js):

```bash
npm --prefix frontend install
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

Start the web app. Run the API in one terminal:

```bash
uvicorn app.main:app --reload
```

Then run the UI in a second terminal and open http://localhost:5173:

```bash
npm --prefix frontend run dev
```

Upload a document and list indexed documents over HTTP without the UI:

```bash
curl -F "file=@/absolute/path/to/report.pdf" http://localhost:8000/upload
curl http://localhost:8000/documents
```

Uploaded files are saved to `data/uploads/<id>/`. Files with the same name are kept as separate documents, and uploading identical content returns the existing document as `duplicate`.

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

- The terminal `/upload` command identifies a document by its local path: it replaces chunks when that file changes and is a no-op when its content hash is unchanged. The web/API upload instead gives every upload its own ID, so uploading an edited file with the same name adds a new document and keeps the old one.
- The default upload limit is 50 MiB and can be changed with `MAX_UPLOAD_BYTES`.
- Image-only PDF pages (scans, phone photos) are OCR'd with Tesseract (`brew install tesseract`; set `OCR_LANGUAGE`, default `eng`). Without Tesseract those pages are skipped with a warning. Handwriting accuracy is limited.
- Standalone image uploads, audio/video, archives, and chart understanding are not supported.
- The existing image extraction pipeline remains optional, but image paths are never treated as textual answer evidence.
- The included relevance calibration is an initial 20-query local-corpus baseline; expand it with held-out genre-specific examples before treating its quality metrics as an SLA.
- Generated image data is stored in `data/extracted_images/` and `image_chroma_db/`.

## Roadmap

Planned next steps:

- **Delete documents from the UI**: a delete button per document and a `DELETE /documents/{id}` endpoint (the terminal chat can already delete). This matters most for removing old versions of same-name files.
- **Replace a document**: an option to replace an existing document when uploading a new version, instead of always keeping both.
- **Choose which documents to search**: pick documents in the UI to limit a question to them, like the terminal `/use` command.
- **Follow-up questions**: send chat history through the API so the UI can hold a conversation, not just answer one question at a time.
- **Richer answers in the UI**: show the answer mode (grounded or general) and the inline `[S1]` citations next to the answer.
- **Guardrails**: `check_input` and `check_output` in `app/services/guardrails.py` currently pass text through unchanged; real input and output checks will be added there.
- **Faster, friendlier uploads**: upload progress, background indexing for large files, and a faster document list for large collections.
- **Cleaner citations**: show a file name instead of the full server path for text-file citations.
