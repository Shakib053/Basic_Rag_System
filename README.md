# Local Personal RAG Assistant

A small terminal-based RAG assistant for asking questions over local files — personal notes, profiles, books, or any text-based PDF. It indexes text and text-based PDFs, retrieves relevant context with hybrid search, optionally retrieves PDF image references, and answers through a Hugging Face Router-compatible chat model.

## Features

- Ingests `.txt` and text-extractable `.pdf` files from `data/`
- Stores chunks as pure document content; source file and page live in chunk metadata
- Uses semantic chunking by default, with recursive chunking available through `CHUNKING_STRATEGY=recursive`
- Stores text embeddings in ChromaDB at `chroma_db/`
- Combines Chroma MMR semantic retrieval with BM25 keyword retrieval
- Rewrites follow-up questions and expands queries with multi-query retrieval
- Reranks retrieved text with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Optionally extracts PDF images and stores CLIP image embeddings in `image_chroma_db/`
- Builds final context from text chunks plus image references

## Architecture

```text
data/
  -> ingestion/run.py (text pipeline)
  -> semantic or recursive chunking
  -> text embeddings
  -> chroma_db/

PDF images
  -> ingestion/run.py (image pipeline)
  -> data/extracted_images/
  -> CLIP embeddings
  -> image_chroma_db/

question
  -> chat.py
  -> query rewrite + multi-query expansion
  -> hybrid retrieval: Chroma MMR + BM25
  -> cross-encoder reranking
  -> combined text + image-reference context
  -> grounded answer
```

Image support retrieves image file references from PDFs. It does not perform full image, audio, or video understanding.

## Project Structure

```text
.
├── chat.py
├── ingestion/
│   ├── __init__.py
│   ├── run.py              # single entry point
│   ├── text_pipeline.py
│   ├── image_pipeline.py
│   └── store.py
├── hybrid_retrieval.py
├── image_retrieval.py
├── image_extractor.py
├── query_enhancement.py
├── context_formatting.py
├── recursive_chunking.py
├── embeddings/
├── tests/
├── data/
├── chroma_db/              # generated text vector store
└── image_chroma_db/        # generated image vector store
```

## Setup

Requirements:

- Python 3.10+
- Hugging Face access token

Create a local `.env` file:

```text
HF_TOKEN=your_hugging_face_token_here
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install langchain langchain-classic langchain-community langchain-chroma langchain-experimental langchain-huggingface langchain-openai langchain-text-splitters python-dotenv chromadb sentence-transformers pypdf pymupdf
```

There is no `requirements.txt` yet, so dependencies are installed directly for now.

## Usage

Build the indexes (text + images):

```bash
python -m ingestion.run
```

Build only the text index:

```bash
python -m ingestion.run --text-only
```

Optionally build only the PDF image index:

```bash
python -m ingestion.run --images-only
```

Start the terminal chat:

```bash
python chat.py
```

Type `exit` to quit.

## Retrieval Notes

- Chroma semantic retrieval uses MMR with `k=12`, `fetch_k=20`, and `lambda_mult=0.5`
- BM25 keyword retrieval returns `k=12` candidates
- The ensemble retriever uses equal semantic and keyword weights
- The top 5 reranked text chunks are sent to the chat model
- Chunk embeddings are computed over pure content; `[Source: … | page N]` citation headers are rendered from metadata only when the answer context is built, after reranking
- Query rewriting resolves follow-up references and expands synonyms; it does not inject any fixed subject name, so retrieval ranks on actual content relevance
- Image retrieval returns up to 3 image references when `image_chroma_db/` exists

After changing ingestion behavior or adding files to `data/`, re-run `python -m ingestion.run` to rebuild the indexes.