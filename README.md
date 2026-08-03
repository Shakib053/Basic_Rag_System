# Local Personal RAG Assistant

A professional local Retrieval-Augmented Generation (RAG) pipeline for querying personal documents from the terminal. The project ingests local text and text-based PDF files, chunks them intelligently, stores embeddings in ChromaDB, retrieves relevant context with hybrid search, reranks the candidates, and generates grounded answers through a Hugging Face Router-compatible chat model.

This is designed as a compact but complete personal knowledge assistant: simple enough to run locally, structured enough to demonstrate the core components of a modern RAG system.

## Features

- **Document ingestion** for `.txt` files and text-extractable `.pdf` files in `data/`
- **PDF parsing with metadata** using LangChain's PyMuPDF loader, including page-level metadata and markdown table extraction
- **Semantic chunking by default** with LangChain `SemanticChunker`
- **Recursive chunking fallback** through `CHUNKING_STRATEGY=recursive`
- **Local vector storage** with ChromaDB and stable chunk IDs
- **Hybrid retrieval** combining Chroma semantic search with BM25 keyword search
- **MMR semantic retrieval** to improve result diversity
- **Cross-encoder reranking** with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Context-grounded chat** that answers only from retrieved documents
- **Follow-up question handling** through lightweight chat-history rewriting
- **Terminal-first workflow** with retrieved context printed for transparency

## Architecture

```text
data/
  -> ingestion.py
  -> semantic or recursive chunking
  -> Hugging Face embeddings
  -> ChromaDB vector store
  -> hybrid retrieval: Chroma MMR + BM25
  -> cross-encoder reranker
  -> Hugging Face Router chat model
  -> grounded answer
```

The pipeline is split into two main stages:

1. **Ingestion**
   - Loads supported documents from `data/`
   - Extracts text from `.txt` and text-based `.pdf` files
   - Preserves source metadata such as file name, file type, page, chunk index, and chunk ID
   - Chunks documents using semantic boundaries by default
   - Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`
   - Rebuilds the local ChromaDB store at `./chroma_db`

2. **Retrieval and chat**
   - Loads the persisted ChromaDB store
   - Builds a BM25 retriever from the stored Chroma documents
   - Combines semantic and keyword retrieval with LangChain's ensemble retriever
   - Applies cross-encoder reranking to select the strongest final context
   - Sends only the selected context to the chat model
   - Rewrites follow-up questions into standalone retrieval queries when chat history exists

## Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face embeddings
- Hugging Face Router-compatible chat models
- Sentence Transformers cross-encoder reranking
- PyMuPDF-based PDF loading
- BM25 keyword retrieval

## Project Structure

```text
.
├── README.md
├── chat.py
├── hybrid_retrieval.py
├── ingestion.py
├── recursive_chunking.py
├── test_hybrid_retrieval.py
├── test_ingestion.py
├── test_recursive_chunking.py
├── bm25_corpus.json
├── chroma_db/              # Created or rebuilt by ingestion
└── data/
    ├── Kazi_Tanjim_Shakib_Professional_Profile.pdf
    ├── Personal_info.txt
    ├── Projects.txt
    ├── Travel_history.txt
    └── multimodal_rag_sample_data.pdf
```

## Prerequisites

- Python 3.10 or newer
- Git
- A Hugging Face access token

Create a local `.env` file in the project root:

```text
HF_TOKEN=your_hugging_face_token_here
```

The current chat model is configured in `chat.py`:

```python
model="Qwen/Qwen2.5-72B-Instruct"
```

## Setup

Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install langchain langchain-classic langchain-community langchain-chroma langchain-experimental langchain-huggingface langchain-openai langchain-text-splitters python-dotenv chromadb sentence-transformers pypdf pymupdf
```

## Usage

### 1. Build the vector store

Run ingestion after adding or updating files in `data/`:

```bash
python ingestion.py
```

Ingestion reads every supported file directly inside `data/`, creates chunks, embeds them, and rebuilds the ChromaDB vector store in `./chroma_db`.

Each ingestion run uses a staging directory and replaces the existing store only after the new store is built successfully. If indexing fails, the previous vector store remains available.

### 2. Choose a chunking strategy

Semantic chunking is the default:

```bash
python ingestion.py
```

The semantic chunker uses neighboring sentence embeddings to identify topic boundaries. It is configured with a 95th percentile breakpoint threshold and a 200-character minimum chunk size.

To use recursive character chunking instead:

```bash
CHUNKING_STRATEGY=recursive python ingestion.py
```

Recursive mode uses 700-character chunks with 100-character overlap. Rebuild the index whenever you change the chunking strategy or modify files in `data/`.

### 3. Start the chat loop

After ingestion finishes, start the terminal chat:

```bash
python chat.py
```

Ask questions in the terminal. To exit:

```text
exit
```

For each question, the app prints:

1. The standalone retrieval query
2. The retrieved and reranked context chunks
3. The final grounded answer

## Retrieval Strategy

The retriever uses a two-stage ranking process:

1. **Hybrid candidate retrieval**
   - Chroma semantic retrieval uses MMR with `k=12`, `fetch_k=10`, and `lambda_mult=0.5`
   - BM25 keyword retrieval returns `k=12` keyword-focused candidates
   - LangChain's ensemble retriever combines both retrievers with equal weights

2. **Cross-encoder reranking**
   - Candidate documents are scored against the query using `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - The top 5 reranked chunks are passed to the chat model as final context

This combination helps the assistant retrieve both semantically related passages and exact keyword matches, then refine the final context with a stronger relevance model.

## Document Support

Supported source files:

- `.txt`
- Text-based `.pdf`

PDF files must contain selectable text. Scanned or image-only PDFs are not OCR processed. Empty, encrypted, corrupt, or unreadable PDFs are skipped with a warning so valid files can still be indexed.

Although this project supports a multimodal-style document workflow across plain text and PDF sources, it does not perform image, audio, or video understanding.

## Testing

Run the unit tests with:

```bash
python -m unittest
```

The current tests cover:

- document loading for text and PDFs
- semantic and recursive chunking configuration
- stable chunk metadata and IDs
- hybrid retrieval setup
- failure-safe ChromaDB rebuild behavior

## Troubleshooting

### Chroma native binding error

If ingestion reports that `chromadb_rust_bindings.chromadb_rust_bindings` is missing, reinstall Chroma from its binary wheel inside the active virtual environment:

```bash
python -m pip install --force-reinstall --no-cache-dir --no-deps --only-binary=:all: chromadb==1.5.9
python -c "import chromadb_rust_bindings.chromadb_rust_bindings; print('Chroma native bindings loaded')"
```

## Notes

- The assistant is instructed to answer only from retrieved context.
- If the answer is not available in the indexed documents, it should respond with a limited answer such as "I don't know."
- Retrieved PDF chunks include source file and page metadata in terminal output.
- This project is intended for local experimentation, prototyping, and portfolio demonstration rather than production deployment.

## Future Work

- Add an offline evaluation dataset
- Add RAGAS or similar RAG quality evaluation
- Add automated pipeline validation
- Package dependencies in a dedicated requirements file
