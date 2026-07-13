# Local RAG Pipeline with LangChain

This repository showcases a lightweight Retrieval-Augmented Generation (RAG) workflow built with [LangChain](https://www.langchain.com/). It combines local document ingestion, local retrieval, reranking, and chat-based generation through the Hugging Face router.

The application runs entirely in the terminal, making it easy to ingest data and ask questions from a simple command-line interface. It is designed as a local RAG demo for anyone who wants to explore the LangChain workflow without building a web app or backend service.

The stack is intentionally simple:

- **ChromaDB** for vector storage
- **BM25Retriever** for keyword retrieval
- **Hugging Face embeddings** for text representation
- **LangChain SemanticChunker** for meaning-aware document boundaries
- **CrossEncoder reranking** for improving the final retrieved documents
- **LangChain ensemble retrieval** for combining semantic and keyword results
- **Hugging Face router** for response generation

## How It Works

The workflow is split into two scripts:

1. `ingestion.py`
   - Reads every `.txt` and text-based `.pdf` file in `data/`
   - Splits documents at semantic topic boundaries by default
   - Generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
   - Rebuilds the local ChromaDB collection at `./chroma_db`

2. `chat.py`
   - Loads the saved ChromaDB vector store
   - Uses the same Hugging Face embedding model for retrieval
   - Builds a BM25Retriever directly from the stored Chroma documents
   - Runs hybrid retrieval with LangChain's weighted ensemble retriever
   - Combines semantic search from Chroma with keyword search from BM25Retriever
   - Reranks the retrieved candidate chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Prints the final retrieved chunks before generating the answer
   - Calls a Hugging Face router chat model to generate responses
   - Keeps a lightweight chat history so follow-up questions remain context-aware

## Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face sentence-transformers
- Hugging Face router-compatible chat model

## Project Structure

```text
.
├── chat.py
├── hybrid_retrieval.py
├── ingestion.py
├── data/
│   ├── Personal_info.txt
│   ├── Projects.txt
│   ├── Travel_history.txt
│   └── Example.pdf
└── chroma_db/              # Created after ingestion
```

## Prerequisites

Make sure the following tools are installed on your machine:

- Python 3.10 or newer
- Git

You also need a Hugging Face token in a local `.env` file:

```text
HF_TOKEN=your_hugging_face_token_here
```

The current chat model is configured in `chat.py`:

```python
model="Qwen/Qwen2.5-72B-Instruct"
```

## Setup

Clone the repository from GitHub:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows, activate it with:

```bash
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install langchain langchain-classic langchain-community langchain-chroma langchain-experimental langchain-huggingface langchain-openai langchain-text-splitters python-dotenv chromadb sentence-transformers pypdf
```

## Usage

### 1) Build the vector store

Run the ingestion script to chunk the source text and create the ChromaDB index:

```bash
python ingestion.py
```

This will read every `.txt` file and text-based `.pdf` file directly inside `data/`, then create the local vector store in `./chroma_db`.
Each run rebuilds the vector store from scratch so old chunks do not stay mixed with new chunks.

Semantic chunking is enabled by default. It embeds neighboring sentences and
creates a boundary when their cosine-distance change reaches the configured
95th percentile, while avoiding chunks smaller than 200 characters. The same
embedding model is then reused to index the resulting chunks in Chroma.

To use the previous character-based splitter instead, set:

```bash
CHUNKING_STRATEGY=recursive python ingestion.py
```

Supported values are `semantic` (default) and `recursive`. Recursive mode uses
700-character chunks with 100-character overlap. Changing the chunking strategy
or semantic threshold requires rerunning `python ingestion.py` to rebuild the
stored index.

PDFs must contain selectable text. Scanned or image-only PDFs are not OCR processed and will be skipped with a warning when they contain no extractable text. Unreadable, corrupt, or encrypted PDFs are also skipped without preventing other valid documents from being indexed.

Ingestion builds the new index in a staging directory and replaces the existing
`chroma_db` only after the build succeeds. If indexing fails, the previous vector
store remains available.

### 2) Start the chat loop

After ingestion is complete, launch the local RAG chat session:

```bash
python chat.py
```

Type your questions in the terminal. To exit, enter:

```text
exit
```

For every question, the app prints:

1. The actual retrieval query
2. The retrieved documents used as context
3. The final AI answer

## Notes

- The assistant only answers from the retrieved context. If the answer is not present in the source text, it will respond with a limited answer such as “I don’t know.”
- The project is designed for local experimentation and learning, not production deployment.
- The retriever first uses LangChain's ensemble weighting, then applies a small cross-encoder reranker.
- Semantic chunking controls ingestion boundaries. It is separate from semantic
  vector retrieval, which compares the question embedding with already-created
  chunk embeddings in Chroma.
- If you change any file in `data/`, rerun `python ingestion.py` so the ChromaDB store stays in sync.
- Retrieved PDF chunks include their source filename and page number in terminal output.

### Chroma native binding error

If ingestion reports that `chromadb_rust_bindings.chromadb_rust_bindings` is
missing, reinstall Chroma from its binary wheel inside the active virtual
environment:

```bash
python -m pip install --force-reinstall --no-cache-dir --no-deps --only-binary=:all: chromadb==1.5.9
python -c "import chromadb_rust_bindings.chromadb_rust_bindings; print('Chroma native bindings loaded')"
```

## Future Plan

- Offline evaluation set
- RAGAS evaluation
- Pipeline and build validation
