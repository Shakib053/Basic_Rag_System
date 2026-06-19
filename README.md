# Local RAG Learning Project

A small local Retrieval-Augmented Generation (RAG) project for learning LangChain, ChromaDB, Hugging Face embeddings, and Ollama.

## Current State

- Ingests `data/personal_info.txt`
- Stores chunks in ChromaDB
- Answers questions in the terminal with Ollama
- Keeps simple chat history for follow-ups

## Future Plan

- Hybrid retrieval (Semantic Search + Keyword Search)
- Re-ranking (Use encoder / LLM)
- Offline evaluation set
- RAGAS evaluation
- Pipeline and build validation

## Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face sentence-transformers
- Ollama

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate
pip install langchain langchain-chroma langchain-huggingface langchain-ollama langchain-text-splitters python-dotenv chromadb
```

On Windows:

```bash
venv\Scripts\activate
```

## Usage

1. Build the vector store:

```bash
python ingestion.py
```

2. Start the chat loop:

```bash
python chat.py
```

Type `exit` to quit.

## Notes

- This project is for learning purposes only.
- Run `python ingestion.py` again after changing `data/personal_info.txt`.
