# Local RAG Pipeline with LangChain

This repository showcases a lightweight, local Retrieval-Augmented Generation (RAG) workflow built with [LangChain](https://www.langchain.com/). It combines local document ingestion, vector search, and chat-based generation without requiring a server API or cloud-hosted inference.

The application runs entirely in the terminal, making it easy to ingest data and ask questions from a simple command-line interface. It is designed as a local RAG demo for anyone who wants to explore the LangChain workflow without building a web app or backend service.

The stack is intentionally simple and fully local:

- **ChromaDB** for vector storage
- **Hugging Face embeddings** for text representation
- **Ollama** with a locally downloaded model for response generation

## Sample Terminal Output

Here is a short excerpt from the terminal-based chat experience:

```bash
$ python chat.py
Local RAG Chat (type 'exit' to quit)

You: what is my name
RAG retrieval query: what is my name

AI: Your name is Kazi Tanjim Shakib.

You: what do I do and where
RAG retrieval query: What is my role and where am I located?

AI: You are a Senior Software Engineer at Samsung R&D Institute Bangladesh. Your role involves iOS application development, specifically using Swift and UIKit, and you are proficient in Swift, C, and C++.

You: When did i join there
RAG retrieval query: When did I join there?

AI: You joined Samsung R&D Institute Bangladesh in 2021.

You: what kind of expertise do I have 
RAG retrieval query: What are the key areas of expertise that I possess?

AI: You have expertise in the following areas:  
- **Programming Languages**: C++, Python, Swift, C  
- **Frameworks/Tools**: React.js, SwiftUI, UIKit  
- **Development Skills**: iOS application development (Swift, UIKit), Agile methodologies, large-scale production apps  
- **Specialization**: iOS developer with focus on Swift and UIKit  

Let me know if you'd like further details!
```

> Output may vary slightly depending on the Ollama model you use and the content in `data/personal_info.txt`.

## How It Works

The workflow is split into two scripts:

1. `ingestion.py`
   - Reads the source text from `data/personal_info.txt`
   - Splits the document into smaller chunks
   - Generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
   - Stores the embedded chunks in a local ChromaDB collection at `./chroma_db`

2. `chat.py`
   - Loads the saved ChromaDB vector store
   - Uses the same Hugging Face embedding model for retrieval
   - Calls an Ollama model to generate responses
   - You can use any model available on the Ollama website by downloading it with `ollama pull <model-name>` and updating the model name in `chat.py`
   - Keeps a lightweight chat history so follow-up questions remain context-aware

## Tech Stack

- Python
- LangChain
- ChromaDB
- Hugging Face sentence-transformers
- Ollama
- Any Ollama-compatible model you choose

## Project Structure

```text
.
├── chat.py
├── ingestion.py
├── data/
│   └── personal_info.txt
└── chroma_db/              # Created after ingestion
```

## Prerequisites

Make sure the following tools are installed on your machine:

- Python 3.10 or newer
- Git
- Ollama

You also need the Ollama model used by the chat script:

```bash
ollama run <model-name>
```

You can browse available models on the Ollama website and use any one that fits your needs. Just make sure the model name in `chat.py` matches the one you pulled locally.

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
pip install langchain langchain-chroma langchain-huggingface langchain-ollama langchain-text-splitters python-dotenv chromadb
```

## Usage

### 1) Build the vector store

Run the ingestion script to chunk the source text and create the ChromaDB index:

```bash
python ingestion.py
```

This will read from `data/personal_info.txt` and create the local vector store in `./chroma_db`.

### 2) Start the chat loop

After ingestion is complete, launch the local RAG chat session:

```bash
python chat.py
```

Type your questions in the terminal. To exit, enter:

```text
exit
```

## Notes

- The assistant only answers from the retrieved context. If the answer is not present in the source text, it will respond with a limited answer such as “I don’t know.”
- The project is designed for local experimentation and learning, not production deployment.
- If you change the source text in `data/personal_info.txt`, rerun `python ingestion.py` so the ChromaDB store stays in sync.
