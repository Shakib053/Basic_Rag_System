from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_retrieval import build_bm25_from_documents, load_txt_documents, split_documents_with_ids

load_dotenv()

data_dir = Path("data")
bm25_path = Path("./bm25_corpus.json")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

source_documents = load_txt_documents(data_dir)
if not source_documents:
    raise RuntimeError(f"No .txt files found in {data_dir.resolve()}")

chunks = split_documents_with_ids(source_documents, splitter)

print(f"Loaded {len(source_documents)} source documents")
print(f"Created {len(chunks)} chunks")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

bm25_index = build_bm25_from_documents(chunks)
bm25_index.save(bm25_path)

print("Data stored in ChromaDB")
print(f"BM25 corpus stored in {bm25_path}")

data = vector_store.get()
print(data)
