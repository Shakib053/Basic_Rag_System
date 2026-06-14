from dotenv import load_dotenv

# from langchain_openai import OpenAIEmbeddings // Need paid API key 

from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

with open("data/personal_info.txt", "r") as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Created {len(chunks)} chunks")

# embedding_model = OpenAIEmbeddings()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Data stored in ChromaDB")

# Get everything
data = vector_store.get()

print(data)