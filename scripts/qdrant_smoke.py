import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url:
    raise ValueError("QDRANT_URL is missing from .env")

if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY is missing from .env")


# --------------------------------------------------
# Connect to Qdrant
# --------------------------------------------------

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

print("Successfully connected to Qdrant!")


# --------------------------------------------------
# Collection configuration
# --------------------------------------------------

collection_name = "rag_text"

existing_collections = [
    collection.name
    for collection in client.get_collections().collections
]

if collection_name not in existing_collections:

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    print(f"Collection '{collection_name}' created successfully!")

else:
    print(f"Collection '{collection_name}' already exists.")


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

model_name = "sentence-transformers/all-MiniLM-L6-v2"

print(f"\nLoading embedding model: {model_name}")

embedding_model = SentenceTransformer(model_name)


# --------------------------------------------------
# Create test text
# --------------------------------------------------

test_text = """
Kazi Tanjim Shakib visited India and Nepal.
He enjoys traveling and exploring new places.
"""


# --------------------------------------------------
# Convert text into a vector
# --------------------------------------------------

vector = embedding_model.encode(test_text).tolist()

print(f"Embedding dimension: {len(vector)}")


# --------------------------------------------------
# Create a Qdrant point
# --------------------------------------------------

point = PointStruct(
    id=str(uuid.uuid4()),
    vector=vector,
    payload={
        "text": test_text,
        "source": "qdrant_test",
        "file_name": "test_document.txt",
        "chunk_index": 0,
    },
)


# --------------------------------------------------
# Insert the point into Qdrant
# --------------------------------------------------

client.upsert(
    collection_name=collection_name,
    points=[point],
)

print("\nTest document inserted successfully!")


# --------------------------------------------------
# Verify collection status
# --------------------------------------------------

collection_info = client.get_collection(
    collection_name=collection_name
)

print(f"\nCollection: {collection_name}")
print(f"Points stored: {collection_info.points_count}")
