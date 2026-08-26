import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient


load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url:
    raise ValueError("QDRANT_URL is missing from .env")

if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY is missing from .env")


client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

collections = client.get_collections()

print("Successfully connected to Qdrant!")
print(f"Collections: {collections.collections}")