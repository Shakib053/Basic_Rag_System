from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

RECURSIVE_STRATEGY = "recursive"
RECURSIVE_CHUNK_SIZE = 700
RECURSIVE_CHUNK_OVERLAP = 100

def build_recursive_chunker() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP
    )
