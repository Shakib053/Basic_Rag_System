"""
ingestion/image_pipeline.py

Image ingestion pipeline.

Walks every PDF in the data directory, extracts embedded images via
image_extractor, embeds each image with CLIP, and stores the vectors in a
dedicated Chroma database.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingestion.image_extractor import extract_images_from_pdf
from embeddings.clip_embeddings import CLIPEmbeddings


PDF_DIR = Path("data")
IMAGE_OUTPUT_DIR = Path("data/extracted_images")
IMAGE_PERSIST_DIR = Path("image_chroma_db")


def load_image_documents() -> list[Document]:
    """
    Walk every PDF inside *PDF_DIR*, extract all embedded images, and return
    one LangChain ``Document`` per image.

    Returns
    -------
    list[Document]
        Each document's ``page_content`` is the image file path; metadata
        carries ``source``, ``page``, ``image_index``, ``image_path``, and
        ``file_type``.
    """
    documents: list[Document] = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        extracted_images = extract_images_from_pdf(pdf_path, IMAGE_OUTPUT_DIR)

        for image in extracted_images:
            document = Document(
                page_content=str(image["image_path"]),
                metadata={
                    "source": image["source"],
                    "page": image["page"],
                    "image_index": image["image_index"],
                    "image_path": str(image["image_path"]),
                    "file_type": "image",
                },
            )
            documents.append(document)

    return documents


def build_image_vector_store(
    documents: Sequence[Document],
    persist_dir: Path,
) -> None:
    """Build a Chroma vector store containing CLIP image embeddings."""
    embedding_model = CLIPEmbeddings()

    Chroma.from_documents(
        documents=list(documents),
        embedding=embedding_model,
        persist_directory=str(persist_dir),
    )


def run_image_pipeline() -> bool:
    """
    Build / rebuild the image Chroma vector store.

    Returns
    -------
    bool
        ``True`` if the store was rebuilt, ``False`` if skipped because no
        images were found in any PDF.
    """
    image_documents = load_image_documents()

    if not image_documents:
        print("⚠  No images found in PDFs — skipping image pipeline.")
        return False

    build_image_vector_store(image_documents, IMAGE_PERSIST_DIR)
    print(f"Stored {len(image_documents)} images into {IMAGE_PERSIST_DIR}")
    return True
