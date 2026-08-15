from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from image_extractor import extract_images_from_pdf
from embeddings.clip_embeddings import CLIPEmbeddings


PDF_DIR = Path("data")
IMAGE_OUTPUT_DIR = Path("data/extracted_images")
IMAGE_PERSIST_DIR = Path("image_chroma_db")

def load_image_documents() -> list[Document]:
    """
    Go through every PDF inside the data folder.

    Extract every image.

    Convert every extracted image into a LangChain Document.

    Returns
    -------
    list[Document]
    """

    documents: list[Document] = []

    # Find every PDF inside the data directory.
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:

        # Extract all images from this PDF.
        extracted_images = extract_images_from_pdf(
            pdf_path,
            IMAGE_OUTPUT_DIR,
        )

        # Create one Document per image.
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
    """
    Build a Chroma vector store containing image embeddings.
    """

    embedding_model = CLIPEmbeddings()

    Chroma.from_documents(
        documents=list(documents),
        embedding=embedding_model,
        persist_directory=str(persist_dir),
    )



def main():

    # Step 1
    image_documents = load_image_documents()

    # Step 2
    if not image_documents:
        raise RuntimeError("No images were found.")

    # Step 3
    build_image_vector_store(
        image_documents,
        IMAGE_PERSIST_DIR,
    )

    print(
        f"Stored {len(image_documents)} images into "
        f"{IMAGE_PERSIST_DIR}"
    )


if __name__ == "__main__":
    main()