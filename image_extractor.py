"""
image_extractor.py

Purpose
-------
Extract every image from a PDF and save it to disk.

This file DOES NOT:
    - create embeddings
    - call CLIP
    - store vectors
    - perform OCR

It only extracts images.

The returned metadata will later be used by image_ingestion.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import fitz  # PyMuPDF


def extract_images_from_pdf(
    pdf_path: str | Path,
    output_root: str | Path,
) -> list[dict]:
    """
    Extract every image from a PDF.

    Parameters
    ----------
    pdf_path:
        Path to the PDF.

    output_root:
        Folder where extracted images will be stored.

    Returns
    -------
    list[dict]

    Example

    [
        {
            "image_path": "data/extracted_images/resume/page_2_img_1.png",
            "page": 2,
            "image_index": 1,
            "source": "data/pdf/resume.pdf"
        }
    ]
    """

    pdf_path = Path(pdf_path)

    # Create one folder per PDF.
    #
    # Example:
    #
    # resume.pdf
    #
    # becomes
    #
    # extracted_images/
    #     resume/
    #
    output_dir = Path(output_root) / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_images = []

    try:
        document = fitz.open(pdf_path)

    except Exception as exc:
        warnings.warn(
            f"Could not open PDF '{pdf_path}': {exc}",
            stacklevel=2,
        )
        return extracted_images

    # Go through every page.
    for page_number in range(len(document)):

        page = document.load_page(page_number)

        # Find every image on this page.
        images = page.get_images(full=True)

        # Skip pages without images.
        if not images:
            continue

        for image_index, image_info in enumerate(images, start=1):

            xref = image_info[0]

            try:
                image = document.extract_image(xref)

            except Exception as exc:
                warnings.warn(
                    f"Failed to extract image from "
                    f"{pdf_path.name} page {page_number + 1}: {exc}",
                    stacklevel=2,
                )
                continue

            image_bytes = image["image"]
            image_extension = image["ext"]

            filename = (
                f"page_{page_number + 1}"
                f"_img_{image_index}."
                f"{image_extension}"
            )

            image_path = output_dir / filename

            with open(image_path, "wb") as file:
                file.write(image_bytes)

            extracted_images.append(
                {
                    "image_path": image_path,
                    "page": page_number + 1,
                    "image_index": image_index,
                    "source": str(pdf_path),
                }
            )

    document.close()

    return extracted_images