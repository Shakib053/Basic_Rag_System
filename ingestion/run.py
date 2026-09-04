"""
ingestion/run.py

Single entry point for all ingestion pipelines.

Usage
-----
Run both pipelines (default):
    python -m ingestion.run

Run only the text pipeline:
    python -m ingestion.run --text-only

Run only the image pipeline:
    python -m ingestion.run --images-only

Override chunking strategy (semantic | recursive):
    python -m ingestion.run --strategy recursive
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from vectorstore.chroma_store import verify_chroma_native_bindings
from ingestion.text_pipeline import run_text_pipeline
from ingestion.image_pipeline import run_image_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild text and/or image vector stores from the data/ directory."
    )
    parser.add_argument(
        "--strategy",
        default=None,
        metavar="STRATEGY",
        help="Chunking strategy for the text pipeline: 'semantic' (default) or 'recursive'.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--text-only",
        action="store_true",
        help="Rebuild the text vector store only; skip image ingestion.",
    )
    group.add_argument(
        "--images-only",
        action="store_true",
        help="Rebuild the image vector store only; skip text ingestion.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()

    args = _parse_args()

    uses_chroma_images = not args.text_only
    if uses_chroma_images:
        verify_chroma_native_bindings()

    if not args.images_only:
        run_text_pipeline(args.strategy)

    if not args.text_only:
        run_image_pipeline()


if __name__ == "__main__":
    main()
