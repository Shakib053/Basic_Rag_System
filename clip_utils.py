"""
clip_utils.py
--------------
Helper functions for working with the CLIP model.

CLIP (Contrastive Language-Image Pretraining) is a model that understands
BOTH images and text, and places them into the SAME "embedding space" (a
list of numbers, aka a vector).

That means: a photo of a dog and the text "a photo of a dog" end up with
very similar vectors. This is exactly why CLIP works for multimodal RAG:

  - We embed every image with CLIP and save the vectors.
  - When the user types a text question, we embed the question with CLIP too.
  - Because text and images share the same space, we can measure how close
    the question's vector is to each image's vector, and find the images
    that best match what the user asked about.

We use the 'sentence-transformers' library (already in your requirements)
because it ships a ready-to-use CLIP model with a simple .encode() method -
no manual image preprocessing code needed on our end.
"""

from sentence_transformers import SentenceTransformer
from PIL import Image

# Load the CLIP model ONE time when this file is first imported.
# "clip-ViT-B-32" is a small/fast CLIP model - good for small projects.
# The first run will download the model (a few hundred MB) and cache it.
clip_model = SentenceTransformer("clip-ViT-B-32")


def embed_image(image_path: str):
    """
    Turn an image file into a CLIP embedding (a list of numbers).

    Args:
        image_path: path to an image file, e.g. "data/images/cat.jpg"

    Returns:
        A plain Python list of floats (Chroma needs plain lists, not
        numpy arrays, so we call .tolist() at the end).
    """
    # .convert("RGB") avoids errors on PNGs that have a transparency channel
    image = Image.open(image_path).convert("RGB")
    embedding = clip_model.encode(image)
    return embedding.tolist()


def embed_text_query(text: str):
    """
    Turn a plain text string (like a user's question) into a CLIP embedding.

    This uses the SAME model as embed_image(), which is what makes the
    resulting vector directly comparable to image vectors.

    Args:
        text: any string, e.g. "a photo of a red car"

    Returns:
        A plain Python list of floats.
    """
    embedding = clip_model.encode(text)
    return embedding.tolist()