"""
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

from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

CLIP_MODEL_NAME = "clip-ViT-B-32"

class CLIPEmbeddings(Embeddings):
    """
    LangChain-compatible CLIP embedding class.

    Chroma calls:

        embed_documents()

    when indexing data.

    Chroma calls:

        embed_query()

    when performing similarity search.
    """

    def __init__(self):

        # Load CLIP only once.
        self.model = SentenceTransformer(CLIP_MODEL_NAME)

    def embed_documents(self, image_paths):

        embeddings = []

        for image_path in image_paths:

            image = Image.open(image_path).convert("RGB")

            vector = self.model.encode(
                image,
                normalize_embeddings=True,
            )

            embeddings.append(vector.tolist())

        return embeddings

    def embed_query(self, text):

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()