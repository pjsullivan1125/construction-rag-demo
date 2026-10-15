"""Step 3a: EMBEDDINGS.

An embedding turns a piece of text into a list of numbers (a "vector").
Texts with similar MEANING get vectors that point in similar directions,
even if they share no words ("cost" vs "price").
all-MiniLM-L6-v2 is a small model that runs on a normal laptop CPU and
produces 384 numbers per text.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None  # loaded once, then reused (loading is slow)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return a (number_of_texts, 384) array. Every row has length 1 (normalized),
    which lets us compute cosine similarity with a simple dot product later."""
    return _get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
