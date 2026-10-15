"""Step 3b: VECTOR INDEX.

The "index" is just a numpy matrix: one row per chunk, 384 columns.
Searching = compare the question's vector with every row and keep the best.
No vector database needed for a corpus this small.
"""
import numpy as np

from rag.chunking import Chunk
from rag.embeddings import embed


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors  # shape (n_chunks, 384), rows have length 1

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorIndex":
        """Embed every chunk's text and store the vectors next to the chunks."""
        return cls(chunks, embed([c.text for c in chunks]))

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the k most similar chunks as (chunk, score), best first."""
        # Cosine similarity = dot product when all vectors have length 1.
        # One matrix-times-vector multiplication scores every chunk at once.
        scores = self.vectors @ query_vector
        best = np.argsort(scores)[::-1][:k]   # indexes sorted high->low, keep first k
        return [(self.chunks[i], float(scores[i])) for i in best]
