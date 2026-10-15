"""Step 4b: THE RETRIEVER.

retrieve(question, k=3) returns the k best chunks with scores.
Three modes:
  "vector"  - meaning-based (embeddings, step 3)
  "bm25"    - exact-word-based (keyword.py)
  "hybrid"  - both, merged with Reciprocal Rank Fusion
"""
import numpy as np

from rag.chunking import Chunk, chunk_documents
from rag.embeddings import embed
from rag.index import VectorIndex
from rag.keyword import BM25Index
from rag.loader import load_all

RRF_CONSTANT = 60  # standard value; dampens the advantage of rank 1 over rank 2


def reciprocal_rank_fusion(rankings: list[list[int]]) -> list[tuple[int, float]]:
    """Merge several best-first lists of chunk numbers into one.
    Each list gives a chunk 1/(60 + rank) points; points are added up.
    Using RANKS (not raw scores) avoids mixing cosine scores with BM25 scores,
    which are on completely different scales."""
    points: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_number in enumerate(ranking, start=1):
            points[chunk_number] = points.get(chunk_number, 0.0) + 1 / (RRF_CONSTANT + rank)
    return sorted(points.items(), key=lambda pair: pair[1], reverse=True)


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vector_index = VectorIndex.build(chunks)
        self.bm25 = BM25Index([c.text for c in chunks])

    def _vector_ranking(self, question: str) -> list[tuple[int, float]]:
        """(chunk number, cosine score) for ALL chunks, best first."""
        scores = self.vector_index.vectors @ embed([question])[0]
        order = np.argsort(scores)[::-1]
        return [(int(i), float(scores[i])) for i in order]

    def _bm25_ranking(self, question: str) -> list[tuple[int, float]]:
        scores = self.bm25.scores(question)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order]

    def retrieve(self, question: str, k: int = 3, mode: str = "hybrid") -> list[tuple[Chunk, float]]:
        """Top-k (chunk, score). Score meaning depends on mode:
        cosine (vector), BM25 (bm25), RRF points (hybrid)."""
        if mode == "vector":
            ranked = self._vector_ranking(question)
        elif mode == "bm25":
            ranked = self._bm25_ranking(question)
        elif mode == "hybrid":
            both = [[i for i, _ in self._vector_ranking(question)],
                    [i for i, _ in self._bm25_ranking(question)]]
            ranked = reciprocal_rank_fusion(both)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode!r}")
        return [(self.chunks[i], score) for i, score in ranked[:k]]


_cache: dict[str, Retriever] = {}  # building embeddings takes a moment; do it once per strategy


def retrieve(question: str, k: int = 3, strategy: str = "structured",
             mode: str = "hybrid") -> list[tuple[Chunk, float]]:
    """Convenience function: build (or reuse) the retriever, then search."""
    if strategy not in _cache:
        _cache[strategy] = Retriever(chunk_documents(load_all(), strategy))
    return _cache[strategy].retrieve(question, k, mode)
