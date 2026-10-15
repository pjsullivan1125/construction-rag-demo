import numpy as np

from rag.chunking import Chunk
from rag.index import VectorIndex


def make_chunk(name: str) -> Chunk:
    return Chunk(name, name, "f", "spec", "00 00 00", "")


def test_search_ranks_by_cosine_similarity():
    # Three tiny 2-D "embeddings": east, north, and mostly-east.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    index = VectorIndex([make_chunk("east"), make_chunk("north"), make_chunk("mostly-east")], vectors)
    results = index.search(np.array([1.0, 0.0]), k=2)
    assert [chunk.id for chunk, _ in results] == ["east", "mostly-east"]
    assert results[0][1] == 1.0 and abs(results[1][1] - 0.8) < 1e-6
