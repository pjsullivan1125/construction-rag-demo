"""End-to-end checks that use the REAL embedding model (marked slow)."""
import pytest

from rag.retriever import retrieve

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("strategy", ["structured", "fixed"])
def test_vector_search_finds_water_cement_ratio(strategy):
    results = retrieve("What is the maximum water-cement ratio for concrete?", 3, strategy, "vector")
    assert any("0.45" in chunk.text for chunk, _ in results)


def test_hybrid_returns_k_results_with_sources():
    results = retrieve("How long is the sealant warranty?", 3, "structured", "hybrid")
    assert len(results) == 3
    assert results[0][0].id == "07 92 00 para 1.3"
    assert all(chunk.source_file for chunk, _ in results)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        retrieve("anything", 3, "structured", "banana")
