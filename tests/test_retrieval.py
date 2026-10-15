from rag.keyword import BM25Index, tokenize
from rag.retriever import reciprocal_rank_fusion


def test_tokenize():
    assert tokenize("ASTM A992, Grade 50") == ["astm", "a992", "grade", "50"]


def test_bm25_prefers_chunk_with_the_rare_word():
    index = BM25Index(["the steel beam is painted", "the sealant warranty is five years",
                       "the paint is the colour grey"])
    scores = index.scores("sealant warranty")
    assert scores.index(max(scores)) == 1
    assert scores[0] == 0.0   # no query word appears in chunk 0


def test_rrf_rewards_agreement_between_rankings():
    # Chunk 2 is 1st in one list and 2nd in the other, so it beats chunks
    # that are strong in only one list.
    merged = reciprocal_rank_fusion([[2, 0, 1], [1, 2, 0]])
    assert merged[0][0] == 2
    assert {n for n, _ in merged} == {0, 1, 2}
