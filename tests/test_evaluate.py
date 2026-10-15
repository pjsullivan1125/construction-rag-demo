from rag.chunking import Chunk, chunk_documents
from rag.evaluate import is_hit, parse_expected, target_span
from rag.loader import load_all


def test_parse_expected_splits_multiple_sources():
    assert parse_expected("RFI-009 and 05 12 00 para 2.1") == ["RFI-009", "05 12 00 para 2.1"]
    assert parse_expected("RFI-004") == ["RFI-004"]


def test_structured_hit_is_exact_id_match():
    chunk = Chunk("RFI-004", "x", "rfi_log.csv", "rfi", "03 30 00", "RFI-004")
    assert is_hit(chunk, "RFI-004") and not is_hit(chunk, "RFI-005")


def test_target_span_finds_paragraph_text():
    docs = load_all()
    source_file, start, end = target_span("03 30 00 para 2.1", docs)
    assert source_file.startswith("03_30_00") and end > start


def test_fixed_chunk_hit_requires_real_overlap_with_target():
    docs = load_all()
    source_file, start, end = target_span("RFI-004", docs)
    fixed = [c for c in chunk_documents(docs, "fixed") if c.source_file == source_file]
    hits = [c for c in fixed if is_hit(c, "RFI-004", docs)]
    assert hits and all(c.start < end and c.start + len(c.text) > start for c in hits)
    assert len(hits) < len(fixed)   # not every chunk counts
