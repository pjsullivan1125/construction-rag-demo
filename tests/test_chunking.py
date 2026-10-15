import pytest

from rag.chunking import chunk_documents, sliding_windows
from rag.loader import load_all


def test_structured_one_chunk_per_row_and_per_paragraph():
    chunks = chunk_documents(load_all(), "structured")
    assert len([c for c in chunks if c.doc_type == "rfi"]) == 18
    assert len([c for c in chunks if c.doc_type == "submittal"]) == 20
    assert "03 30 00 para 2.1" in [c.id for c in chunks]


def test_structured_paragraph_contains_its_text_and_no_part_heading():
    para = next(c for c in chunk_documents(load_all(), "structured") if c.id == "03 30 00 para 2.1")
    assert "4,500 psi" in para.text and "0.45" in para.text
    assert "PART" not in para.text.split("\n", 1)[1]  # ignoring the title line


def test_sliding_windows_size_and_overlap():
    text = "x" * 1000
    windows = sliding_windows(text, size=400, overlap=50)
    assert all(len(piece) <= 400 for _, piece in windows)
    assert windows[1][0] == 350                       # second window starts 400-50 chars in
    assert windows[0][1][-50:] == windows[1][1][:50]  # neighbours share 50 chars


def test_fixed_chunks_keep_metadata():
    chunks = chunk_documents(load_all(), "fixed")
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.source_file and c.spec_section for c in chunks)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        chunk_documents([], "banana")
