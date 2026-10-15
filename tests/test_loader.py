from rag.loader import load_all, load_rfis, load_specs, load_submittals


def test_counts():
    assert len(load_specs()) == 4
    assert len(load_rfis()) == 18
    assert len(load_submittals()) == 20


def test_metadata_is_kept():
    for doc in load_all():
        assert doc.source_file and doc.spec_section and doc.doc_type


def test_rfi_004_text_contains_response():
    rfi = next(d for d in load_rfis() if d.ref == "RFI-004")
    assert "4,500 psi" in rfi.text and rfi.spec_section == "03 30 00"


def test_spec_section_number_read_from_title():
    sections = {d.spec_section for d in load_specs()}
    assert sections == {"03 30 00", "05 12 00", "07 92 00", "09 91 23"}
