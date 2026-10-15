"""Step 2: CHUNKING.

A "chunk" is the unit of text we embed, search and cite. Two strategies:
  A "structured": follow the document's own structure
                  (one chunk per spec paragraph, one per RFI / submittal row).
  B "fixed":      ignore structure; slide a window of ~400 characters
                  across the text, each window overlapping the last by 50.
The strategy is just a parameter of chunk_documents().
"""
from dataclasses import dataclass

from rag.loader import Document

CHUNK_SIZE = 400  # characters per fixed-size chunk
OVERLAP = 50      # characters shared between neighbouring fixed-size chunks


@dataclass
class Chunk:
    """One searchable piece of text, plus where it came from."""
    id: str            # what we cite, e.g. "03 30 00 para 2.1" or "RFI-004"
    text: str
    source_file: str
    doc_type: str      # "spec", "rfi" or "submittal"
    spec_section: str
    ref: str           # RFI / submittal number or spec paragraph; "" if unknown
    start: int = 0     # fixed strategy only: character offset in the source text


# ---------- Strategy A: by structure ----------

def split_spec_by_heading(doc: Document) -> list[Chunk]:
    """Split one spec file at each '### ' heading (1.1, 1.2, 2.1, ...)."""
    title = doc.text.splitlines()[0].lstrip("# ").strip()  # "SECTION 03 30 00 - ..."
    chunks = []
    # Text before the first '### ' is the title/Part headings, so skip it ([1:]).
    for piece in doc.text.split("\n### ")[1:]:
        heading, _, body = piece.partition("\n")      # "2.1 CONCRETE MIXTURES" / lines A, B, ...
        number = heading.split()[0]                   # "2.1"
        # Drop PART headings ("## PART 3 - EXECUTION") that trail the last paragraph of a part.
        body = body.split("\n## ")[0].strip()
        # Put the section title in the text so the paragraph says which spec it belongs to.
        text = f"{title}\n{heading}\n{body}"
        chunks.append(Chunk(f"{doc.spec_section} para {number}", text, doc.source_file,
                            "spec", doc.spec_section, f"para {number}"))
    return chunks


def chunk_structured(docs: list[Document]) -> list[Chunk]:
    """Strategy A: specs by paragraph, every RFI / submittal row is its own chunk."""
    chunks = []
    for doc in docs:
        if doc.doc_type == "spec":
            chunks.extend(split_spec_by_heading(doc))
        else:
            chunks.append(Chunk(doc.ref, doc.text, doc.source_file, doc.doc_type,
                                doc.spec_section, doc.ref))
    return chunks


# ---------- Strategy B: fixed size with overlap ----------

def sliding_windows(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[tuple[int, str]]:
    """Cut text into (start_offset, piece) windows of `size` chars, `overlap` shared."""
    step = size - overlap
    windows = []
    start = 0
    while True:
        windows.append((start, text[start:start + size]))
        if start + size >= len(text):   # this window reached the end of the text
            break
        start += step
    return windows


def join_file_docs(file_docs: list[Document]) -> tuple[str, list[tuple[int, Document]]]:
    """Join all documents of one file into one text. Also return where each one starts.
    (The evaluation reuses this to find where an expected RFI or paragraph sits.)"""
    text = ""
    starts = []                          # (offset where each document begins, document)
    for doc in file_docs:
        starts.append((len(text), doc))
        text += doc.text + "\n"
    return text, starts


def chunk_fixed(docs: list[Document], size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[Chunk]:
    """Strategy B: join all documents from one file into one text, then slide a window."""
    # Group documents by file (a CSV file has many rows, a spec file has one document).
    by_file: dict[str, list[Document]] = {}
    for doc in docs:
        by_file.setdefault(doc.source_file, []).append(doc)

    chunks = []
    for source_file, file_docs in by_file.items():
        text, starts = join_file_docs(file_docs)
        for n, (start, piece) in enumerate(sliding_windows(text, size, overlap)):
            # Metadata comes from the document the window STARTS in (approximate:
            # a window can run on into the next row).
            owner = [d for offset, d in starts if offset <= start][-1]
            chunks.append(Chunk(f"{source_file}#{n}", piece, source_file, owner.doc_type,
                                owner.spec_section, owner.ref, start))
    return chunks


# ---------- The one function the rest of the project calls ----------

def chunk_documents(docs: list[Document], strategy: str = "structured") -> list[Chunk]:
    """strategy = 'structured' (A) or 'fixed' (B)."""
    if strategy == "structured":
        return chunk_structured(docs)
    if strategy == "fixed":
        return chunk_fixed(docs)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
