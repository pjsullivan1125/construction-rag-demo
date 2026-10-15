"""Step 1: LOADING.

Read the raw files (spec markdown, RFI csv, submittal csv) into plain Python
objects. Every object keeps its metadata (where it came from) so that later
steps can cite sources. No chunking happens here.
"""
import csv
import re
from dataclasses import dataclass
from pathlib import Path

# Project root = two folders up from this file (src/rag/loader.py).
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Document:
    """One piece of raw source material, before chunking."""
    text: str          # the full text
    source_file: str   # e.g. "03_30_00_cast_in_place_concrete.md"
    doc_type: str      # "spec", "rfi" or "submittal"
    spec_section: str  # e.g. "03 30 00"
    ref: str           # RFI number, submittal number, or "" for a whole spec file


def load_specs(data_dir: Path = DATA_DIR) -> list[Document]:
    """Each spec markdown file becomes ONE document (chunking splits it later)."""
    docs = []
    for path in sorted((data_dir / "specs").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # The first line looks like: "# SECTION 03 30 00 - CAST-IN-PLACE CONCRETE"
        match = re.search(r"SECTION (\d\d \d\d \d\d)", text)
        section = match.group(1) if match else "unknown"
        docs.append(Document(text, path.name, "spec", section, ""))
    return docs


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of {column name: value} dictionaries."""
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_rfis(data_dir: Path = DATA_DIR) -> list[Document]:
    """Each RFI row becomes one document, written out as readable text."""
    docs = []
    for row in _read_rows(data_dir / "rfi_log.csv"):
        text = (
            f"{row['rfi_number']} | spec section {row['spec_section']} | "
            f"drawing {row['drawing_ref']} | status: {row['status']}. "
            f"Subject: {row['subject']}. Question: {row['question']} "
            f"Response: {row['response'] or 'No response yet.'}"
        )
        docs.append(Document(text, "rfi_log.csv", "rfi", row["spec_section"], row["rfi_number"]))
    return docs


def load_submittals(data_dir: Path = DATA_DIR) -> list[Document]:
    """Each submittal row becomes one document, written out as readable text."""
    docs = []
    for row in _read_rows(data_dir / "submittal_register.csv"):
        text = (
            f"{row['submittal_number']} | spec section {row['spec_section']} | "
            f"status: {row['status']}. {row['description']}. "
            f"Submitted by {row['submitted_by']} on {row['date_submitted']}; "
            f"reviewer {row['reviewer']}; revision {row['revision']}"
            + (f"; related {row['related_rfi']}." if row["related_rfi"] else ".")
        )
        docs.append(Document(text, "submittal_register.csv", "submittal",
                             row["spec_section"], row["submittal_number"]))
    return docs


def load_all(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load everything: specs first, then RFIs, then submittals."""
    return load_specs(data_dir) + load_rfis(data_dir) + load_submittals(data_dir)
