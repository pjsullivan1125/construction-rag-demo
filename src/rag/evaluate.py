"""Step 6: EVALUATION.

Question: for each test question, does the expected source appear in the
top 3 retrieved chunks ("hit rate @3")? We try both chunking strategies
and all three retrieval modes.

Run:  python -m rag.evaluate
"""
import csv
import re
from pathlib import Path

from rag.chunking import Chunk, join_file_docs
from rag.loader import load_all
from rag.retriever import retrieve

QUESTIONS_FILE = Path(__file__).resolve().parents[2] / "eval" / "questions.csv"
K = 3
MIN_OVERLAP = 100  # fixed chunks: how many characters of the target must be inside the chunk

STRATEGIES = ["structured", "fixed"]
MODES = ["vector", "bm25", "hybrid"]


def load_questions() -> list[dict[str, str]]:
    with QUESTIONS_FILE.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_expected(expected_source: str) -> list[str]:
    """'RFI-009 and 05 12 00 para 2.1' -> ['RFI-009', '05 12 00 para 2.1']."""
    return [part.strip() for part in expected_source.split(" and ")]


# ---------- Finding the expected source inside the documents ----------

def target_span(expected: str, docs=None) -> tuple[str, int, int]:
    """Where does the expected source sit? Returns (source_file, start, end) as
    character offsets in that file's joined text (the same text the fixed chunker uses)."""
    docs = docs or load_all()
    by_file: dict[str, list] = {}
    for doc in docs:
        by_file.setdefault(doc.source_file, []).append(doc)

    if "para" in expected:                        # e.g. "03 30 00 para 2.1"
        section, number = expected.split(" para ")
        doc = next(d for d in docs if d.doc_type == "spec" and d.spec_section == section)
        text, _ = join_file_docs(by_file[doc.source_file])
        start = text.index(f"### {number} ")      # the paragraph's heading line
        ends = [text.find(marker, start + 1) for marker in ("\n### ", "\n## ")]
        end = min([e for e in ends if e != -1], default=len(text))
        return doc.source_file, start, end

    doc = next(d for d in docs if d.ref == expected)   # e.g. "RFI-004"
    text, starts = join_file_docs(by_file[doc.source_file])
    start = next(offset for offset, d in starts if d is doc)
    return doc.source_file, start, start + len(doc.text)


def is_hit(chunk: Chunk, expected: str, docs=None) -> bool:
    """Does this retrieved chunk correspond to the expected source?"""
    if chunk.id == expected:                       # strategy A: ids match exactly
        return True
    if "#" not in chunk.id:                        # fixed-size chunk ids look like "file.md#3"
        return False                               # strategy A chunk with a different id
    # Strategy B: the chunk is a window of text. Count it if it contains enough of the target.
    source_file, start, end = target_span(expected, docs)
    if chunk.source_file != source_file:
        return False
    overlap = min(chunk.start + len(chunk.text), end) - max(chunk.start, start)
    return overlap >= min(MIN_OVERLAP, (end - start) // 2)


# ---------- Running the evaluation ----------

def evaluate_question(question: dict[str, str], strategy: str, mode: str, docs) -> tuple[bool, bool]:
    """Return (any_expected_found, all_expected_found) within the top K."""
    expected_list = parse_expected(question["expected_source"])
    chunks = [c for c, _score in retrieve(question["question"], K, strategy, mode)]
    found = [any(is_hit(c, e, docs) for c in chunks) for e in expected_list]
    return any(found), all(found)


def run() -> None:
    docs = load_all()
    questions = load_questions()
    answerable = [q for q in questions if q["type"] in ("fact", "conflict")]
    configs = [(s, m) for s in STRATEGIES for m in MODES]

    # results[(strategy, mode)][question id] = (any, all)
    results = {cfg: {q["id"]: evaluate_question(q, *cfg, docs) for q in answerable} for cfg in configs}

    print(f"\nHit rate: expected source appears in the top {K} chunks "
          f"({len(answerable)} answerable questions; Q10-Q12 handled separately below)\n")
    print(f"{'Chunking':<11}{'Mode':<8}{'Any source':>12}{'All sources':>13}")
    for cfg in configs:
        any_hits = sum(a for a, _ in results[cfg].values())
        all_hits = sum(b for _, b in results[cfg].values())
        print(f"{cfg[0]:<11}{cfg[1]:<8}{f'{any_hits}/{len(answerable)}':>12}{f'{all_hits}/{len(answerable)}':>13}")

    print("\nPer question (Y = expected source in top 3; Q06 needs two sources, shown as 'Y' only if both found):")
    print(f"{'':<5}" + "".join(f"{s[:4]}-{m[:4]:<6}" for s, m in configs))
    for q in answerable:
        print(f"{q['id']:<5}" + "".join(f"{'Y' if results[c][q['id']][1] else '-':<11}" for c in configs))

    report_structured(questions, docs)
    report_unanswerable(questions, answerable)


def report_structured(questions, docs) -> None:
    """Q10/Q11: counting / filtering questions. Show how many expected items the top 3 can hold."""
    print("\nStructured questions (counting / filtering) - vector search, strategy A, top 3:")
    for q in questions:
        if q["type"] != "structured":
            continue
        wanted = set(re.findall(r"RFI-\d+|S-\d+-\d+", q["expected_answer"]))
        chunks = [c for c, _ in retrieve(q["question"], K, "structured", "vector")]
        found = wanted & {c.ref for c in chunks}
        print(f"  {q['id']}: needs {len(wanted)} items {sorted(wanted)}; top 3 contained {len(found)}")
    print("  -> The top 3 can never hold all the rows, and similarity cannot count or filter by status.")
    print("     Use SQL for these (see sql/sample_queries.sql): SELECT COUNT(*) ... WHERE status = 'Open'.")


def report_unanswerable(questions, answerable) -> None:
    """Q12: nothing in the documents answers it. Show that retrieval still returns 'something'."""
    print("\nUnanswerable question (Q12) - vector search always returns its nearest neighbours:")
    top_score = lambda text: retrieve(text, 1, "structured", "vector")[0][1]
    q12 = next(q for q in questions if q["type"] == "unanswerable")
    chunk, score = retrieve(q12["question"], 1, "structured", "vector")[0]
    answerable_scores = sorted(top_score(q["question"]) for q in answerable)
    print(f"  Q12 top result: [{chunk.id}] with cosine score {score:.3f}")
    print(f"  Top scores of the answerable questions range {answerable_scores[0]:.3f} to {answerable_scores[-1]:.3f}")
    print("  -> It still returns a best guess. Here a cutoff near 0.5 would happen to separate Q12 from the")
    print("     rest, but one unanswerable question proves nothing. The real safeguard is the prompt's rule")
    print("     'say you cannot find it'; testing that needs a live model call (not run in this demo).")


if __name__ == "__main__":
    run()
