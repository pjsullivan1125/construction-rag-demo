# Construction RAG Demo

A small retrieval-augmented generation (RAG) project over fictional construction documents ("Lakeside Medical Pavilion" training project). Built to learn and demonstrate how chunking, embeddings, retrieval, grounding, and evaluation work together.

> All data is fictional. Built with AI assistance (Claude Code); I reviewed and can explain each part.

# What Code is stating with RAG (written by Patrick) 

Below explains the Loader.py file and how the Python Script (Sonnet 5.5 generated this script, but I can explain it) 

* Loader.py -> this will house the Python scripting for
* dataclass -> this is a shortcut for defining a simple class that holds data
* re -> this is a regular expression so that finding patterns in text is possible
* CSV & Path -> this is the spreadsheet style file that contains data and Path will be an easier way to work with file paths an just plain strings
* __file__ -> is the pathway of the file
* .resolve() -> will turn this into an absolute path
* .parents[2] -> will go into two different folder, rage is parent folder 0, src is parent folder 1 and project root is parent 2

* text, source_file, doc_type, spec_section, ref are all strings
* @dataclass will modify the class below and it will write the constructor

* ```def load_specs(data_dir: Path = DATA_DIR) -> list[Document]:
    docs = []
    for path in sorted((data_dir / "specs").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"SECTION (\d\d \d\d \d\d)", text)
        section = match.group(1) if match else "unknown"
        docs.append(Document(text, path.name, "spec", section, ""))
    return docs ```

  The aforementioned portion of code states that we're taking the data_dir which is the same as DATA_DIR and then make a list of documents. For the documents, sort the data directory by specs and then add a markdown file to it. The text encoding is utf-8 means that this should be read as 8 bit code. From here, re.search is going to look for text in the word followed by 3 pairs (\d\d \d\d \d\d) which means its going to look for a retrieval in the chunking index of, for example, SECTION 03 30 00. The parentheses mark the part that are needed to capture. The reason for the r before the quote means that it makes it a raw strings so that backslashes aren't interacted with.

  Lines 39-42
  ```def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f)) ```

  The purpose of this is to ensure that this is an internal help er and isn't for other files to use.  With path.open(enconding="utf-8", newline="") as f means that the file can garantee it is closed afterward even if the error occurs.

## What it does
You ask a question in plain English ("What is the maximum water-cement ratio for concrete?"). The program finds the few passages in the specs, RFI log and submittal register most likely to contain the answer, then builds a prompt that tells a language model to answer **only** from those passages and to cite their IDs. If no API key is set, it prints that prompt instead of calling a model. An evaluation script measures how often retrieval finds the right passage.

## How it works
```
specs (.md) + RFI log (.csv) + submittal register (.csv)
        |  loader.py      read files, keep metadata (file, spec section, RFI/submittal number)
        v
     chunks             chunking.py   A: by paragraph / by row     B: 400 chars, 50 overlap
        |  embeddings.py  all-MiniLM-L6-v2 -> 384 numbers per chunk
        v
  numpy matrix          index.py      cosine similarity = matrix @ question vector
        |  retriever.py   vector | bm25 (keyword.py) | hybrid (rank fusion)   -> top k chunks
        v
  grounded prompt       answer.py     "answer ONLY from sources, cite [IDs], else say you cannot find it"
        |
        v
  Claude (Haiku 5.5) if ANTHROPIC_API_KEY is set, otherwise the prompt is printed
```

## Setup
Python 3.11+ (developed on 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch (smaller)
pip install -r requirements.txt
```
The first run downloads the embedding model (about 90 MB). An API key is **optional**: `export ANTHROPIC_API_KEY=...` to get real answers; without it the prompt is printed.

## Run
```bash
export PYTHONPATH=src                                  # lets Python find the rag package
python -m rag.cli "How long must concrete be kept moist for curing?"
python -m rag.cli "Which bolt grade is used at moment connections?" --strategy fixed --mode vector -k 3
python -m rag.evaluate                                 # prints the results below
pytest -q                                              # 23 tests (add -m "not slow" to skip the 4 that load the model)
```
Options: `--strategy structured|fixed`, `--mode vector|bm25|hybrid`, `-k` (chunks to retrieve).

## Results
Hit rate @3 = how often the expected source appears in the top 3 retrieved chunks, over the 9 answerable questions (Q01-Q09) in `eval/questions.csv`. "All sources" matters for Q06, which needs two sources.

| Chunking strategy | Retrieval mode | Any expected source | All expected sources | Notes |
|---|---|---|---|---|
| A: by heading / row | vector | 9/9 | 9/9 | |
| A: by heading / row | BM25 keyword | 9/9 | 8/9 | misses the spec half of Q06 |
| A: by heading / row | hybrid | 9/9 | 8/9 | misses the spec half of Q06 |
| B: fixed 400 chars, 50 overlap | vector | 9/9 | 9/9 | |
| B: fixed 400 chars, 50 overlap | BM25 keyword | 9/9 | 8/9 | misses the spec half of Q06 |
| B: fixed 400 chars, 50 overlap | hybrid | 9/9 | 8/9 | misses the spec half of Q06 |

Questions the hit rate does not cover:
- **Q10, Q11 (counting / filtering):** the top 3 chunks held only 2 of the 4 open RFIs and 2 of the 5 pending submittals. Similarity search cannot count or filter by status, and Q11 needs 5 rows while k is 3. These belong in SQL (`sql/sample_queries.sql`).
- **Q12 (no answer in the documents):** retrieval still returns a nearest neighbour (cosine 0.446, versus 0.558-0.816 for answerable questions). Retrieval cannot say "not found"; the prompt's "say you cannot find it" rule has to do that, and I have not tested it with a live model.

## What I learned
- Vector search found everything, including both sources for Q06. BM25 and hybrid lost the spec paragraph for Q06 because the question's wording matches the RFI's words, not the spec's. Hybrid is not automatically better.
- The two chunking strategies tied. With 42-66 chunks, putting the right one in the top 3 is easy, so this dataset cannot separate them.
- Chunking by structure gives clean citations (`03 30 00 para 2.1`); fixed windows cut paragraphs mid-sentence and have no natural IDs, so I had to define "hit" for them myself.
- Some questions are not retrieval problems at all (counting, status filters). Knowing when to use SQL instead is part of the design.
- Grounding is a prompt instruction, not a guarantee; it needs testing against unanswerable questions.

## Limitations
- Tiny, fictional dataset (4 specs, 18 RFIs, 20 submittals) and only 12 test questions, so hit rates are noisy and flatter every method.
- Fixed-chunk scoring is an approximation: a chunk counts as a hit if it holds at least 100 characters of the expected paragraph or row. That threshold is my choice.
- No reranker, no score threshold, no query rewriting; k is fixed at 3.
- Vectors are rebuilt in memory on every run (fine at this size; a real system would store them).
- The live Claude call (`answer.py`) is written but untested here because no API key was available. Only prompt building is tested.
- Structured questions (counts, status) are not answered by this pipeline.
