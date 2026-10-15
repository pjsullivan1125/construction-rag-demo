# Construction RAG Demo

A small retrieval-augmented generation (RAG) project over fictional construction documents ("Lakeside Medical Pavilion" training project). Built to learn and demonstrate how chunking, embeddings, retrieval, grounding, and evaluation work together.

> All data is fictional. Built with AI assistance (Claude Code); I reviewed and can explain each part.

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
