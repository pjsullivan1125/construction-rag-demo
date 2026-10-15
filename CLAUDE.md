# Project: construction-rag-demo

A small retrieval-augmented generation (RAG) demo built to learn the concepts and show them in a portfolio.
All data is FICTIONAL ("Lakeside Medical Pavilion" training project). Never add real customer or employer data.

## Data
- `data/specs/*.md`: four fictional specification sections (concrete, steel, sealants, painting).
- `data/rfi_log.csv`: 18 RFIs (requests for information). Some are open, some closed.
- `data/submittal_register.csv`: 20 submittals with review status.
- `eval/questions.csv`: 12 test questions with expected answers and expected sources.
- `sql/`: SQLite practice (make_db.py, sample_queries.sql).

## Conventions
- Python 3.11+, virtual environment in `.venv`, dependencies in `requirements.txt`.
- Keep the code SIMPLE and well commented. The owner must be able to explain every line in an interview.
- Small functions, type hints, no clever tricks. Prefer the standard library where reasonable.
- No secrets in the repo. The Anthropic API key comes from the ANTHROPIC_API_KEY environment variable.
- After each step, explain in plain English what the code does and why, and add a short note to LEARNING_NOTES.md.

## Commands
(Update this section as commands are added.)
