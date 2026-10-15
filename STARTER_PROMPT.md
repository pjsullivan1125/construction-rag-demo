Read CLAUDE.md and look at the files in data/ and eval/. Do NOT write code yet. First, in plan mode, propose a simple design for a retrieval-augmented generation (RAG) demo over this fictional construction data, and wait for my approval.

I am learning this to explain it in interviews, so keep everything simple, readable, and well commented. After each step, explain in plain English what it does and why, and add a short note to LEARNING_NOTES.md.

The plan should cover:
1. Loading: read the spec markdown files, the RFI log, and the submittal register. Keep metadata with every chunk (source file, spec section, RFI or submittal number).
2. Chunking: implement TWO strategies so I can compare them. (A) split specs by heading or paragraph (1.2, 2.1, and so on) and make each RFI or submittal row one chunk. (B) fixed-size chunks of about 400 characters with 50 characters of overlap. Make the strategy a parameter.
3. Embeddings and index: use a small local embedding model (sentence-transformers, all-MiniLM-L6-v2) and store vectors in a simple numpy array with cosine similarity. No external vector database yet.
4. Retriever: a function retrieve(question, k=3) that returns the top chunks with scores and sources. As a stretch goal, add BM25 keyword search and a hybrid (vector plus keyword) option.
5. Grounded answer: build a prompt that says "answer only from the sources below, cite source IDs, and say you cannot find it if it is not there." If ANTHROPIC_API_KEY is set, call Claude through the Anthropic Python SDK. If not, just print the prompt.
6. Evaluation: read eval/questions.csv and report, for each chunking strategy, how often the expected source appears in the top 3 retrieved chunks. Print a small results table. Note that Q10 and Q11 are structured questions and Q12 has no answer in the documents; explain what happens with those and why.
7. A short README with setup, how to run, the results table, and honest limitations.
8. A few simple tests with pytest.

Also set up .gitignore, requirements.txt, and a virtual environment. Do not add any real or confidential data. Ask me before installing anything.
