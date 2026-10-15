"""Step 4a: KEYWORD SEARCH (BM25).

BM25 scores a chunk by how many of the question's words it contains, giving
more weight to RARE words ("A992") than common ones ("the"), and discounting
very long chunks. It matches exact words, so it is good at part numbers and
codes where embeddings can be fuzzy. Written by hand (standard library only)
so every line can be explained.
"""
import math
import re

K1 = 1.5   # how quickly repeating a word stops adding score
B = 0.75   # how strongly long chunks are penalised


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words/numbers: 'A992, Grade 50' -> ['a992', 'grade', '50']."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, texts: list[str]):
        self.docs = [tokenize(t) for t in texts]
        self.avg_len = sum(len(d) for d in self.docs) / len(self.docs)
        # Document frequency: in how many chunks does each word appear?
        self.doc_freq: dict[str, int] = {}
        for doc in self.docs:
            for word in set(doc):
                self.doc_freq[word] = self.doc_freq.get(word, 0) + 1

    def _idf(self, word: str) -> float:
        """Rare word -> big number, common word -> small number."""
        n, df = len(self.docs), self.doc_freq.get(word, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        """One BM25 score per chunk, in the same order as the texts."""
        query_words = tokenize(query)
        result = []
        for doc in self.docs:
            score = 0.0
            for word in query_words:
                tf = doc.count(word)   # how many times the word appears in this chunk
                if tf == 0:
                    continue
                length_penalty = 1 - B + B * len(doc) / self.avg_len
                score += self._idf(word) * tf * (K1 + 1) / (tf + K1 * length_penalty)
            result.append(score)
        return result
