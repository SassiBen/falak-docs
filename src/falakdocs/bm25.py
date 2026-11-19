"""BM25 en mémoire, pour la recherche par mots-clés.

La recherche vectorielle rate ce qui ne se paraphrase pas : un numéro d'article
(L133-3), un nom de produit, un code d'erreur. BM25 les retrouve à coup sûr.
"""

import math
import re
import unicodedata
from collections import Counter

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "d", "l", "et", "ou", "en", "au", "aux",
    "a", "à", "est", "sont", "pour", "par", "sur", "dans", "avec", "ce", "cet", "cette", "ces",
    "qui", "que", "quoi", "dont", "il", "elle", "ils", "elles", "on", "nous", "vous", "je",
    "tu", "se", "sa", "son", "ses", "leur", "leurs", "ne", "pas", "plus", "y", "the", "of",
    "and", "to", "in", "is"
}


def tokens(text: str) -> list[str]:
    norm = unicodedata.normalize("NFKD", text.lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", norm) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(d)) for d in docs]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths) if self.docs else 0.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in q:
                if t in doc:
                    tf = doc[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
