"""Recherche hybride : vecteurs + BM25, fusionnés par rang (Reciprocal Rank Fusion).

Les deux scores ne sont pas comparables (une similarité cosinus, un score BM25
non borné) : plutôt que de les normaliser à la main, la RRF ne regarde que le
rang de chaque passage dans chaque liste. Simple, et robuste aux changements de
modèle d'embedding.
"""

from dataclasses import dataclass

from .chunking import Chunk
from .embeddings import Embedder
from .store import Index

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float
    similarity: float | None = None
    keyword: float | None = None


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


def search(index: Index, embedder: Embedder, question: str, top_k: int = 5,
           min_similarity: float = 0.45, pool: int = 20) -> list[Hit]:
    vector = embedder.embed([question])[0]
    vec_hits = [(c, s) for c, s in index.vector_search(vector, pool) if s >= min_similarity]
    chunks = index.chunks()
    kw_hits = [(chunks[i], s) for i, s in index.bm25().top(question, pool)]

    by_id = {c.id: c for c, _ in vec_hits + kw_hits}
    sim = {c.id: s for c, s in vec_hits}
    kw = {c.id: s for c, s in kw_hits}
    fused = rrf([[c.id for c, _ in vec_hits], [c.id for c, _ in kw_hits]])
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [Hit(by_id[cid], score, sim.get(cid), kw.get(cid)) for cid, score in ranked]
