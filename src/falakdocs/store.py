"""Index des passages : ChromaDB pour les vecteurs, BM25 reconstruit en mémoire."""

from pathlib import Path

import chromadb

from .bm25 import BM25
from .chunking import Chunk

COLLECTION = "passages"


def _meta(c: Chunk) -> dict:
    # ChromaDB refuse les métadonnées None : la page absente vaut 0.
    return {"source": c.source, "title": c.title, "section": c.section, "page": c.page or 0}


def _chunk(cid: str, text: str, meta: dict) -> Chunk:
    return Chunk(cid, meta["source"], meta["title"], meta["section"], meta["page"] or None, text)


class Index:
    def __init__(self, path: Path | None):
        client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        self.collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None)
        self._chunks: list[Chunk] | None = None
        self._bm25: BM25 | None = None

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[_meta(c) for c in chunks],
        )
        self._chunks = self._bm25 = None

    def remove_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})
        self._chunks = self._bm25 = None

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks():
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def chunks(self) -> list[Chunk]:
        if self._chunks is None:
            data = self.collection.get(include=["documents", "metadatas"])
            self._chunks = [_chunk(i, d, m) for i, d, m in
                            zip(data["ids"], data["documents"], data["metadatas"], strict=True)]
        return self._chunks

    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([c.indexed_text() for c in self.chunks()])
        return self._bm25

    def vector_search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(query_embeddings=[vector], n_results=k,
                                    include=["documents", "metadatas", "distances"])
        hits = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
                   strict=True)
        # distance cosinus -> similarité
        return [(_chunk(i, d, m), 1.0 - dist) for i, d, m, dist in hits]
