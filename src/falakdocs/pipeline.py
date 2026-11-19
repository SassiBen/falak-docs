from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_document
from .config import Settings
from .embeddings import Embedder, OllamaEmbedder
from .generation import NOT_FOUND, Answer, ClaudeGenerator, Generator, OllamaGenerator
from .loaders import load, load_dir
from .retrieval import Hit, search
from .store import Index


@dataclass
class Result:
    answer: Answer
    hits: list[Hit]


class Assistant:
    def __init__(self, index: Index, embedder: Embedder, generator: Generator,
                 top_k: int = 5, min_similarity: float = 0.45):
        self.index = index
        self.embedder = embedder
        self.generator = generator
        self.top_k = top_k
        self.min_similarity = min_similarity

    @classmethod
    def from_settings(cls, s: Settings, provider: str | None = None) -> "Assistant":
        provider = provider or s.provider
        if provider == "claude":
            generator: Generator = ClaudeGenerator(s.claude_model, s.claude_effort)
        elif provider == "ollama":
            generator = OllamaGenerator(s.ollama_url, s.ollama_model)
        else:
            raise ValueError(f"fournisseur inconnu : {provider}")
        return cls(Index(s.index_dir), OllamaEmbedder(s.ollama_url, s.embed_model), generator,
                   s.top_k, s.min_similarity)

    def ingest(self, path: Path) -> dict[str, int]:
        docs = load_dir(path) if path.is_dir() else [load(path)]
        added = {}
        for doc in docs:
            chunks = chunk_document(doc)
            self.index.remove_source(doc.source)
            if chunks:
                self.index.add(chunks, self.embedder.embed([c.indexed_text() for c in chunks]))
            added[doc.source] = len(chunks)
        return added

    def retrieve(self, question: str) -> list[Hit]:
        return search(self.index, self.embedder, question, self.top_k, self.min_similarity)

    def ask(self, question: str) -> Result:
        hits = self.retrieve(question)
        if not hits:
            # Rien de pertinent : on ne sollicite pas le modèle, qui serait tenté
            # de répondre avec ce qu'il sait d'ailleurs.
            return Result(Answer(NOT_FOUND, [], "aucun"), [])
        return Result(self.generator.answer(question, [h.chunk for h in hits]), hits)
