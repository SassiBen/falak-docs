import hashlib
import math
import re
from typing import Protocol

import httpx


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    """Embeddings calculés localement (bge-m3 par défaut, multilingue)."""

    def __init__(self, url: str, model: str, batch: int = 32, timeout: float = 120.0):
        self.url, self.model, self.batch, self.timeout = url, model, batch, timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            r = httpx.post(f"{self.url}/api/embed", timeout=self.timeout,
                           json={"model": self.model, "input": texts[i:i + self.batch]})
            r.raise_for_status()
            vectors += r.json()["embeddings"]
        return vectors


class HashEmbedder:
    """Sac de mots haché dans un vecteur fixe. Pour les tests : déterministe, sans modèle."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
