from pathlib import Path

import pytest

from falakdocs.chunking import Chunk
from falakdocs.embeddings import HashEmbedder
from falakdocs.generation import Answer
from falakdocs.pipeline import Assistant
from falakdocs.store import Index

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


class EchoGenerator:
    name = "echo"

    def __init__(self):
        self.calls = 0

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        self.calls += 1
        return Answer(chunks[0].text, chunks[:1], self.name)


@pytest.fixture(scope="session")
def assistant() -> Assistant:
    a = Assistant(Index(None), HashEmbedder(), EchoGenerator(), top_k=5, min_similarity=0.2)
    a.ingest(CORPUS)
    return a
