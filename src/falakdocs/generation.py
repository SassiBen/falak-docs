"""Génération de la réponse à partir des passages retrouvés.

Deux fournisseurs, une même sortie :
- Ollama (local) : les passages sont numérotés dans le prompt et le modèle cite
  [n] ; on ne garde que les numéros qui existent.
- Claude (API) : chaque passage est envoyé comme bloc document avec les
  citations activées ; c'est l'API qui rattache chaque phrase à son passage.
Dans les deux cas, seuls les passages retrouvés quittent la machine, jamais le
corpus.
"""

import re
from dataclasses import dataclass, field
from typing import Protocol

import anthropic
import httpx

from .chunking import Chunk

NOT_FOUND = "Je ne trouve pas cette information dans les documents disponibles."

SYSTEM = (
    "Tu réponds aux questions des salariés à partir des documents internes fournis, et "
    "uniquement à partir d'eux. Si un passage traite de la question, même en d'autres "
    "termes, réponds à partir de lui. Seulement si aucun passage n'en traite, réponds "
    f"exactement : « {NOT_FOUND} » Réponds en français, de façon directe, en citant les "
    "chiffres et les délais tels qu'ils sont écrits."
)


@dataclass
class Answer:
    text: str
    sources: list[Chunk] = field(default_factory=list)
    provider: str = ""
    refused: bool = False

    @property
    def found(self) -> bool:
        return not self.refused and NOT_FOUND not in self.text


class Generator(Protocol):
    name: str

    def answer(self, question: str, chunks: list[Chunk]) -> Answer: ...


class OllamaGenerator:
    name = "ollama"

    def __init__(self, url: str, model: str, timeout: float = 180.0,
                 transport: httpx.BaseTransport | None = None):
        self.model = model
        self.http = httpx.Client(base_url=url, timeout=timeout, transport=transport)

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        excerpts = "\n\n".join(f"[{i}] ({c.label})\n{c.text}" for i, c in enumerate(chunks, 1))
        prompt = (f"Documents :\n\n{excerpts}\n\nQuestion : {question}\n\n"
                  "Cite tes sources avec leur numéro entre crochets, par exemple [2].")
        r = self.http.post("/api/chat", json={
            "model": self.model, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
        })
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)
                        if 1 <= int(n) <= len(chunks)})
        return Answer(text, [chunks[n - 1] for n in cited], self.name)


class ClaudeGenerator:
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium",
                 client: anthropic.Anthropic | None = None):
        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        documents = [{
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c.text},
            "title": c.label,
            "citations": {"enabled": True},
        } for c in chunks]
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=8000,
            system=SYSTEM,
            output_config={"effort": self.effort},
            # Si un classifieur de sécurité refuse la requête, l'API la rejoue sur
            # un modèle de repli plutôt que de renvoyer une réponse vide.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user",
                       "content": [*documents, {"type": "text", "text": question}]}],
        )
        if response.stop_reason == "refusal":
            return Answer(NOT_FOUND, [], self.name, refused=True)

        parts: list[str] = []
        order: list[int] = []
        for block in response.content:
            if block.type != "text":
                continue
            parts.append(block.text)
            marks = []
            for citation in block.citations or []:
                idx = citation.document_index
                if idx not in order:
                    order.append(idx)
                mark = f"[{order.index(idx) + 1}]"
                if mark not in marks:
                    marks.append(mark)
            if marks:
                parts.append(" " + "".join(marks))
        text = "".join(parts).strip()
        return Answer(text, [chunks[i] for i in order if i < len(chunks)], self.name)
