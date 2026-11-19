"""Découpage en passages.

On coupe d'abord sur la structure (titres), puis on regroupe les paragraphes
jusqu'à une taille cible. Un passage ne chevauche jamais deux sections : mêler
la fin des « délais » et le début des « réclamations » produit des passages qui
remontent pour les deux questions et ne répondent bien à aucune.
Chaque passage garde son chemin de titres, ajouté devant le texte à indexer :
« Réclamations > Avaries » aide autant la recherche que le contenu lui-même.
"""

import hashlib
import re
from dataclasses import dataclass

from .loaders import Document

HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    section: str
    page: int | None
    text: str

    @property
    def label(self) -> str:
        where = f", p. {self.page}" if self.page else ""
        section = f" — {self.section}" if self.section else ""
        return f"{self.title}{section}{where}"

    def indexed_text(self) -> str:
        return f"{self.section}\n{self.text}" if self.section else self.text


def _sections(text: str) -> list[tuple[str, str]]:
    path: list[str] = []
    current: list[str] = []
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            if "\n".join(current).strip():
                out.append((" > ".join(path), "\n".join(current).strip()))
            level = len(m.group(1))
            # Le titre de niveau 1 est celui du document : inutile de le répéter.
            path = path[: max(level - 2, 0)] + ([m.group(2).strip()] if level > 1 else [])
            current = []
        else:
            current.append(line)
    if "\n".join(current).strip():
        out.append((" > ".join(path), "\n".join(current).strip()))
    return out


def _pack(paragraphs: list[str], max_chars: int, overlap: int) -> list[str]:
    chunks, buf = [], []
    for para in paragraphs:
        if buf and sum(len(p) for p in buf) + len(para) > max_chars:
            chunks.append("\n\n".join(buf))
            tail = buf[-1]
            buf = [tail] if len(tail) <= overlap else []
        buf.append(para)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def chunk_document(doc: Document, max_chars: int = 900, overlap: int = 220) -> list[Chunk]:
    chunks = []
    for page in doc.pages:
        for section, body in _sections(page.text):
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            for text in _pack(paragraphs, max_chars, overlap):
                key = f"{doc.source}|{page.number}|{section}|{text}"
                cid = hashlib.sha1(key.encode()).hexdigest()[:16]
                chunks.append(Chunk(cid, doc.source, doc.title, section, page.number, text))
    return chunks
