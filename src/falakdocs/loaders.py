"""Lecture des documents sources : Markdown, texte, HTML et PDF.

Chaque document devient une liste de pages (une seule pour les formats sans
pagination). Le numéro de page suit le texte jusqu'à la citation finale : une
réponse qui dit « page 3 » se vérifie en dix secondes.
"""

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".md", ".txt", ".html", ".htm", ".pdf"}


@dataclass
class Page:
    number: int | None
    text: str


@dataclass
class Document:
    source: str
    title: str
    pages: list[Page]


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer"}
    BLOCK = {"p", "div", "section", "article", "li", "tr", "br", "table"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pdf_structure(text: str) -> str:
    """Un PDF n'a pas de balisage : on reconnaît un titre à sa forme (ligne courte,
    commençant par une majuscule, sans ponctuation finale, suivie d'un vrai
    paragraphe). Sans cela, une page entière devient un seul passage et la
    réponse se noie dans le reste de la page."""
    lines = [ln.strip() for ln in text.splitlines()]
    out = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        is_heading = (3 <= len(line) <= 40 and line[0].isupper()
                      and not line.endswith((".", ":", ";", ",")) and len(nxt) > 40)
        out.append(f"\n## {line}\n" if is_heading else line)
    return "\n".join(out)


def _first_heading(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def load(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(path)
        pages = [Page(i + 1, _tidy(_pdf_structure(p.extract_text() or "")))
                 for i, p in enumerate(reader.pages)]
        title = (reader.metadata.title if reader.metadata else None) or path.stem
        return Document(path.name, title, [p for p in pages if p.text])
    raw = path.read_text(encoding="utf-8")
    if suffix in (".html", ".htm"):
        parser = _TextExtractor()
        parser.feed(raw)
        text = _tidy("".join(parser.parts))
        return Document(path.name, parser.title.strip() or _first_heading(text, path.stem),
                        [Page(None, text)])
    if suffix in (".md", ".txt"):
        text = _tidy(raw)
        return Document(path.name, _first_heading(text, path.stem), [Page(None, text)])
    raise ValueError(f"format non pris en charge : {path.name}")


def load_dir(directory: Path) -> list[Document]:
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED)
    return [load(p) for p in files]
