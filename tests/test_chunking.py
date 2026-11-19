from falakdocs.chunking import chunk_document
from falakdocs.loaders import Document, Page

TEXT = """# Guide

## Délais
Paragraphe A sur les délais.

Paragraphe B sur les délais.

## Réclamations
### Avaries
Les avaries se signalent sous 3 jours.
"""


def test_chunks_never_cross_sections():
    chunks = chunk_document(Document("g.md", "Guide", [Page(None, TEXT)]), max_chars=1000)
    assert [c.section for c in chunks] == ["Délais", "Réclamations > Avaries"]
    assert "3 jours" in chunks[1].text and "délais" not in chunks[1].text


def test_long_sections_are_split_with_overlap():
    paras = "\n\n".join(f"Paragraphe numéro {i} " + "x" * 150 for i in range(10))
    doc = Document("g.md", "Guide", [Page(None, f"## Long\n{paras}")])
    chunks = chunk_document(doc, max_chars=500, overlap=220)
    assert len(chunks) > 2
    assert all(len(c.text) <= 700 for c in chunks)
    first_tail = chunks[0].text.split("\n\n")[-1]
    assert chunks[1].text.startswith(first_tail)


def test_ids_are_stable():
    doc = Document("g.md", "Guide", [Page(None, TEXT)])
    assert [c.id for c in chunk_document(doc)] == [c.id for c in chunk_document(doc)]


def test_label_mentions_page():
    doc = Document("c.pdf", "Charte", [Page(2, "## Mots de passe\nDouze caractères.")])
    assert chunk_document(doc)[0].label == "Charte — Mots de passe, p. 2"
