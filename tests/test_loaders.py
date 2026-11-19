from falakdocs.loaders import load

from .conftest import CORPUS


def test_markdown_title_and_text():
    doc = load(CORPUS / "conditions-generales-transport.md")
    assert doc.title == "Conditions générales de transport"
    assert "L133-3" in doc.pages[0].text


def test_html_drops_navigation_and_keeps_headings():
    doc = load(CORPUS / "rapport-rse-2025.html")
    text = doc.pages[0].text
    assert doc.title.startswith("Rapport RSE 2025")
    assert "## Flotte" in text
    assert "Accueil | Contact" not in text


def test_pdf_keeps_page_numbers():
    doc = load(CORPUS / "charte-informatique.pdf")
    assert len(doc.pages) >= 2
    assert all(p.number for p in doc.pages)
    assert any("mot de passe" in p.text.lower() for p in doc.pages)


def test_pdf_headings_are_recovered():
    from falakdocs.chunking import chunk_document
    sections = {c.section for c in chunk_document(load(CORPUS / "charte-informatique.pdf"))}
    assert {"Mots de passe", "Supports amovibles"} <= sections
