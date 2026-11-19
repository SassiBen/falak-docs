from falakdocs.bm25 import BM25, tokens
from falakdocs.retrieval import rrf


def test_tokens_fold_accents_and_drop_stopwords():
    assert tokens("Les Réserves écrites de l'article L133-3") == ["reserves", "ecrites",
                                                                  "article", "l133"]


def test_bm25_finds_exact_references():
    bm = BM25(["délais de livraison", "article L133-3 du code de commerce", "mutuelle"])
    assert bm.top("que dit L133-3 ?", 1)[0][0] == 1


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["b", "d"]])
    assert max(fused, key=fused.get) == "b"


def test_hybrid_search_on_corpus(assistant):
    hits = assistant.retrieve("Combien de jours pour confirmer des réserves sur une avarie ?")
    assert hits[0].chunk.source == "conditions-generales-transport.md"


def test_keyword_only_match(assistant):
    hits = assistant.retrieve("B100")
    assert hits and hits[0].chunk.source == "rapport-rse-2025.html"
