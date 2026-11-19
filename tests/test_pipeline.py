from falakdocs.evaluate import load_questions, retrieval_scores
from falakdocs.generation import NOT_FOUND

from .conftest import ROOT


def test_unrelated_question_does_not_reach_the_model(assistant):
    before = assistant.generator.calls
    result = assistant.ask("zzzz qwxv")
    assert result.answer.text == NOT_FOUND
    assert assistant.generator.calls == before


def test_reingesting_a_source_replaces_its_passages(assistant):
    before = assistant.index.sources()["guide-accueil-salarie.md"]
    assistant.ingest(ROOT / "corpus" / "guide-accueil-salarie.md")
    assert assistant.index.sources()["guide-accueil-salarie.md"] == before


def test_retrieval_quality_floor(assistant):
    # Plancher volontairement bas : l'embedder de test n'est qu'un sac de mots.
    # Avec bge-m3, la mesure réelle est reportée dans le README.
    scores = retrieval_scores(assistant, load_questions(ROOT / "eval" / "questions.jsonl"))
    assert scores.hit_at_k >= 0.7
