"""Évaluation de la recherche et des réponses sur un jeu de questions annoté.

La recherche se mesure sans LLM : le bon document est-il dans les k premiers
passages (hit@k), et à quel rang (MRR) ? C'est la mesure qui compte le plus,
parce qu'un générateur ne peut pas citer ce qu'on ne lui a pas donné.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .pipeline import Assistant


@dataclass
class Question:
    question: str
    source: str | None
    must_include: list[str]


@dataclass
class Scores:
    total: int
    hit_at_k: float
    mrr: float
    answered: float | None = None


def load_questions(path: Path) -> list[Question]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [Question(r["question"], r.get("source"), r.get("must_include", [])) for r in rows]


def retrieval_scores(assistant: Assistant, questions: list[Question]) -> Scores:
    hits = rr = 0.0
    graded = [q for q in questions if q.source]
    for q in graded:
        sources = [h.chunk.source for h in assistant.retrieve(q.question)]
        if q.source in sources:
            hits += 1
            rr += 1 / (sources.index(q.source) + 1)
    n = max(len(graded), 1)
    return Scores(len(graded), hits / n, rr / n)


def answer_scores(assistant: Assistant, questions: list[Question]) -> tuple[float, list[str]]:
    ok, failures = 0, []
    for q in questions:
        result = assistant.ask(q.question)
        text = result.answer.text.lower()
        if q.source is None:
            passed = not result.answer.found
        else:
            passed = result.answer.found and all(w.lower() in text for w in q.must_include)
        ok += passed
        if not passed:
            failures.append(f"{q.question} -> {result.answer.text[:120]}")
    return ok / max(len(questions), 1), failures
