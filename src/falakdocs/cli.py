import argparse
import sys
from pathlib import Path

from .config import Settings
from .evaluate import answer_scores, load_questions, retrieval_scores
from .pipeline import Assistant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="falak-docs")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="indexe un fichier ou un dossier")
    p.add_argument("path", type=Path)
    p = sub.add_parser("ask", help="pose une question")
    p.add_argument("question")
    p.add_argument("--provider", choices=["ollama", "claude"])
    p.add_argument("--show-passages", action="store_true")
    sub.add_parser("sources", help="liste les documents indexés")
    p = sub.add_parser("eval", help="mesure la recherche (et les réponses avec --answers)")
    p.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    p.add_argument("--answers", action="store_true")
    p.add_argument("--provider", choices=["ollama", "claude"])
    p = sub.add_parser("serve", help="lance l'API")
    p.add_argument("--port", type=int, default=8002)

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.cmd == "serve":
        import uvicorn

        from .api import create_app
        uvicorn.run(create_app(settings), host="0.0.0.0", port=args.port)
        return 0

    assistant = Assistant.from_settings(settings, getattr(args, "provider", None))

    if args.cmd == "ingest":
        for source, n in assistant.ingest(args.path).items():
            print(f"{n:>4} passages  {source}")
    elif args.cmd == "sources":
        for source, n in sorted(assistant.index.sources().items()):
            print(f"{n:>4} passages  {source}")
    elif args.cmd == "ask":
        result = assistant.ask(args.question)
        print(result.answer.text)
        for i, chunk in enumerate(result.answer.sources, 1):
            print(f"  [{i}] {chunk.label}")
        if args.show_passages:
            for h in result.hits:
                sim = f"{h.similarity:.2f}" if h.similarity is not None else "  - "
                print(f"  rrf={h.score:.4f} sim={sim}  {h.chunk.label}")
    elif args.cmd == "eval":
        questions = load_questions(args.questions)
        s = retrieval_scores(assistant, questions)
        print(f"recherche  hit@{settings.top_k} = {s.hit_at_k:.0%}   MRR = {s.mrr:.2f}"
              f"   ({s.total} questions)")
        if args.answers:
            rate, failures = answer_scores(assistant, questions)
            print(f"réponses   {rate:.0%} conformes")
            for f in failures:
                print(f"  ✗ {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
