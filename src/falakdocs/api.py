import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import Assistant


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if not settings.api_key:
        raise RuntimeError("FALAKDOCS_API_KEY manquant : démarrage refusé")
    assistant = assistant or Assistant.from_settings(settings)
    app = FastAPI(title="falak-docs", version="0.2.0")

    def authorized(x_api_key: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_api_key.encode(), settings.api_key.encode()):
            raise HTTPException(401, "clé d'API invalide")

    @app.get("/health")
    def health():
        return {"status": "ok", "provider": assistant.generator.name}

    @app.get("/v1/sources", dependencies=[Depends(authorized)])
    def sources():
        return assistant.index.sources()

    @app.post("/v1/ask", dependencies=[Depends(authorized)])
    def ask(req: AskIn):
        result = assistant.ask(req.question)
        return {
            "answer": result.answer.text,
            "found": result.answer.found,
            "provider": result.answer.provider,
            "citations": [{"n": i, "source": c.source, "label": c.label, "excerpt": c.text[:400]}
                          for i, c in enumerate(result.answer.sources, 1)],
            "retrieved": [{"source": h.chunk.source, "label": h.chunk.label,
                           "score": round(h.score, 4), "similarity": h.similarity}
                          for h in result.hits],
        }

    return app
