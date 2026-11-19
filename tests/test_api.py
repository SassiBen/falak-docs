from fastapi.testclient import TestClient

from falakdocs.api import create_app
from falakdocs.config import Settings


def test_ask_returns_answer_and_citations(assistant):
    app = create_app(Settings(api_key="k"), assistant)
    with TestClient(app) as c:
        assert c.post("/v1/ask", json={"question": "mutuelle"}).status_code == 401
        r = c.post("/v1/ask", headers={"X-API-Key": "k"},
                   json={"question": "Quelle est la part de B100 dans le carburant ?"})
    data = r.json()
    assert r.status_code == 200
    assert data["citations"][0]["source"] == "rapport-rse-2025.html"
    assert data["retrieved"]
