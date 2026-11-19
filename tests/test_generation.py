import json

import anthropic
import httpx

from falakdocs.chunking import Chunk
from falakdocs.generation import NOT_FOUND, ClaudeGenerator, OllamaGenerator

try:  # le SDK 1.x repose sur httpx2, les versions 0.x sur httpx
    import httpx2 as sdk_http
except ImportError:
    sdk_http = httpx

CHUNKS = [
    Chunk("a", "cgt.md", "CGT", "Réclamations", None, "Réserves confirmées sous 3 jours ouvrés."),
    Chunk("b", "cgt.md", "CGT", "Délais", None, "Livraison en J+1 dans la région."),
]


def claude_with(response_body: dict, seen: list):
    def handler(request):
        seen.append(json.loads(request.content))
        return sdk_http.Response(200, json=response_body)
    http = anthropic.DefaultHttpxClient(transport=sdk_http.MockTransport(handler))
    return anthropic.Anthropic(api_key="test", http_client=http, max_retries=0)


def message(content, stop_reason="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 30}}


def test_claude_request_uses_documents_with_citations_and_fallback():
    seen: list = []
    client = claude_with(message([{"type": "text", "text": "ok"}]), seen)
    ClaudeGenerator(client=client).answer("Délai de réserve ?", CHUNKS)
    body = seen[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "medium"}
    docs = [b for b in body["messages"][0]["content"] if b["type"] == "document"]
    assert len(docs) == 2 and docs[0]["citations"] == {"enabled": True}


def test_claude_citations_become_numbered_sources():
    cit = {"type": "char_location", "cited_text": "Réserves confirmées sous 3 jours ouvrés.",
           "document_index": 0, "document_title": "CGT", "start_char_index": 0,
           "end_char_index": 40}
    body = message([{"type": "text", "text": "Les réserves se confirment sous 3 jours ouvrés",
                     "citations": [cit]},
                    {"type": "text", "text": "."}])
    answer = ClaudeGenerator(client=claude_with(body, [])).answer("?", CHUNKS)
    assert answer.text == "Les réserves se confirment sous 3 jours ouvrés [1]."
    assert [c.id for c in answer.sources] == ["a"]


def test_claude_refusal_is_reported_not_shown():
    body = message([], stop_reason="refusal")
    answer = ClaudeGenerator(client=claude_with(body, [])).answer("?", CHUNKS)
    assert answer.refused and answer.text == NOT_FOUND


def test_ollama_keeps_only_existing_citation_numbers():
    def handler(request):
        reply = "Sous 3 jours ouvrés [1], voir aussi [7]."
        return httpx.Response(200, json={"message": {"content": reply}})
    gen = OllamaGenerator("http://ollama.test", "m", transport=httpx.MockTransport(handler))
    answer = gen.answer("?", CHUNKS)
    assert [c.id for c in answer.sources] == ["a"]
