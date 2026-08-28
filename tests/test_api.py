"""End-to-end HTTP tests against the real app, with fake providers."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

SAMPLE = b"""\
# Support

## Refunds

Approved refunds are returned to the original payment method within 5 business days.

## Faults

Error code AX-7781 indicates a failed temperature sensor.
"""


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("VECTOR_STORE", "memory")
    # Seeding would add the sample corpus on top of this test's fixture.
    monkeypatch.setenv("SEED_ON_STARTUP", "false")

    from app.config import get_settings
    from app.providers.factory import get_chat_provider, get_embedding_provider

    # Settings and providers are lru_cached; clear them so the env above wins.
    get_settings.cache_clear()
    get_chat_provider.cache_clear()
    get_embedding_provider.cache_clear()

    with TestClient(create_app()) as test_client:
        test_client.post(
            "/api/ingest/files", files={"files": ("support.md", SAMPLE, "text/markdown")}
        )
        yield test_client


def test_health_reports_the_active_configuration(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "fake"
    assert body["chunks"] > 0


def test_ingest_is_idempotent(client):
    second = client.post(
        "/api/ingest/files", files={"files": ("support.md", SAMPLE, "text/markdown")}
    ).json()
    assert second["documents"][0]["skipped"] is True


def test_unsupported_file_type_is_rejected(client):
    response = client.post(
        "/api/ingest/files", files={"files": ("archive.zip", b"PK\x03\x04", "application/zip")}
    )
    assert response.status_code == 415


def test_search_exposes_the_retrieval_signals(client):
    body = client.get("/api/search", params={"q": "AX-7781"}).json()
    assert body["results"]
    top = body["results"][0]
    assert "AX-7781" in top["text"]
    # These are what make a bad ranking debuggable rather than mysterious.
    assert "dense_rank" in top and "sparse_rank" in top and "rerank_score" in top


def test_chat_returns_citations_and_cost(client):
    body = client.post("/api/chat", json={"question": "How long do refunds take?"}).json()
    assert body["grounded"] is True
    assert body["citations"]
    assert body["cost_usd"] >= 0
    assert body["timings_ms"]["total"] > 0


def test_chat_refuses_what_the_corpus_cannot_answer(client):
    body = client.post("/api/chat", json={"question": "What is the CEO's salary?"}).json()
    assert body["grounded"] is False
    assert body["citations"] == []
    assert body["refusal_reason"]


def test_chat_rejects_an_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_stream_emits_server_sent_events(client):
    with client.stream(
        "POST", "/api/chat/stream", json={"question": "How long do refunds take?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]

    assert payloads[0]["type"] == "meta"
    assert payloads[-1]["type"] == "done"
    assert payloads[-1]["citations"]


def test_deleting_an_unknown_document_is_a_404(client):
    assert client.delete("/api/documents/doc_missing").status_code == 404


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/api/chat" in schema["paths"]
