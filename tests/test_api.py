"""Tests for the HTTP layer.

The pipeline is replaced with stand-ins, so these run in milliseconds without
loading an embedding model or reaching a language model. What is being checked
is the wiring: that a request produces the right shape, that failures come back
as sensible status codes rather than tracebacks, and that the log line carries
what an operator would need to explain an answer after the fact.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import api  # noqa: E402
from generator import Answer, verify_citations  # noqa: E402


def make_hit(record_id="bgb-551-1", citation="BGB § 551 Abs. 1", score=0.9):
    return {
        "record_id": record_id,
        "score": score,
        "payload": {
            "record_id": record_id,
            "citation": citation,
            "title": "Begrenzung und Anlage von Mietsicherheiten",
            "text": "Höchstens das Dreifache der auf einen Monat entfallenden Miete.",
            "source_url": "https://www.gesetze-im-internet.de/bgb/__551.html",
        },
    }


class StubRetriever:
    def __init__(self, hits=None, error=None):
        self.hits = hits if hits is not None else [make_hit()]
        self.error = error
        self.calls = []

    def search(self, query, k=10):
        self.calls.append((query, k))
        if self.error:
            raise self.error
        return self.hits[:k]


class StubBackend:
    name, model = "stub", "stub-model"

    def __init__(self, reply="Höchstens drei Monatsmieten [§ 551 Abs. 1].", error=None):
        self.reply = reply
        self.error = error

    def complete(self, system, user):
        if self.error:
            raise self.error
        return self.reply, 400, 60


@pytest.fixture
def client(monkeypatch):
    """A running app with the model swapped out."""
    def build(retriever=None, backend=None):
        api.pipeline = api.Pipeline(
            retriever or StubRetriever(),
            backend or StubBackend(),
            threading.Lock(),
        )
        return TestClient(api.app)
    yield build
    api.pipeline = None


def test_health_reports_not_ready_before_the_model_loads():
    # The handler is called directly. Entering TestClient as a context manager
    # runs the real lifespan, which loads bge-m3 and turns a millisecond test
    # into a ten second one that needs the corpus on disk.
    api.pipeline = None
    assert api.health()["ready"] is False


def test_asking_before_the_model_is_loaded_is_503(client):
    api.pipeline = None
    c = TestClient(api.app)
    r = c.post("/ask", json={"question": "Wie hoch darf die Kaution sein?"})
    assert r.status_code == 503


def test_ask_returns_answer_sources_and_disclaimer(client):
    c = client()
    r = c.post("/ask", json={"question": "Wie hoch darf die Kaution sein?"})
    assert r.status_code == 200
    body = r.json()

    assert body["answer"].startswith("Höchstens")
    assert body["outcome"] == "answer"
    assert body["citations"]["grounded"] is True
    assert len(body["sources"]) == 1
    assert body["sources"][0]["citation"] == "BGB § 551 Abs. 1"
    assert body["sources"][0]["url"].startswith("https://www.gesetze-im-internet.de")
    assert body["timings_ms"]["total"] >= 0
    assert body["tokens"] == {"input": 400, "output": 60}


def test_every_answer_carries_the_rdg_notice(client):
    """Not a footer somebody can miss. Every response says it."""
    r = client().post("/ask", json={"question": "Darf ich untervermieten?"})
    assert "Rechtsdienstleistungsgesetz" in r.json()["disclaimer"]


def test_no_retrieval_hits_is_a_404_not_an_invented_answer(client):
    c = client(retriever=StubRetriever(hits=[]))
    r = c.post("/ask", json={"question": "Was kostet ein Flug nach Rom?"})
    assert r.status_code == 404
    assert "no tenancy provisions" in r.json()["detail"]


def test_a_model_failure_is_a_502_not_a_traceback(client):
    c = client(backend=StubBackend(error=RuntimeError("produced no answer")))
    r = c.post("/ask", json={"question": "Wie hoch darf die Kaution sein?"})
    assert r.status_code == 502
    assert "did not return an answer" in r.json()["detail"]


def test_requests_are_rejected_before_reaching_the_model(client):
    """Validation belongs in front of a ten second model call."""
    c = client()
    assert c.post("/ask", json={"question": "hi"}).status_code == 422        # too short
    assert c.post("/ask", json={"question": "x" * 1001}).status_code == 422  # too long
    assert c.post("/ask", json={"question": "gültige Frage?", "k": 99}).status_code == 422


def test_k_is_passed_through_to_retrieval(client):
    retriever = StubRetriever(hits=[make_hit(f"bgb-{i}") for i in range(10)])
    c = client(retriever=retriever)
    c.post("/ask", json={"question": "Wie hoch darf die Kaution sein?", "k": 3})
    assert retriever.calls[-1][1] == 3


def test_search_returns_hits_without_calling_the_model(client):
    backend = StubBackend(error=AssertionError("the generator must not run"))
    c = client(backend=backend)
    r = c.get("/search", params={"q": "Kaution", "k": 1})
    assert r.status_code == 200
    assert r.json()["hits"][0]["citation"] == "BGB § 551 Abs. 1"


def test_the_log_line_can_explain_an_answer_after_the_fact(client, caplog):
    """The point of structured logging: everything needed to reconstruct why."""
    c = client()
    with caplog.at_level(logging.INFO, logger="rag"):
        c.post("/ask", json={"question": "Wie hoch darf die Kaution sein?"})

    entry = next(r.msg for r in caplog.records if isinstance(r.msg, dict)
                 and r.msg.get("event") == "ask")
    assert entry["question"] == "Wie hoch darf die Kaution sein?"
    assert entry["retrieved"][0]["id"] == "bgb-551-1"
    assert entry["retrieved"][0]["score"] == 0.9
    assert entry["citations"]["exact"] == ["§ 551 Abs. 1"]
    assert entry["input_tokens"] == 400 and entry["output_tokens"] == 60
    assert entry["retrieval_ms"] >= 0 and entry["generation_ms"] >= 0
    assert entry["request_id"]


def test_questions_can_be_kept_out_of_the_log(client, caplog, monkeypatch):
    """A question can carry personal circumstances. Whoever runs the service
    decides whether those are written to disk."""
    monkeypatch.setitem(api.SETTINGS, "log_questions", False)
    c = client()
    with caplog.at_level(logging.INFO, logger="rag"):
        c.post("/ask", json={"question": "Mein Vermieter kündigte mir wegen Eigenbedarf"})

    entry = next(r.msg for r in caplog.records if isinstance(r.msg, dict)
                 and r.msg.get("event") == "ask")
    assert "question" not in entry
    assert entry["retrieved"]          # everything else still logged


def test_log_lines_are_valid_json():
    formatted = api.JsonLines().format(logging.LogRecord(
        "rag", logging.INFO, __file__, 1,
        {"event": "ask", "question": "Wie hoch darf die Kaution sein?"}, None, None))
    parsed = json.loads(formatted)
    assert parsed["event"] == "ask"
    assert parsed["level"] == "info"
    # German text must survive rather than becoming escape sequences
    assert parsed["question"].endswith("sein?")


def test_answer_dataclass_shape_matches_what_the_route_reads():
    """Guards against generator.Answer drifting away from the API response."""
    hits = [make_hit()]
    check = verify_citations("Text [§ 551 Abs. 1].", hits)
    answer = Answer(question="q", text="Text [§ 551 Abs. 1].", backend="stub",
                    model="m", citations=check, outcome="answer",
                    input_tokens=1, output_tokens=2, latency_ms=3.0,
                    cost_usd=0.0, retrieved=["bgb-551-1"])
    for attribute in ("text", "outcome", "citations", "model",
                      "input_tokens", "output_tokens", "latency_ms"):
        assert hasattr(answer, attribute)
