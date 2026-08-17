"""Tests for the switches that let the same code run on a laptop and in a container.

Phase 5.2 is mostly configuration, and configuration is where a system quietly
does the wrong thing: pointing at the wrong Qdrant, or at a language model that
is not there, fails in ways that look like a model problem rather than a wiring
one. These check the wiring without needing either service.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import generator  # noqa: E402
import retrievers  # noqa: E402


class FakeQdrant:
    """Records how it was constructed instead of connecting to anything."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        FakeQdrant.last_kwargs = kwargs


def use_fake_qdrant(monkeypatch):
    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrant)
    FakeQdrant.last_kwargs = {}


def test_qdrant_defaults_to_the_on_disk_directory(monkeypatch):
    use_fake_qdrant(monkeypatch)
    monkeypatch.delenv("QDRANT_URL", raising=False)

    retrievers.qdrant_client()
    assert "path" in FakeQdrant.last_kwargs
    assert "url" not in FakeQdrant.last_kwargs
    assert FakeQdrant.last_kwargs["path"].endswith("qdrant_storage")


def test_qdrant_url_switches_to_a_server(monkeypatch):
    """The one line that makes the container setup possible."""
    use_fake_qdrant(monkeypatch)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    retrievers.qdrant_client()
    assert FakeQdrant.last_kwargs["url"] == "http://qdrant:6333"
    assert "path" not in FakeQdrant.last_kwargs


def test_an_api_key_is_passed_only_when_set(monkeypatch):
    use_fake_qdrant(monkeypatch)
    monkeypatch.setenv("QDRANT_URL", "https://cloud.example/qdrant")

    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    retrievers.qdrant_client()
    assert FakeQdrant.last_kwargs["api_key"] is None

    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    retrievers.qdrant_client()
    assert FakeQdrant.last_kwargs["api_key"] == "secret"


def test_indexer_and_retriever_agree_on_where_qdrant_is(monkeypatch):
    """Indexing into one store and reading from another would look like a
    retrieval failure and be neither."""
    import index_corpus
    assert index_corpus.qdrant_client is retrievers.qdrant_client


def test_ollama_host_comes_from_the_environment(monkeypatch):
    """A container cannot reach the host's ollama on localhost."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert generator.OllamaBackend().host == "http://localhost:11434"

    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    assert generator.OllamaBackend().host == "http://host.docker.internal:11434"

    # an explicit argument still wins, so tests and scripts can override
    assert generator.OllamaBackend(host="http://elsewhere:1234").host == "http://elsewhere:1234"


def test_a_trailing_slash_does_not_produce_a_double_slash(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434/")
    assert generator.OllamaBackend().host == "http://host.docker.internal:11434"
