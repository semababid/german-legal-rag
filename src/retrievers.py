"""Retrieval strategies, all sharing one `.search(query, k)` shape.

Keeping them behind a common interface means the eval harness never needs to know
which strategy it is holding, so every phase 3 experiment reuses the same scoring
path and the rows stay comparable.

    DenseRetriever  embedding similarity over a Qdrant collection
    BM25Retriever   lexical scoring over the same records
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = REPO_ROOT / "qdrant_storage"

# Unicode-aware, so German umlauts and ß stay inside tokens rather than splitting
# words apart. No stemming and no compound splitting: this is the honest lexical
# baseline, and decompounding is its own experiment later.
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_records(granularity: str = "absatz") -> list[dict]:
    path = REPO_ROOT / "data" / "processed" / f"bgb.{granularity}.jsonl"
    if not path.exists():
        raise SystemExit(f"No corpus at {path}. Run: python src/parse_bgb.py --law bgb")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def indexed_text(record: dict, with_context: bool) -> str:
    """Text a retriever actually scores against.

    With context, the breadcrumb and section title are prepended, matching what
    index_corpus.py embeds for the absatz_ctx collection.
    """
    if with_context and record.get("context"):
        return f"{record['context']}\n{record['text']}"
    return record["text"]


def _payload(record: dict) -> dict:
    return {
        "record_id": record["id"],
        "paragraph": record["paragraph"],
        "citation": record["citation"],
        "title": record.get("title"),
        "text": record["text"],
        "source_url": record["source_url"],
    }


class DenseRetriever:
    """Embedding similarity over a Qdrant collection."""

    name = "dense"

    def __init__(self, client, embedder, collection: str):
        self.client = client
        self.embedder = embedder
        self.collection = collection

    def search(self, query: str, k: int = 10) -> list[dict]:
        vector = self.embedder.encode([query])[0]
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=k,
            with_payload=True,
        )
        return [
            {"record_id": p.payload["record_id"], "score": p.score, "payload": p.payload}
            for p in result.points
        ]


class BM25Retriever:
    """Lexical scoring with BM25 over the same records.

    Where dense retrieval generalises across wording, BM25 rewards literal overlap.
    That is the interesting axis in German legal text: a query using the exact
    compound noun from a section ("Kuendigungsfrist", "Betriebskostenabrechnung")
    should score highly here even when the embedding misses it, and a query that
    paraphrases the same idea should do worse.
    """

    name = "bm25"

    def __init__(self, records: list[dict], with_context: bool = False):
        from rank_bm25 import BM25Okapi

        self.records = records
        self.with_context = with_context
        corpus = [tokenize(indexed_text(r, with_context)) for r in records]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        # argsort descending, taking only k rather than sorting all 4,921
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "record_id": self.records[i]["id"],
                "score": float(scores[i]),
                "payload": _payload(self.records[i]),
            }
            for i in top
        ]


class HybridRetriever:
    """Reciprocal rank fusion over two or more retrievers.

    RRF scores a record by summing 1/(rrf_k + rank) across the retrievers that
    returned it, so agreement between strategies outweighs any single strong score.
    It needs no score normalisation, which matters here because cosine similarity
    and BM25 scores are on unrelated scales.

    Each retriever is asked for `candidates` results, well past the final k, so a
    record ranked poorly by one and well by the other can still surface.
    """

    name = "hybrid"

    def __init__(self, retrievers: list, rrf_k: int = 60, candidates: int = 50):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.candidates = candidates

    def search(self, query: str, k: int = 10) -> list[dict]:
        fused: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, self.candidates), 1):
                rid = hit["record_id"]
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (self.rrf_k + rank)
                payloads.setdefault(rid, hit["payload"])

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [{"record_id": rid, "score": score, "payload": payloads[rid]} for rid, score in ranked]


def build_retriever(kind: str, collection: str = "absatz_raw", model: str | None = None,
                    granularity: str = "absatz", with_context: bool = False):
    """Construct a retriever by name, loading only what that strategy needs.

    Dense pulls in torch and the embedding model; BM25 does not, so a lexical run
    stays fast and dependency-light.
    """
    if kind == "bm25":
        return BM25Retriever(load_records(granularity), with_context)

    if kind == "dense":
        from qdrant_client import QdrantClient

        from embedder import DEFAULT_MODEL, Embedder

        client = QdrantClient(path=str(QDRANT_PATH))
        return DenseRetriever(client, Embedder(model or DEFAULT_MODEL), collection)

    if kind == "hybrid":
        dense = build_retriever("dense", collection=collection, model=model)
        bm25 = build_retriever("bm25", granularity=granularity, with_context=with_context)
        return HybridRetriever([dense, bm25])

    raise ValueError(f"unknown retriever {kind!r}")
