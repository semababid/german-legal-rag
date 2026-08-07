"""Dense retrieval over a Qdrant collection.

Thin wrapper so the eval harness, and later the API, share one retrieval path.
Later phases add BM25 and hybrid retrievers with the same `.search()` shape, so
the eval harness does not need to know which one it is holding.

    python src/retrieve.py "Wie hoch darf die Kaution sein?"
    python src/retrieve.py --collection absatz_ctx "Kündigungsfrist Wohnung" -k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from embedder import DEFAULT_MODEL, Embedder

REPO_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = REPO_ROOT / "qdrant_storage"


class DenseRetriever:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--collection", default="absatz_raw")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    client = QdrantClient(path=str(QDRANT_PATH))
    retriever = DenseRetriever(client, Embedder(args.model), args.collection)

    for rank, hit in enumerate(retriever.search(args.query, args.k), 1):
        p = hit["payload"]
        print(f"{rank:2}. {hit['score']:.3f}  {p['citation']}  {p.get('title') or ''}")
        print(f"     {p['text'][:120]}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
