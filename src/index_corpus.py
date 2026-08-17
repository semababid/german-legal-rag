"""Embed the parsed corpus and load it into a Qdrant collection.

Runs against a local, on-disk Qdrant (no server, no Docker). For ~5,000 records
that is plenty, and the client API is identical to a real server, so phase 5 can
point at a Docker Qdrant by changing one line.

Collections are named <granularity>_<field> so the phase 3 experiments can each
build their own without colliding:

    absatz_raw       Absatz text only           (the baseline)
    absatz_ctx       breadcrumb + title + text  (the context ablation)
    paragraph_ctx    whole-section text         (the chunking comparison)

    python src/index_corpus.py                                          # absatz_raw
    python src/index_corpus.py --with-context                           # absatz_ctx
    python src/index_corpus.py --granularity paragraph --with-context   # paragraph_ctx
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qdrant_client import models

from embedder import DEFAULT_MODEL, Embedder

# Shared with the retrievers on purpose. If the indexer and the reranker ever
# built their pair text differently, the reranker would score text the index was
# never built from, and nothing would fail loudly.
from retrievers import indexed_text, qdrant_client

REPO_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = REPO_ROOT / "qdrant_storage"


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def collection_name(granularity: str, with_context: bool) -> str:
    return f"{granularity}_{'ctx' if with_context else 'raw'}"


def build(granularity: str, with_context: bool, model_name: str, batch_size: int) -> None:
    corpus_path = REPO_ROOT / "data" / "processed" / f"bgb.{granularity}.jsonl"
    if not corpus_path.exists():
        raise SystemExit(
            f"No corpus at {corpus_path}. Run: python src/parse_bgb.py --law bgb "
            f"--granularity {granularity}"
        )

    records = load_records(corpus_path)
    texts = [indexed_text(r, with_context) for r in records]

    print(f"Loading {model_name} ...")
    embedder = Embedder(model_name)
    print(f"Embedding {len(texts)} records on {embedder.device} ...")
    vectors = embedder.encode(texts, batch_size=batch_size)

    name = collection_name(granularity, with_context)
    client = qdrant_client()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
    )

    # The record id (e.g. "bgb-573c-1") lives in the payload; Qdrant point ids are
    # the row index. Everything the eval harness and UI need travels in the payload
    # so neither has to re-open the JSONL.
    client.upload_points(
        collection_name=name,
        points=[
            models.PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={
                    "record_id": r["id"],
                    "paragraph": r["paragraph"],
                    "citation": r["citation"],
                    "title": r.get("title"),
                    "text": r["text"],
                    "source_url": r["source_url"],
                },
            )
            for i, r in enumerate(records)
        ],
        batch_size=256,
    )
    client.close()

    target = os.getenv("QDRANT_URL") or QDRANT_PATH
    print(f"Indexed {len(records)} records into collection '{name}' at {target}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--granularity", choices=("absatz", "paragraph"), default="absatz")
    ap.add_argument("--with-context", action="store_true", help="prepend breadcrumb + title before embedding")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    build(args.granularity, args.with_context, args.model, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
