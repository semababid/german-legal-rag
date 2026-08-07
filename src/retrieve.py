"""Query the corpus from the command line.

    python src/retrieve.py "Wie hoch darf die Kaution sein?"        # final config
    python src/retrieve.py --retriever bm25 "Kuendigungsfrist Wohnung"
    python src/retrieve.py --collection absatz_ctx -k 5 "Schimmel in der Wohnung"

Retrieval strategies live in retrievers.py; this is the thin CLI over them.
"""

from __future__ import annotations

import argparse

from retrievers import build_retriever


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--retriever", choices=("dense", "bm25", "hybrid", "rerank", "domain"), default="domain")
    ap.add_argument("--collection", default="absatz_ctx", help="dense only")
    ap.add_argument("--with-context", action="store_true", help="score against breadcrumb + text (bm25 and rerank)")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--base", choices=("dense", "bm25", "hybrid"), default="dense", help="rerank only: retriever supplying candidates")
    ap.add_argument("--candidates", type=int, default=25, help="rerank only: how many candidates to rescore")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    retriever = build_retriever(
        args.retriever,
        collection=args.collection,
        model=args.model,
        with_context=args.with_context,
        base=args.base, candidates=args.candidates,
    )

    for rank, hit in enumerate(retriever.search(args.query, args.k), 1):
        p = hit["payload"]
        print(f"{rank:2}. {hit['score']:.3f}  {p['citation']}  {p.get('title') or ''}")
        print(f"     {p['text'][:120]}")

    if hasattr(retriever, "close"):
        retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
