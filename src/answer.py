"""Ask one question and get a cited answer.

    python src/answer.py "Wie hoch darf die Kaution sein?"
    python src/answer.py --backend anthropic "Wann darf ich die Miete mindern?"

Retrieval defaults to the configuration that won phase 3: Absatz chunks embedded
with their breadcrumb, restricted to the tenancy sections.
"""

from __future__ import annotations

import argparse

from generator import build_backend, generate
from retrievers import build_retriever


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--backend", choices=("ollama", "anthropic"), default="ollama")
    ap.add_argument("--model", default=None, help="backend default if unset")
    ap.add_argument("--think", action="store_true", help="ollama only: let the model reason first, slower but better German")
    ap.add_argument("--retriever", choices=("dense", "bm25", "hybrid", "rerank", "domain"), default="domain")
    ap.add_argument("--collection", default="absatz_ctx", help="dense only")
    ap.add_argument("-k", type=int, default=6, help="how many sections to put in the prompt")
    ap.add_argument("--show-context", action="store_true", help="print the retrieved sections too")
    args = ap.parse_args()

    retriever = build_retriever(args.retriever, collection=args.collection)
    hits = retriever.search(args.question, k=args.k)
    if hasattr(retriever, "close"):
        retriever.close()

    if args.show_context:
        for rank, hit in enumerate(hits, 1):
            p = hit["payload"]
            print(f"{rank:2}. {p['citation']}  {p.get('title') or ''}")
        print()

    answer = generate(args.question, hits, build_backend(args.backend, args.model, think=args.think))

    print(answer.text)
    print()

    check = answer.citations
    if answer.refused:
        print("declined: the retrieved sections do not cover this question")
    elif check.unsupported:
        # Worth shouting about. The model cited a § it was never shown, which is
        # the one failure mode that turns a retrieval demo into a liability.
        cited = ", ".join(f"§ {s}" for s, _ in check.unsupported)
        print(f"warning: cites sections that were not retrieved: {cited}")
    elif check.total == 0:
        print("warning: answer cites nothing")
    else:
        print(f"{len(check.exact)} exact citations, {len(check.section)} at section level, all grounded")

    print(
        f"{answer.model}  {answer.input_tokens} in / {answer.output_tokens} out  "
        f"{answer.latency_ms:.0f} ms  ${answer.cost_usd:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
