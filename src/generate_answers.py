"""Generate an answer for every question in the eval set.

This produces the file the next two steps read: the hand-graded calibration
sample in 4.2 is drawn from it, and the judge in 4.3 scores it. Running it is
the only expensive step in phase 4, so the output is written per question as it
goes and a rerun skips whatever is already there.

    python src/generate_answers.py --backend ollama --label local-qwen
    python src/generate_answers.py --backend anthropic --label api-haiku --limit 20

The summary at the end is deliberately not a quality score. Refusal rate and
ungrounded-citation rate are countable without a judge; whether an answer is
correct is not, and pretending otherwise here would put a number in front of the
calibration step that is supposed to produce it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generator import build_backend, generate
from metrics import percentile
from retrievers import build_retriever

REPO_ROOT = Path(__file__).resolve().parent.parent
ANSWERS_DIR = REPO_ROOT / "results" / "answers"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in json.load(path.open(encoding="utf-8"))["answers"]}


def rescore(path: Path) -> list[dict]:
    """Recheck saved answers against the sections they were shown.

    Generation is the expensive step and the answers are deterministic, so a fix
    to the citation checker should not cost another run of the model. The
    retrieved record ids are stored with each answer, which is enough to rebuild
    what the model saw and score it again.
    """
    from generator import classify_response, verify_citations
    from retrievers import load_records

    payloads = {r["id"]: r for r in load_records("absatz")}
    data = json.load(path.open(encoding="utf-8"))

    for row in data["answers"]:
        hits = [{"payload": payloads[rid]} for rid in row["retrieved"] if rid in payloads]
        check = verify_citations(row["answer"], hits)
        row["citations"] = check.as_dict()
        row["outcome"] = classify_response(row["answer"], check)
        row["refused"] = row["outcome"] == "refusal"

    data["summary"] = summarise(data["answers"])
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return data["answers"]


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    # An answer that cites and then hedges is not a refusal, so it stays in the
    # pool whose citations get checked.
    refused = sum(1 for r in rows if r.get("outcome", "answer") == "refusal")
    hedged = sum(1 for r in rows if r.get("outcome") == "hedged")
    answered = [r for r in rows if r.get("outcome", "answer") != "refusal"]
    invented = sum(1 for r in answered if r["citations"]["unsupported"])
    crossref = sum(1 for r in answered if r["citations"].get("cross_referenced"))
    uncited = sum(1 for r in answered if not any(
        r["citations"].get(k) for k in ("exact", "section_only", "cross_referenced")))
    latencies = [r["latency_ms"] for r in rows]

    return {
        "n": n,
        "refused": refused,
        "hedged": hedged,
        "refusal_rate": round(refused / n, 3),
        "invented_citations": invented,
        "cross_referenced_citations": crossref,
        "answers_citing_nothing": uncited,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "cost_per_1k_queries_usd": round(sum(r["cost_usd"] for r in rows) / n * 1000, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("ollama", "anthropic"), default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--think", action="store_true", help="ollama only: let the model reason first, slower but better German")
    ap.add_argument("--label", default=None, help="name for the output file")
    ap.add_argument("--retriever", choices=("dense", "bm25", "hybrid", "rerank", "domain"), default="domain")
    ap.add_argument("--collection", default="absatz_ctx")
    ap.add_argument("-k", type=int, default=6, help="how many sections to put in each prompt")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many questions")
    ap.add_argument("--redo", action="store_true", help="regenerate answers already in the output file")
    ap.add_argument("--rescore", action="store_true", help="recheck saved answers without calling the model again")
    ap.add_argument("--eval", type=Path, default=REPO_ROOT / "data" / "eval" / "mietrecht.jsonl")
    args = ap.parse_args()

    questions = load_jsonl(args.eval)
    if args.limit:
        questions = questions[: args.limit]

    if args.rescore:
        if not args.label:
            raise SystemExit("--rescore needs --label naming the run to recheck")
        path = ANSWERS_DIR / f"{args.label}.json"
        if not path.exists():
            raise SystemExit(f"No run at {path}")
        rows = rescore(path)
        stats = summarise(rows)
        print(f"rescored {args.label}: {json.dumps(stats, indent=2)}")
        return 0

    backend = build_backend(args.backend, args.model, think=args.think)
    label = args.label or (
        f"{args.backend}-{backend.model.replace(':', '-')}" + ("-think" if args.think else "")
    )

    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANSWERS_DIR / f"{label}.json"
    done = {} if args.redo else load_existing(out_path)
    if done:
        print(f"{len(done)} answers already in {out_path.name}, generating the rest")

    retriever = build_retriever(args.retriever, collection=args.collection)
    rows: list[dict] = []
    failures: list[dict] = []

    try:
        for i, q in enumerate(questions, 1):
            if q["id"] in done:
                rows.append(done[q["id"]])
                continue

            hits = retriever.search(q["question"], k=args.k)
            try:
                answer = generate(q["question"], hits, backend)
            except Exception as exc:
                # One stalled question must not cost a run that takes hours.
                # ev-009 spent more than ten minutes reasoning about § 543 and
                # took the whole job down with it. Failures are recorded by id
                # and skipped; rerunning picks up exactly those, because
                # anything already answered is left alone.
                failures.append({"id": q["id"], "error": f"{type(exc).__name__}: {exc}"})
                print(f"{i:3}/{len(questions)}  {q['id']}  FAILED  {type(exc).__name__}")
                continue

            row = {"id": q["id"], "type": q["type"], "topic": q["topic"],
                   "ground_truth": q["ground_truth"], **answer.as_dict()}
            rows.append(row)

            flag = answer.outcome if answer.outcome != "answer" else (
                "UNGROUNDED" if answer.citations.unsupported else "ok")
            # flushed, because a run this long is usually watched through a
            # redirected log rather than a terminal, and Python buffers those
            print(f"{i:3}/{len(questions)}  {q['id']}  {answer.latency_ms:6.0f} ms  {flag}",
                  flush=True)

            # Written every question rather than at the end. A run over 150
            # questions against a local 14B model takes long enough that losing
            # it to an interrupted terminal would be a real cost.
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump({"label": label, "backend": args.backend, "model": backend.model,
                           "k": args.k, "retriever": args.retriever,
                           "failures": failures, "answers": rows}, fh,
                          ensure_ascii=False, indent=2)
    finally:
        if hasattr(retriever, "close"):
            retriever.close()

    stats = summarise(rows)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"label": label, "backend": args.backend, "model": backend.model,
                   "k": args.k, "retriever": args.retriever,
                   "summary": stats, "failures": failures, "answers": rows}, fh,
                  ensure_ascii=False, indent=2)

    print(f"\n{label}  {stats['n']} answers")
    if failures:
        print(f"  failed, rerun to retry  {len(failures)}: "
              f"{', '.join(f['id'] for f in failures[:8])}")
    print(f"  refused                {stats['refused']} ({stats['refusal_rate']:.1%})")
    print(f"  answered then hedged   {stats['hedged']}")
    print(f"  invented citations     {stats['invented_citations']}")
    print(f"  cross-referenced only  {stats['cross_referenced_citations']}")
    print(f"  citing nothing         {stats['answers_citing_nothing']}")
    print(f"  latency p50 / p95      {stats['latency_p50_ms']:.0f} / {stats['latency_p95_ms']:.0f} ms")
    print(f"  cost per 1k queries    ${stats['cost_per_1k_queries_usd']:.2f}")
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
