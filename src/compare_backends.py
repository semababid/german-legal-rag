"""Put two answer runs side by side.

Built for the local against API question, which in Germany is asked about GDPR
at least as often as about budget, but it compares any two runs over the same
questions.

Only countable things are compared. Quality is missing on purpose: the judge
built in 4.3 was worse than guessing on faithfulness, so there is no automatic
quality number worth printing here. What is left still decides the question for
most people, because a model that is cheaper and faster and cites just as
carefully is a defensible choice even before quality is settled.

    python src/compare_backends.py local-qwen35-9b api-haiku
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANSWERS_DIR = REPO_ROOT / "results" / "answers"


def load(label: str) -> dict:
    path = ANSWERS_DIR / f"{label}.json"
    if not path.exists():
        raise SystemExit(f"No run at {path}")
    return json.load(path.open(encoding="utf-8"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(p * len(ordered)), len(ordered) - 1)]


def measure(run: dict, ids: set[str]) -> dict:
    """Countable properties of one run, over a fixed set of questions."""
    rows = [r for r in run["answers"] if r["id"] in ids]
    n = len(rows)
    answered = [r for r in rows if r.get("outcome", "answer") != "refusal"]
    latencies = [r["latency_ms"] for r in rows]
    cost = sum(r["cost_usd"] for r in rows)

    return {
        "label": run["label"],
        "model": run["model"],
        "n": n,
        "invented": sum(1 for r in answered if r["citations"]["unsupported"]),
        "cross_referenced": sum(1 for r in answered if r["citations"].get("cross_referenced")),
        "fully_verifiable": sum(1 for r in answered if r["citations"].get("fully_verifiable")),
        "citing_nothing": sum(1 for r in answered if not any(
            r["citations"].get(k) for k in ("exact", "section_only", "cross_referenced"))),
        "refused": sum(1 for r in rows if r.get("outcome") == "refusal"),
        "hedged": sum(1 for r in rows if r.get("outcome") == "hedged"),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "output_tokens_median": percentile([r["output_tokens"] for r in rows], 0.50),
        "cost_usd": cost,
        "cost_per_1k_usd": cost / n * 1000 if n else 0.0,
    }


def disagreements(a: dict, b: dict, ids: set[str]) -> list[dict]:
    """Questions where the two runs behaved differently enough to look at.

    Two models answering the same question with the same sections should mostly
    agree about whether the sections answer it. Where they do not, one of them
    is wrong, and that is a shorter list to read than 150 answers.
    """
    rows_a = {r["id"]: r for r in a["answers"] if r["id"] in ids}
    rows_b = {r["id"]: r for r in b["answers"] if r["id"] in ids}
    out = []
    for qid in sorted(ids):
        ra, rb = rows_a[qid], rows_b[qid]
        why = []
        if ra.get("outcome") != rb.get("outcome"):
            why.append(f"outcome {ra.get('outcome')} against {rb.get('outcome')}")
        if bool(ra["citations"]["unsupported"]) != bool(rb["citations"]["unsupported"]):
            side = a["label"] if ra["citations"]["unsupported"] else b["label"]
            why.append(f"only {side} invented a citation")
        if why:
            out.append({"id": qid, "type": ra["type"], "why": why})
    return out


COL = 22


def row(name: str, left, right, fmt: str = "{}") -> str:
    return f"  {name:24}{fmt.format(left):>{COL}}{fmt.format(right):>{COL}}"


def report(a: dict, b: dict, diffs: list[dict]) -> None:
    # Labels are long enough to collide with the row names at a narrower width,
    # which produced a header reading "local-qwen35-9blocal-qwen35-9b-think".
    print(f"\n  {'':24}{a['label'][:COL]:>{COL}}{b['label'][:COL]:>{COL}}")
    print(f"  {'model':24}{a['model']:>{COL}}{b['model']:>{COL}}")
    print(f"  {'questions':24}{a['n']:>{COL}}{b['n']:>{COL}}")

    print("\n  citations")
    print(row("invented", a["invented"], b["invented"]))
    print(row("cross-referenced only", a["cross_referenced"], b["cross_referenced"]))
    print(row("fully verifiable", a["fully_verifiable"], b["fully_verifiable"]))
    print(row("citing nothing", a["citing_nothing"], b["citing_nothing"]))

    print("\n  refusal")
    print(row("clean refusals", a["refused"], b["refused"]))
    print(row("token plus content", a["hedged"], b["hedged"]))

    print("\n  speed and cost")
    print(row("latency p50", a["latency_p50_ms"], b["latency_p50_ms"], "{:.0f} ms"))
    print(row("latency p95", a["latency_p95_ms"], b["latency_p95_ms"], "{:.0f} ms"))
    print(row("median out tokens", a["output_tokens_median"], b["output_tokens_median"], "{:.0f}"))
    # A local model has no per-token price, and printing $0.00 next to another
    # $0.00 hides that one of them takes twenty times as long to produce. The
    # machine time is the cost there, and it is in the latency rows above.
    print(row("cost per 1k queries",
              f"${a['cost_per_1k_usd']:.2f}" if a["cost_per_1k_usd"] else "free, local",
              f"${b['cost_per_1k_usd']:.2f}" if b["cost_per_1k_usd"] else "free, local"))
    print(row("hours per 1k queries",
              a["latency_p50_ms"] * 1000 / 3_600_000, b["latency_p50_ms"] * 1000 / 3_600_000,
              "{:.1f} h"))

    print(f"\n  {len(diffs)} questions where the two behaved differently")
    for d in diffs[:10]:
        print(f"    {d['id']:8} {d['type']:14} {'; '.join(d['why'])}")
    if len(diffs) > 10:
        print(f"    ... and {len(diffs) - 10} more")

    print("\n  Quality is not compared here. The judge from 4.3 could not tell")
    print("  faithful answers from unfaithful ones, so any quality column would")
    print("  be decoration. Hand grading is the only measure this project trusts.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--json", action="store_true", help="print the measurements as JSON")
    args = ap.parse_args()

    a, b = load(args.run_a), load(args.run_b)
    # Only questions both runs answered, so a partial run cannot flatter itself
    # by having skipped the hard ones.
    ids = {r["id"] for r in a["answers"]} & {r["id"] for r in b["answers"]}
    if not ids:
        raise SystemExit("the two runs share no questions")
    if len(ids) < max(len(a["answers"]), len(b["answers"])):
        print(f"note: comparing the {len(ids)} questions both runs cover")

    ma, mb = measure(a, ids), measure(b, ids)
    diffs = disagreements(a, b, ids)

    if args.json:
        print(json.dumps({"a": ma, "b": mb, "disagreements": diffs}, indent=2))
    else:
        report(ma, mb, diffs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
