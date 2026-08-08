"""Test whether the difference between two runs is real or sampling noise.

With 68 questions, a five point difference in hit@1 is three or four questions.
That is well inside the range chance produces, so a results table without some
measure of uncertainty invites conclusions it cannot support. This reports:

  McNemar exact test on hit@1, which is the right test for paired binary
  outcomes: it looks only at questions where the two runs disagree.

  A bootstrap confidence interval on the paired MRR difference, which uses the
  full ranking rather than collapsing to hit or miss and so has more power.

Both runs must have been scored on the same eval set.

    python src/compare_runs.py dense-context domain-scoped
    python src/compare_runs.py --all
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load(name: str, level: str = "paragraph") -> dict[str, dict]:
    """Load one run, projected onto the requested scoring level.

    Runs store both views per question, so switching level does not require
    re-running the retriever.
    """
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"No run at {path}")
    rows = json.load(path.open(encoding="utf-8"))["results"]
    gold = "gold_paragraphs" if level == "paragraph" else "gold_ids"
    got = "retrieved_paragraphs" if level == "paragraph" else "retrieved_ids"
    out = {}
    for r in rows:
        # older result files predate the dual view and only carry the strict keys
        out[r["id"]] = {"gold_ids": r.get(gold, r["gold_ids"]),
                        "retrieved_ids": r.get(got, r["retrieved_ids"])}
    return out


def hit_at_1(row: dict) -> int:
    return int(bool(row["retrieved_ids"]) and row["retrieved_ids"][0] in set(row["gold_ids"]))


def reciprocal_rank(row: dict) -> float:
    gold = set(row["gold_ids"])
    for i, rid in enumerate(row["retrieved_ids"], 1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def mcnemar_exact(fixed: int, broke: int) -> float:
    """Two-sided exact p-value over the discordant pairs.

    Under the null the two runs are equally likely to win any question they
    disagree on, so the discordant count follows Binomial(n, 0.5).
    """
    n = fixed + broke
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(fixed, broke) + 1))
    return min(1.0, 2 * tail / 2**n)


def bootstrap_ci(diffs: list[float], reps: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def compare(name_a: str, name_b: str, reps: int = 20000, level: str = "paragraph") -> dict:
    a, b = load(name_a, level), load(name_b, level)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit(f"{name_a} and {name_b} share no questions")
    if len(shared) != len(a) or len(shared) != len(b):
        print(f"  note: comparing {len(shared)} shared questions "
              f"({len(a)} in {name_a}, {len(b)} in {name_b})")

    ha = [hit_at_1(a[i]) for i in shared]
    hb = [hit_at_1(b[i]) for i in shared]
    ra = [reciprocal_rank(a[i]) for i in shared]
    rb = [reciprocal_rank(b[i]) for i in shared]

    fixed = sum(1 for p, q in zip(ha, hb) if p == 0 and q == 1)
    broke = sum(1 for p, q in zip(ha, hb) if p == 1 and q == 0)
    lo, hi = bootstrap_ci([q - p for p, q in zip(ra, rb)], reps)
    n = len(shared)

    return {
        "n": n,
        "hit1_a": sum(ha) / n,
        "hit1_b": sum(hb) / n,
        "fixed": fixed,
        "broke": broke,
        "p": mcnemar_exact(fixed, broke),
        "mrr_a": sum(ra) / n,
        "mrr_b": sum(rb) / n,
        "mrr_ci": (lo, hi),
    }


def verdict(res: dict) -> str:
    """A difference counts as real only if at least one test rules out no change."""
    lo, hi = res["mrr_ci"]
    sig_hit = res["p"] < 0.05
    sig_mrr = lo > 0 or hi < 0
    if sig_hit and sig_mrr:
        return "real on both"
    if sig_mrr:
        return "real on MRR only"
    if sig_hit:
        return "real on hit@1 only"
    return "not distinguishable"


def report(name_a: str, name_b: str, res: dict) -> None:
    lo, hi = res["mrr_ci"]
    print(f"{name_a}  ->  {name_b}   (n={res['n']})")
    print(f"  hit@1  {res['hit1_a']:.3f} -> {res['hit1_b']:.3f}   "
          f"fixed {res['fixed']}, broke {res['broke']}, McNemar p={res['p']:.4f}")
    print(f"  MRR    {res['mrr_a']:.3f} -> {res['mrr_b']:.3f}   "
          f"95% CI of diff [{lo:+.3f}, {hi:+.3f}]")
    print(f"  verdict: {verdict(res)}\n")


# The comparisons the results table claims, all on the 150-question set.
# paragraph_only marks a pair that cannot be scored strictly: a section index
# returns section ids, which never string-match Absatz-level ground truth, so a
# strict run of it measures the id format rather than retrieval quality.
PAIRS = [
    ("v150-raw", "v150-context", "breadcrumb context", False),
    ("v150-context", "v150-domain", "domain scoping", False),
    ("v150-context", "v150-rerank", "cross-encoder reranking", False),
    ("v150-raw", "v150-bm25", "BM25 instead of dense", False),
    ("v150-raw", "v150-hybrid", "hybrid RRF", False),
    ("v150-chunk-section", "v150-chunk-absatz", "Absatz instead of whole section", True),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", nargs="?")
    ap.add_argument("run_b", nargs="?")
    ap.add_argument("--all", action="store_true", help="run every comparison the results table claims")
    ap.add_argument("--level", choices=("paragraph", "strict"), default="paragraph")
    ap.add_argument("--reps", type=int, default=20000)
    args = ap.parse_args()

    if args.all:
        print(f"scoring level: {args.level}\n")
        for a, b, label, paragraph_only in PAIRS:
            print(f"### {label}")
            if paragraph_only and args.level != "paragraph":
                print("  skipped: only meaningful at paragraph level\n")
                continue
            report(a, b, compare(a, b, args.reps, args.level))
        return 0

    if not (args.run_a and args.run_b):
        ap.error("give two run names, or --all")
    report(args.run_a, args.run_b, compare(args.run_a, args.run_b, args.reps, args.level))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
