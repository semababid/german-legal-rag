"""Score a retriever against the eval set.

Reports hit-rate@k and MRR overall and per question type, and writes the full
per-question detail to a JSON file so a later run can be compared against it.
This is the number every phase 3 experiment has to beat.

    python src/eval_retrieval.py                       # dense over absatz_raw
    python src/eval_retrieval.py --collection absatz_ctx --label dense+context
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KS = (1, 3, 5, 10)

# The sections a tenant actually asks about. Real query traffic would concentrate
# here, while the eval set deliberately also covers peripheral provisions for
# breadth, so the two are reported separately rather than averaged into one number
# that describes neither population.
CORE_SECTIONS = {
    "§ 535", "§ 536", "§ 536a", "§ 536c", "§ 538", "§ 543", "§ 546", "§ 548",
    "§ 551", "§ 553", "§ 554", "§ 556", "§ 556b", "§ 558", "§ 559", "§ 566",
    "§ 569", "§ 573", "§ 573c", "§ 574",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def first_hit_rank(retrieved_ids: list[str], gold_ids: set[str]) -> int | None:
    """1-indexed rank of the first retrieved id that is a ground-truth answer."""
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in gold_ids:
            return rank
    return None


def add_scope_breakdown(block: dict, results: list[dict], ks=DEFAULT_KS) -> dict:
    """Split the questions into core and peripheral tenancy topics.

    Averaging the two hides that they behave differently, and a deployed assistant
    would see mostly core traffic.
    """
    for scope in ("core", "peripheral"):
        subset = [r for r in results if r.get("scope") == scope]
        if subset:
            block.setdefault("by_scope", {})[scope] = compute_metrics(subset, ks)["overall"]
    return block


def compute_metrics(results: list[dict], ks=DEFAULT_KS) -> dict:
    """Aggregate per-question hits into hit-rate@k and MRR.

    `results` is a list of {type, gold_ids, retrieved_ids}. Kept as a pure
    function of that shape so it can be tested without a retriever or a model.
    hit-rate@k counts a question as hit if any ground-truth id is in the top k.
    MRR uses the reciprocal of the first hit's rank, 0 when there is no hit.
    """
    n = len(results)
    hits_at = {k: 0 for k in ks}
    rr_sum = 0.0
    by_type: dict[str, dict] = defaultdict(lambda: {"n": 0, "rr": 0.0, **{k: 0 for k in ks}})

    for r in results:
        gold = set(r["gold_ids"])
        rank = first_hit_rank(r["retrieved_ids"], gold)
        rr = 1.0 / rank if rank else 0.0
        rr_sum += rr

        t = by_type[r["type"]]
        t["n"] += 1
        t["rr"] += rr

        for k in ks:
            in_top_k = rank is not None and rank <= k
            hits_at[k] += int(in_top_k)
            t[k] += int(in_top_k)

    overall = {"n": n, "mrr": rr_sum / n if n else 0.0}
    for k in ks:
        overall[f"hit@{k}"] = hits_at[k] / n if n else 0.0

    per_type = {}
    for name, t in by_type.items():
        entry = {"n": t["n"], "mrr": t["rr"] / t["n"]}
        for k in ks:
            entry[f"hit@{k}"] = t[k] / t["n"]
        per_type[name] = entry

    return {"overall": overall, "by_type": per_type}


def _row(entry: dict, ks) -> str:
    return "  ".join(f"{entry[f'hit@{k}']:.3f}" for k in ks)


def print_report(metrics: dict, label: str, ks=DEFAULT_KS) -> None:
    """Print both scoring levels side by side.

    Paragraph level is listed first because it is what predicts answer quality:
    the generator is handed the whole section, so retrieving a sibling Absatz of
    the right section is a near miss rather than a failure. Absatz level is the
    strict reading and stays visible so neither number can be quoted alone.
    """
    head = "  ".join(f"hit@{k}" for k in ks)

    for level in ("paragraph", "strict"):
        block = metrics.get(level)
        if not block:
            continue
        o = block["overall"]
        title = "paragraph level (section counts)" if level == "paragraph" else "strict (exact Absatz)"
        print(f"\n{label}  {title}   ({o['n']} questions)")
        print(f"  {'':14}{head}  MRR")
        print(f"  {'overall':14}{_row(o, ks)}  {o['mrr']:.3f}")
        for name, t in sorted(block["by_type"].items()):
            print(f"  {name:14}{_row(t, ks)}  {t['mrr']:.3f}   (n={t['n']})")
        for name in ("core", "peripheral"):
            t = block.get("by_scope", {}).get(name)
            if t:
                print(f"  {name:14}{_row(t, ks)}  {t['mrr']:.3f}   (n={t['n']})")


def run(kind: str, collection: str, model: str | None, eval_path: Path,
        with_context: bool = False, granularity: str = "absatz",
        base: str = "dense", candidates: int = 25, match_level: str = "record", ks=DEFAULT_KS) -> tuple[dict, list[dict]]:
    from retrievers import build_retriever

    questions = load_jsonl(eval_path)
    retriever = build_retriever(
        kind, collection=collection, model=model,
        granularity=granularity, with_context=with_context,
        base=base, candidates=candidates,
    )

    from retrievers import load_records

    # Ground truth is Absatz-level ("bgb-573c-1"). Scoring at paragraph level maps
    # both sides onto the containing section, so a sibling Absatz of the right
    # section counts. That is the reading that matches what the generator receives
    # downstream, and it is also the only way an Absatz index and a section index
    # can be compared at all. Both levels are computed from one pass.
    records = load_records("absatz")
    gold_paragraph = {r["id"]: r["paragraph"] for r in records}

    max_k = max(ks)
    results = []
    latencies = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=max_k)
        latencies.append(time.perf_counter() - t0)

        paragraphs = [gold_paragraph.get(g, g) for g in q["ground_truth"]]
        scope = "core" if any(p in CORE_SECTIONS for p in paragraphs) else "peripheral"

        results.append(
            {
                "id": q["id"],
                "type": q["type"],
                "scope": scope,
                "gold_ids": q["ground_truth"],
                "retrieved_ids": [h["record_id"] for h in hits],
                "gold_paragraphs": paragraphs,
                "retrieved_paragraphs": [h["payload"]["paragraph"] for h in hits],
            }
        )

    if hasattr(retriever, "close"):
        retriever.close()

    def view(gold_key: str, retrieved_key: str) -> list[dict]:
        return [
            {"type": r["type"], "scope": r["scope"],
             "gold_ids": r[gold_key], "retrieved_ids": r[retrieved_key]}
            for r in results
        ]

    strict = view("gold_ids", "retrieved_ids")
    para = view("gold_paragraphs", "retrieved_paragraphs")

    metrics = {
        "paragraph": add_scope_breakdown(compute_metrics(para, ks), para, ks),
        "strict": add_scope_breakdown(compute_metrics(strict, ks), strict, ks),
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2] * 1000, 1),
    }
    # match_level selects which view compare_runs.py reads back from the file.
    metrics["primary"] = match_level if match_level in ("paragraph", "strict") else "paragraph"
    return metrics, results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retriever", choices=("dense", "bm25", "hybrid", "rerank", "domain"), default="dense")
    ap.add_argument("--collection", default="absatz_raw", help="dense only")
    ap.add_argument("--granularity", choices=("absatz", "paragraph"), default="absatz")
    ap.add_argument("--with-context", action="store_true", help="score against breadcrumb + text (bm25 and rerank)")
    ap.add_argument("--base", choices=("dense", "bm25", "hybrid"), default="dense", help="rerank only: retriever supplying candidates")
    ap.add_argument("--candidates", type=int, default=25, help="rerank only: how many candidates to rescore")
    ap.add_argument("--level", choices=("paragraph", "strict"), default="paragraph",
                    help="which level compare_runs treats as primary; both are always computed")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default=None, help="name for this run in the report and results file")
    ap.add_argument("--eval", type=Path, default=REPO_ROOT / "data" / "eval" / "mietrecht.jsonl")
    args = ap.parse_args()

    label = args.label or (f"{args.retriever}/{args.collection}" if args.retriever == "dense" else args.retriever)
    metrics, results = run(
        args.retriever, args.collection, args.model, args.eval,
        with_context=args.with_context, granularity=args.granularity,
        base=args.base, candidates=args.candidates, match_level=args.level,
    )
    print_report(metrics, label)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    slug = label.replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"{slug}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"label": label, "retriever": args.retriever, "collection": args.collection, "level": args.level, "metrics": metrics, "results": results}, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
