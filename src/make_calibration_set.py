"""Draw the sample of answers to be graded by hand.

These grades are the reference the phase 4.3 judge is measured against, so they
have to come from a person. A model grading another model's output produces
agreement figures that describe two models agreeing, which says nothing about
whether either is right.

Sampling is stratified rather than random, and the strata are the reason. A
plain random draw of 30 from 150 would be expected to contain three or four
answers with an ungrounded citation and two or three that used the refusal
token, which is too few to say anything about the judge on exactly the cases
that matter. Oversampling them fixes that and introduces a different problem:
agreement measured on a sample richer in hard cases understates what the judge
would score on real traffic.

Both are avoided by recording which stratum each answer came from and reporting
agreement per stratum instead of as one average. That is the same reasoning the
retrieval results use for core against peripheral topics.

    python src/make_calibration_set.py --run local-qwen35-9b
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANSWERS_DIR = REPO_ROOT / "results" / "answers"
DEFAULT_OUT = REPO_ROOT / "data" / "eval" / "calibration.jsonl"

# Every hedged answer is taken, not sampled. There are only 12, they are the
# cases no rule could classify, and leaving any out means the ambiguity stays
# unresolved rather than being decided by a person.
STRATA = [
    ("hedged", "used the refusal token", None),
    ("invented", "cited a section named nowhere in the text shown", None),
    ("cross_referenced", "cited a section named in the text but never shown", 8),
    ("uncited", "cited nothing at all", None),
    ("clean", "every citation points at text it read", 8),
]


def load_run(label: str) -> list[dict]:
    path = ANSWERS_DIR / f"{label}.json"
    if not path.exists():
        raise SystemExit(f"No run at {path}. Generate answers first.")
    return json.load(path.open(encoding="utf-8"))["answers"]


def stratum_of(row: dict) -> str:
    """One label per answer, in priority order so the strata do not overlap."""
    c = row["citations"]
    if row["outcome"] == "hedged":
        return "hedged"
    if c["unsupported"]:
        return "invented"
    if c.get("cross_referenced"):
        return "cross_referenced"
    if not c["exact"] and not c["section_only"]:
        return "uncited"
    return "clean"


def draw(rows: list[dict], seed: int = 0) -> list[dict]:
    """Take every answer from the small strata, sample the large ones.

    Within a stratum the draw is seeded, so the set can be rebuilt exactly. A
    calibration set that shifts between runs cannot be compared against itself.
    """
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict]] = {}
    for row in rows:
        by_stratum.setdefault(stratum_of(row), []).append(row)

    picked = []
    for name, _, cap in STRATA:
        pool = sorted(by_stratum.get(name, []), key=lambda r: r["id"])
        if cap is None or len(pool) <= cap:
            chosen = pool
        else:
            # spread across question types rather than letting one dominate
            chosen = []
            by_type: dict[str, list[dict]] = {}
            for r in pool:
                by_type.setdefault(r["type"], []).append(r)
            for t in sorted(by_type):
                rng.shuffle(by_type[t])
            while len(chosen) < cap:
                for t in sorted(by_type):
                    if by_type[t] and len(chosen) < cap:
                        chosen.append(by_type[t].pop())
            chosen.sort(key=lambda r: r["id"])
        for r in chosen:
            picked.append((name, r))

    return picked


def build_rows(picked, questions: dict[str, dict], records: dict[str, dict]) -> list[dict]:
    """Everything a grader needs to judge an answer, in one record.

    The retrieved section text is copied in rather than referenced. Grading
    faithfulness means checking a claim against the text the model was shown,
    and that is not a job to do with a second terminal open.
    """
    out = []
    for stratum, row in picked:
        q = questions.get(row["id"], {})
        sections = []
        for rid in row["retrieved"]:
            rec = records.get(rid)
            if rec:
                sections.append({
                    "id": rid,
                    "citation": rec["citation"],
                    "title": rec.get("title"),
                    "text": rec["text"],
                })
        out.append({
            "id": row["id"],
            "stratum": stratum,
            "type": row["type"],
            "question": row["question"],
            "answer": row["answer"],
            "outcome": row["outcome"],
            "citation_check": row["citations"],
            "reference_answer": q.get("reference_answer", ""),
            "gold_citations": q.get("citations", []),
            "sections_shown": sections,
            # filled in by hand; see grade.py
            "faithful": None,
            "relevant": None,
            "refusal_correct": None,
            "note": "",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="local-qwen35-9b", help="label of the answer run to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--eval", type=Path, default=REPO_ROOT / "data" / "eval" / "mietrecht.jsonl")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(
            f"{args.out.relative_to(REPO_ROOT)} already exists. Grading it again from "
            f"scratch would throw away work; delete it deliberately if that is the intent."
        )

    from retrievers import load_records

    rows = load_run(args.run)
    with args.eval.open(encoding="utf-8") as fh:
        questions = {json.loads(l)["id"]: json.loads(l) for l in fh if l.strip()}
    records = {r["id"]: r for r in load_records("absatz")}

    picked = draw(rows, args.seed)
    out_rows = build_rows(picked, questions, records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in out_rows:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1

    print(f"{len(out_rows)} answers drawn from {len(rows)} for hand grading\n")
    for name, description, _ in STRATA:
        n = counts.get(name, 0)
        total = sum(1 for row in rows if stratum_of(row) == name)
        print(f"  {name:12} {n:2} of {total:3}   {description}")
    print(f"\nwrote {args.out.relative_to(REPO_ROOT)}")
    print("Grade them with: python src/grade.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
