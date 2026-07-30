"""Check an evaluation set against the parsed corpus before anyone trusts it.

An eval set is only worth as much as its ground truth. A single question
pointing at a section ID that does not exist, or at the wrong Absatz, quietly
corrupts every retrieval number you report afterwards. Run this after every
edit to the eval file.

    python src/validate_eval.py
    python src/validate_eval.py --eval data/eval/mietrecht.jsonl --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

QUESTION_TYPES = {"direct", "multi_section", "layperson"}
REQUIRED_FIELDS = ("id", "question", "type", "topic", "ground_truth", "reference_answer")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path.name}:{lineno} is not valid JSON: {exc}")
    return rows


def validate(eval_rows: list[dict], corpus: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for i, row in enumerate(eval_rows, 1):
        where = row.get("id") or f"line {i}"

        for field in REQUIRED_FIELDS:
            if field not in row or row[field] in ("", None, []):
                errors.append(f"{where}: missing or empty field '{field}'")

        if "id" in row:
            if row["id"] in seen_ids:
                errors.append(f"{where}: duplicate id")
            seen_ids.add(row["id"])

        if row.get("type") not in QUESTION_TYPES:
            errors.append(f"{where}: type {row.get('type')!r} not in {sorted(QUESTION_TYPES)}")

        truth = row.get("ground_truth") or []
        if not isinstance(truth, list):
            errors.append(f"{where}: ground_truth must be a list")
            continue

        for record_id in truth:
            if record_id not in corpus:
                errors.append(f"{where}: ground_truth id {record_id!r} is not in the corpus")

        # A multi_section question that points at one record is mislabelled, and
        # it will skew any per-type breakdown of the results.
        if row.get("type") == "multi_section" and len(truth) < 2:
            errors.append(f"{where}: type is multi_section but only {len(truth)} ground-truth id(s)")

        # The citation strings are redundant with the IDs on purpose: if they
        # disagree, one of the two was edited by hand and is wrong.
        stated = row.get("citations")
        if stated is not None:
            derived = [corpus[r]["citation"] for r in truth if r in corpus]
            if stated != derived:
                errors.append(f"{where}: citations {stated} do not match ground_truth {derived}")

    return errors


def summarize(eval_rows: list[dict], corpus: dict[str, dict]) -> None:
    by_type = Counter(r.get("type") for r in eval_rows)
    by_topic = Counter(r.get("topic") for r in eval_rows)
    total = len(eval_rows)

    print(f"{total} questions")
    print()
    print("  by type")
    for name, count in by_type.most_common():
        print(f"    {name:14} {count:3}  {count / total:5.0%}")
    print()
    print("  by topic")
    for name, count in by_topic.most_common():
        print(f"    {name:14} {count:3}")

    sections = Counter()
    for row in eval_rows:
        for record_id in row.get("ground_truth", []):
            if record_id in corpus:
                sections[corpus[record_id]["paragraph"]] += 1
    print()
    print(f"  distinct sections referenced: {len(sections)}")
    print(f"  most referenced: {', '.join(f'{s} ({n})' for s, n in sections.most_common(5))}")

    refs = [len(r.get("ground_truth", [])) for r in eval_rows]
    print(f"  ground-truth refs per question: min {min(refs)}, max {max(refs)}, mean {sum(refs) / len(refs):.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=REPO_ROOT / "data" / "eval" / "mietrecht.jsonl")
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "processed" / "bgb.absatz.jsonl")
    ap.add_argument("--target", type=int, default=120, help="question count the set is aiming at")
    ap.add_argument("--strict", action="store_true", help="also fail when below --target")
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"No corpus at {args.corpus}. Run: python src/parse_bgb.py --law bgb", file=sys.stderr)
        return 1
    if not args.eval.exists():
        print(f"No eval set at {args.eval}", file=sys.stderr)
        return 1

    corpus = {r["id"]: r for r in load_jsonl(args.corpus)}
    eval_rows = load_jsonl(args.eval)

    errors = validate(eval_rows, corpus)

    if errors:
        print(f"FAILED: {len(errors)} problem(s)\n", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    print(f"OK: all {len(eval_rows)} questions resolve against {len(corpus)} corpus records\n")
    summarize(eval_rows, corpus)

    if len(eval_rows) < args.target:
        message = f"\n{len(eval_rows)} of {args.target} target questions written ({args.target - len(eval_rows)} to go)"
        print(message)
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
