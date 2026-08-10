"""Grade the calibration answers by hand, one at a time.

Run this yourself. The grades produced here are the reference the phase 4.3
judge is scored against, and a reference written by a model measures nothing
except two models agreeing with each other.

It shows the question, the sections the model was given, its answer, and the
reference answer from the eval set, then asks three questions. Progress is
saved after every grade, so it can be stopped and resumed.

    python src/grade.py                # grade the next ungraded answer
    python src/grade.py --review       # reread what has been graded
    python src/grade.py --summary      # counts so far

The three judgments, kept deliberately few:

  faithful         is every claim in the answer supported by the sections shown?
                   This is the one that matters. An answer can be correct German
                   law and still fail here if it came from the model's memory
                   rather than the text, because that is unverifiable at runtime.

  relevant         does it answer the question that was asked?

  refusal_correct  only for answers using the refusal token. Was declining the
                   right call, given the sections shown? Asked separately
                   because a correct refusal is a success and scoring it as an
                   unfaithful answer would punish the behaviour we want.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "data" / "eval" / "calibration.jsonl"

SCALE = {
    "y": "yes",
    "p": "partly",
    "n": "no",
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"No calibration set at {path}.\n"
            f"Build one first: python src/make_calibration_set.py"
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(line, width=88, initial_indent=indent, subsequent_indent=indent)
        for line in text.splitlines() or [""]
    )


def show(row: dict, position: str) -> None:
    print("\n" + "=" * 92)
    print(f"{position}   {row['id']}   {row['type']}   stratum: {row['stratum']}")
    print("=" * 92)

    print("\nFRAGE")
    print(wrap(row["question"]))

    print("\nVORGELEGTE VORSCHRIFTEN")
    for s in row["sections_shown"]:
        head = s["citation"] + (f" — {s['title']}" if s.get("title") else "")
        print(f"  {head}")
        print(wrap(s["text"], indent="      "))

    print("\nANTWORT DES MODELLS")
    print(wrap(row["answer"]))

    if row.get("reference_answer"):
        print("\nREFERENZ AUS DEM EVAL-SET")
        print(wrap(row["reference_answer"]))
        if row.get("gold_citations"):
            print(wrap("erwartete Fundstellen: " + ", ".join(row["gold_citations"])))

    c = row["citation_check"]
    print("\nAUTOMATISCHE PRÜFUNG")
    print(f"    exact: {c['exact'] or '-'}   section only: {c['section_only'] or '-'}")
    print(f"    nicht vorgelegt: {c['unsupported'] or '-'}   outcome: {row['outcome']}")
    print()


def ask(question: str, allow_skip: bool = True) -> str | None:
    options = "y=yes  p=partly  n=no" + ("  s=skip  q=quit" if allow_skip else "")
    while True:
        raw = input(f"  {question}\n    [{options}] > ").strip().lower()
        if raw in SCALE:
            return SCALE[raw]
        if raw == "s" and allow_skip:
            return None
        if raw == "q" and allow_skip:
            raise KeyboardInterrupt
        print("    not one of those.")


def grade_one(row: dict) -> bool:
    """Ask the three questions. Returns False if the grader wants to stop."""
    try:
        row["faithful"] = ask("Is every claim supported by the sections shown?")
        if row["faithful"] is None:
            return True
        row["relevant"] = ask("Does it answer the question that was asked?")
        if row["outcome"] in ("refusal", "hedged"):
            print("\n    This answer used the refusal token.")
            row["refusal_correct"] = ask("Was declining right, given these sections?")
        note = input("  note (optional) > ").strip()
        if note:
            row["note"] = note
    except (KeyboardInterrupt, EOFError):
        print("\nstopped, progress saved")
        return False
    return True


def summarise(rows: list[dict]) -> None:
    graded = [r for r in rows if r["faithful"] is not None]
    print(f"\n{len(graded)} of {len(rows)} graded")
    if not graded:
        return

    for field in ("faithful", "relevant", "refusal_correct"):
        counts: dict[str, int] = {}
        for r in graded:
            v = r.get(field)
            if v is not None:
                counts[v] = counts.get(v, 0) + 1
        if counts:
            line = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            print(f"  {field:16} {line}")

    print("\n  by stratum")
    for stratum in sorted({r["stratum"] for r in rows}):
        sub = [r for r in graded if r["stratum"] == stratum]
        if sub:
            faithful = sum(1 for r in sub if r["faithful"] == "yes")
            print(f"    {stratum:12} {faithful}/{len(sub)} fully faithful")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--summary", action="store_true", help="counts so far, grade nothing")
    ap.add_argument("--review", action="store_true", help="print what has been graded")
    ap.add_argument("--regrade", metavar="ID", help="grade one answer again")
    args = ap.parse_args()

    rows = load(args.path)

    if args.summary:
        summarise(rows)
        return 0

    if args.review:
        for r in rows:
            if r["faithful"] is not None:
                print(f"{r['id']:8} {r['stratum']:12} faithful={r['faithful']:6} "
                      f"relevant={r['relevant']:6} refusal={r['refusal_correct'] or '-'}")
                if r["note"]:
                    print(f"         note: {r['note']}")
        return 0

    if args.regrade:
        targets = [r for r in rows if r["id"] == args.regrade]
        if not targets:
            raise SystemExit(f"no answer with id {args.regrade}")
    else:
        targets = [r for r in rows if r["faithful"] is None]

    if not targets:
        print("everything is graded.")
        summarise(rows)
        return 0

    total = len(rows)
    for row in targets:
        position = f"{rows.index(row) + 1}/{total}"
        show(row, position)
        if not grade_one(row):
            break
        save(args.path, rows)

    save(args.path, rows)
    summarise(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
