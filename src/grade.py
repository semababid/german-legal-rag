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


def show(row: dict, position: str, show_check: bool = False) -> None:
    """Put one answer on screen for grading.

    The stratum label and the automatic citation check are hidden. Both are
    this project's own verdict on the answer, and showing them above the grading
    questions invites the grader to agree with a checker that has been wrong
    five times so far. The first 30 were graded with both visible and an
    instruction to ignore them, which is a weaker guarantee than not printing
    them; `--show-check` reproduces that older view.
    """
    print("\n" + "=" * 92)
    header = f"{position}   {row['id']}   {row['type']}"
    if show_check:
        header += f"   stratum: {row['stratum']}"
    print(header)
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

    if show_check:
        c = row["citation_check"]
        print("\nAUTOMATISCHE PRÜFUNG (zur Information, entscheide selbst)")
        print(f"    exact: {c['exact'] or '-'}   nur Paragraf: {c['section_only'] or '-'}")
        print(f"    nur erwähnt: {c.get('cross_referenced') or '-'}   "
              f"nirgends: {c['unsupported'] or '-'}")
        print(f"    outcome: {row['outcome']}")

    print("\n" + "-" * 92)


def ask(question: str, step: int, steps: int) -> str | None:
    """Put one question and insist on a real answer.

    Every branch here exists because the first version had none of them: it
    ended the prompt without a newline, so the next question printed on the same
    line as the reply and the whole screen read as one blob, and pressing return
    was not a recognised choice, so the most natural key on the keyboard
    silently re-asked the same question.
    """
    prompt = (
        f"  ({step} of {steps})  {question}\n"
        f"      y = yes    p = partly    n = no\n"
        f"      s = skip this one    q = save and quit\n"
        f"      > "
    )
    while True:
        raw = input(prompt).strip().lower()
        if raw in SCALE:
            print(f"      recorded: {SCALE[raw]}\n")
            return SCALE[raw]
        if raw == "s":
            print("      skipped\n")
            return None
        if raw == "q":
            raise KeyboardInterrupt
        if raw == "":
            print("      nothing entered. Press y, p or n, then return.\n")
        else:
            print(f"      '{raw}' is not one of y, p, n, s, q.\n")


def grade_one(row: dict) -> bool:
    """Ask the questions for one answer. Returns False if the grader stops.

    Two questions normally, three when the answer used the refusal token. The
    count is announced up front so a third prompt does not arrive as a surprise.
    """
    declined = row["outcome"] in ("refusal", "hedged")
    steps = 3 if declined else 2

    try:
        row["faithful"] = ask(
            "Is every claim supported by the sections shown above?", 1, steps)
        if row["faithful"] is None:
            return True

        row["relevant"] = ask(
            "Does it answer the question that was asked?", 2, steps)

        if declined:
            print("  This answer used the refusal token.")
            row["refusal_correct"] = ask(
                "Was declining right, given these sections?", 3, steps)

        note = input("  note, or return to skip\n      > ").strip()
        if note:
            row["note"] = note
        print()
    except (KeyboardInterrupt, EOFError):
        print("\n\nstopped. Everything graded so far is saved.")
        print("Pick up where you left off with: python src/grade.py")
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
    ap.add_argument("--show-check", action="store_true",
                    help="also show this project's own citation verdict, as the first 30 were graded")
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
        show(row, position, args.show_check)
        if not grade_one(row):
            break
        save(args.path, rows)

    save(args.path, rows)
    summarise(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
