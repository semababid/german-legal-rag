"""Score answers with a model, and check that model against the hand grades.

The judge is only worth having if it agrees with a person, and the only way to
know is to measure it on answers a person already graded. That is what the 30 in
data/eval/calibration.jsonl are for.

    python src/judge.py --calibration          # judge the 30 hand-graded ones
    python src/judge.py --agreement            # compare against the hand grades
    python src/judge.py --run local-qwen35-9b  # judge a full answer run

Two rules this file exists to respect.

The judge sees exactly what the human grader saw: the question, the sections the
answering model was shown, its answer, and the reference answer. It is not shown
the automatic citation check, because that would let it copy a verdict rather
than reach one, and the whole point is to find out whether it can reach one.

The prompt is not tuned against the calibration set. Rewriting it until
agreement improves would fit the judge to those 30 answers and produce a number
that says nothing about the other 120. If the prompt does get revised, the
agreement figure afterwards is optimistic and has to be labelled as such.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = REPO_ROOT / "data" / "eval" / "calibration.jsonl"
JUDGE_DIR = REPO_ROOT / "results" / "judge"

SCALE = ("yes", "partly", "no")

# Deliberately the same definitions the human grader worked from, in the same
# order, with the same three-point scale. A judge asked a different question
# than the person cannot be compared to the person.
# Version 2. Version 1 is kept in git history and its scores stay published.
# It offered "partly" without ever saying what earns it, and returned yes 23
# times, no 7 times and partly not once across 30 answers. A judge using two
# thirds of a three-point scale cannot match a grader who puts half the answers
# in the middle. That defect is visible in the judge's own output distribution,
# with no reference to the human grades, which is why fixing it is not fitting
# the prompt to them. The agreement measured afterwards is still optimistic, for
# the reason given in the module docstring, and is labelled that way.

JUDGE_SYSTEM = """Du bewertest die Antwort eines anderen Modells auf eine Frage \
zum deutschen Mietrecht. Dir werden die Vorschriften vorgelegt, die dieses Modell \
gesehen hat.

Prüfe jede einzelne Aussage der Antwort gegen die vorgelegten Vorschriften. \
Die meisten Antworten enthalten mehrere Aussagen, und dass einige gedeckt sind \
und andere nicht, ist der Normalfall und nicht die Ausnahme.

faithful: Wird jede Aussage durch die vorgelegten Vorschriften gedeckt?
  "yes"    jede Aussage steht so im vorgelegten Text
  "partly" mindestens eine Aussage ist gedeckt und mindestens eine nicht
  "no"     im Wesentlichen nichts ist gedeckt
Entscheidend ist allein der vorgelegte Text, nicht das deutsche Recht im \
Allgemeinen. Eine Aussage, die juristisch zutrifft, aber in den vorgelegten \
Vorschriften nicht steht, ist nicht gedeckt und macht die Antwort "partly".

relevant: Beantwortet die Antwort die gestellte Frage?
  "yes"    die gestellte Frage wird beantwortet
  "partly" richtiges Thema, aber die gestellte Frage bleibt offen
  "no"     eine andere Frage wird beantwortet, oder gar keine

refusal_correct: Nur wenn die Antwort das Wort NICHT_BEANTWORTBAR enthält. \
Sonst null. Enthält sie es, dann ist dieses Feld Pflicht und darf nicht null \
sein. War die Verweigerung angesichts der vorgelegten Vorschriften richtig?
  "yes"    die Vorschriften beantworten die Frage tatsächlich nicht
  "partly" sie beantworten sie teilweise
  "no"     die Antwort stand in den Vorschriften, das Modell hat trotzdem \
verweigert

Zähle zuerst still die Aussagen der Antwort und prüfe jede einzeln, bevor du \
dich für "yes" entscheidest.

Antworte ausschließlich mit einem JSON-Objekt, ohne Erklärung davor oder danach:
{"faithful": "...", "relevant": "...", "refusal_correct": "..." oder null, \
"reason": "ein kurzer Satz"}"""


def build_judge_prompt(row: dict) -> str:
    sections = "\n\n".join(
        f"{s['citation']}" + (f" — {s['title']}" if s.get("title") else "") + f"\n{s['text']}"
        for s in row["sections_shown"]
    )
    parts = [
        "VORGELEGTE VORSCHRIFTEN:\n\n" + sections,
        "FRAGE:\n" + row["question"],
        "ANTWORT DES MODELLS:\n" + row["answer"],
    ]
    if row.get("reference_answer"):
        # The human grader had this, so the judge does too. Withholding it would
        # make the two graders answer different questions.
        parts.append("REFERENZANTWORT (zur Orientierung):\n" + row["reference_answer"])
    return "\n\n".join(parts)


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str) -> dict | None:
    """Pull the JSON verdict out of a reply, tolerating chat around it.

    Returns None when the reply cannot be read as a verdict. That is recorded
    rather than guessed: a judge that failed to answer in the required shape has
    not graded the answer, and scoring it as agreement or disagreement would
    invent data.
    """
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    out = {}
    for field in ("faithful", "relevant", "refusal_correct"):
        value = raw.get(field)
        value = value.strip().lower() if isinstance(value, str) else value
        out[field] = value if value in SCALE else None
    out["reason"] = str(raw.get("reason", ""))[:300]
    return out


def judge_rows(rows: list[dict], backend) -> list[dict]:
    from generator import Answer  # noqa: F401  (keeps the import surface obvious)

    out = []
    for i, row in enumerate(rows, 1):
        prompt = build_judge_prompt(row)
        text, tin, tout = backend.complete(JUDGE_SYSTEM, prompt)
        verdict = parse_verdict(text)

        out.append({
            "id": row["id"],
            "stratum": row["stratum"],
            "outcome": row["outcome"],
            "verdict": verdict,
            "raw": None if verdict else text[:400],
            "input_tokens": tin,
            "output_tokens": tout,
        })
        status = "unparseable" if verdict is None else " ".join(
            f"{k}={verdict[k]}" for k in ("faithful", "relevant") )
        print(f"{i:3}/{len(rows)}  {row['id']:8} {status}")
    return out


# --- agreement -------------------------------------------------------------

def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Agreement above what two graders would reach by guessing.

    Raw percentage agreement flatters a judge on a skewed set: if 80% of answers
    are "partly", always saying "partly" scores 80% while knowing nothing. Kappa
    subtracts the agreement chance alone would produce. 0 means no better than
    guessing, 1 means perfect.
    """
    n = len(pairs)
    if not n:
        return 0.0
    observed = sum(1 for a, b in pairs if a == b) / n

    labels = {a for a, _ in pairs} | {b for _, b in pairs}
    expected = sum(
        (sum(1 for a, _ in pairs if a == label) / n)
        * (sum(1 for _, b in pairs if b == label) / n)
        for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def adjacent(a: str, b: str) -> bool:
    """Off by one on the yes/partly/no scale.

    The three grades are ordered, so yes against partly is a smaller
    disagreement than yes against no, and reporting only exact matches treats
    them as the same mistake.
    """
    order = {"no": 0, "partly": 1, "yes": 2}
    return abs(order[a] - order[b]) <= 1


# Collapses of the three-point scale, reported alongside it.
#
# The grader said afterwards that "no" felt clear every time but the line
# between "yes" and "partly" often did not. That is worth taking seriously: if
# most of the judge's disagreement sat on a boundary the human was unsure of,
# the judge would be getting blamed for the scale rather than for its
# judgment. Merging the two grades either way answers that directly, and the
# check costs nothing, so it runs on every comparison rather than being
# remembered as a one-off.
COLLAPSES = {
    "is it broken (yes+partly vs no)": lambda v: "ok" if v in ("yes", "partly") else "no",
    "is it perfect (yes vs partly+no)": lambda v: "yes" if v == "yes" else "not",
}


def collapsed_scores(pairs: list[tuple[str, str]]) -> dict:
    out = {}
    for name, fn in COLLAPSES.items():
        merged = [(fn(a), fn(b)) for a, b in pairs]
        n = len(merged)
        majority = max({a for a, _ in merged},
                       key=lambda s: sum(1 for a, _ in merged if a == s))
        out[name] = {
            "exact": round(sum(1 for a, b in merged if a == b) / n, 3),
            "kappa": round(cohens_kappa(merged), 3),
            "majority_baseline": round(sum(1 for a, _ in merged if a == majority) / n, 3),
        }
    return out


def agreement(hand: list[dict], judged: list[dict]) -> dict:
    by_id = {j["id"]: j for j in judged}
    report: dict = {"unparseable": [j["id"] for j in judged if j["verdict"] is None]}

    for field in ("faithful", "relevant", "refusal_correct"):
        pairs = []
        for row in hand:
            j = by_id.get(row["id"])
            if not j or not j["verdict"]:
                continue
            human, machine = row.get(field), j["verdict"].get(field)
            if human in SCALE and machine in SCALE:
                pairs.append((human, machine))

        if not pairs:
            continue

        n = len(pairs)
        majority = max(SCALE, key=lambda s: sum(1 for h, _ in pairs if h == s))
        report[field] = {
            "n": n,
            "exact": round(sum(1 for a, b in pairs if a == b) / n, 3),
            "within_one": round(sum(1 for a, b in pairs if adjacent(a, b)) / n, 3),
            "kappa": round(cohens_kappa(pairs), 3),
            # what a judge that always guessed the commonest human grade scores
            "majority_baseline": round(sum(1 for h, _ in pairs if h == majority) / n, 3),
            "confusion": {
                f"human={h},judge={m}": sum(1 for a, b in pairs if a == h and b == m)
                for h in SCALE for m in SCALE
                if sum(1 for a, b in pairs if a == h and b == m)
            },
            "collapsed": collapsed_scores(pairs),
        }

    report["disagreements"] = [
        {
            "id": row["id"],
            "stratum": row["stratum"],
            "human": {f: row[f] for f in ("faithful", "relevant", "refusal_correct")},
            "judge": {f: by_id[row["id"]]["verdict"].get(f)
                      for f in ("faithful", "relevant", "refusal_correct")},
            "judge_reason": by_id[row["id"]]["verdict"].get("reason", ""),
        }
        for row in hand
        if row["id"] in by_id and by_id[row["id"]]["verdict"]
        and any(row[f] != by_id[row["id"]]["verdict"].get(f)
                for f in ("faithful", "relevant"))
    ]
    return report


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def print_agreement(report: dict) -> None:
    print("\nagreement with the hand grades")
    print(f"  {'':18}{'n':>4}{'exact':>8}{'±1':>8}{'kappa':>8}{'baseline':>10}")
    for field in ("faithful", "relevant", "refusal_correct"):
        r = report.get(field)
        if r:
            print(f"  {field:18}{r['n']:>4}{r['exact']:>8.0%}{r['within_one']:>8.0%}"
                  f"{r['kappa']:>8.2f}{r['majority_baseline']:>10.0%}")
    faithful = report.get("faithful")
    if faithful and faithful.get("collapsed"):
        print("\n  faithfulness with the scale collapsed, in case the "
              "disagreement is only about where\n  the middle grade starts")
        for name, r in faithful["collapsed"].items():
            print(f"    {name:36}exact {r['exact']:.0%}   kappa {r['kappa']:+.2f}"
                  f"   baseline {r['majority_baseline']:.0%}")

    if report["unparseable"]:
        print(f"\n  unreadable verdicts: {len(report['unparseable'])} "
              f"({', '.join(report['unparseable'][:5])})")
    print(f"\n  {len(report['disagreements'])} answers where judge and human differ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", action="store_true", help="judge the hand-graded 30")
    ap.add_argument("--agreement", action="store_true", help="compare a saved judge run to the hand grades")
    ap.add_argument("--backend", choices=("ollama", "anthropic"), default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--think", action="store_true", help="ollama only: let the judge reason first")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    label = args.label or f"judge-{args.backend}" + ("-think" if args.think else "")
    out_path = JUDGE_DIR / f"{label}.json"

    if args.agreement:
        if not out_path.exists():
            raise SystemExit(f"No judge run at {out_path}")
        judged = json.load(out_path.open(encoding="utf-8"))["judged"]
        report = agreement(load_jsonl(CALIBRATION), judged)
        print_agreement(report)
        with out_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        data["agreement"] = report
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        return 0

    if not args.calibration:
        raise SystemExit("give --calibration to judge the hand-graded set, or --agreement to score a run")

    from generator import build_backend

    rows = load_jsonl(CALIBRATION)
    backend = build_backend(args.backend, args.model, think=args.think)
    judged = judge_rows(rows, backend)

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"label": label, "backend": args.backend, "model": backend.model,
                   "think": args.think, "judged": judged}, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")
    print("Now compare: python src/judge.py --agreement" +
          (f" --label {label}" if args.label else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
