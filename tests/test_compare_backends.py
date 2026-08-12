"""Tests for the two-run comparison.

The arithmetic here decides a stack question, so it is worth checking that the
numbers mean what the column headings claim. Everything runs without a model or
an API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compare_backends import disagreements, measure, percentile  # noqa: E402


def answer(qid, *, outcome="answer", invented=(), crossref=(), exact=("§ 551 Abs. 1",),
           latency=1000.0, out_tokens=100, cost=0.0, qtype="direct"):
    return {
        "id": qid,
        "type": qtype,
        "outcome": outcome,
        "latency_ms": latency,
        "output_tokens": out_tokens,
        "cost_usd": cost,
        "citations": {
            "exact": list(exact),
            "section_only": [],
            "cross_referenced": list(crossref),
            "unsupported": list(invented),
            "grounded": not invented,
            "fully_verifiable": not invented and not crossref,
        },
    }


def run(label, model, answers):
    return {"label": label, "model": model, "answers": answers}


def test_percentile_picks_a_real_observation():
    assert percentile([5.0, 1.0, 3.0], 0.5) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert percentile([], 0.5) == 0.0


def test_measure_counts_only_the_shared_questions():
    """A run that skipped the hard questions must not look better for it."""
    r = run("x", "m", [answer("a"), answer("b", invented=["§ 902"]), answer("c")])
    m = measure(r, {"a", "b"})
    assert m["n"] == 2
    assert m["invented"] == 1


def test_refusals_are_excluded_from_citation_counts():
    """A refusal cites nothing, and counting it as an answer citing nothing
    would punish the behaviour the prompt asks for."""
    r = run("x", "m", [answer("a", outcome="refusal", exact=()), answer("b")])
    m = measure(r, {"a", "b"})
    assert m["refused"] == 1
    assert m["citing_nothing"] == 0
    assert m["n"] == 2


def test_cost_per_1k_scales_from_the_sample():
    r = run("x", "m", [answer(str(i), cost=0.002) for i in range(10)])
    m = measure(r, {str(i) for i in range(10)})
    assert m["cost_usd"] == 0.02
    assert m["cost_per_1k_usd"] == 2.0


def test_a_free_local_model_reports_zero_not_missing():
    r = run("local", "qwen3.5:9b", [answer("a", cost=0.0)])
    assert measure(r, {"a"})["cost_per_1k_usd"] == 0.0


def test_disagreements_flag_only_the_questions_worth_reading():
    a = run("A", "m1", [answer("q1"), answer("q2", outcome="refusal"),
                        answer("q3", invented=["§ 902"])])
    b = run("B", "m2", [answer("q1"), answer("q2"), answer("q3")])
    diffs = disagreements(a, b, {"q1", "q2", "q3"})
    ids = {d["id"] for d in diffs}

    assert "q1" not in ids                      # both behaved the same
    assert "q2" in ids                          # one refused, one answered
    assert "q3" in ids                          # one invented a citation
    q3 = next(d for d in diffs if d["id"] == "q3")
    assert "only A invented a citation" in q3["why"]
