"""Tests for the judge's parsing and agreement arithmetic.

Kappa in particular is worth testing against hand-computed values. It is the
number that decides whether the judge is trusted on the other 120 answers, and a
sign error or a wrong denominator would produce a figure that looks reasonable
and is wrong.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge import (  # noqa: E402
    adjacent,
    agreement,
    build_judge_prompt,
    cohens_kappa,
    parse_verdict,
)


def test_verdict_parses_out_of_a_chatty_reply():
    reply = 'Hier meine Bewertung:\n{"faithful": "partly", "relevant": "yes", ' \
            '"refusal_correct": null, "reason": "teilweise gedeckt"}\nHoffe das hilft.'
    v = parse_verdict(reply)
    assert v["faithful"] == "partly"
    assert v["relevant"] == "yes"
    assert v["refusal_correct"] is None
    assert "gedeckt" in v["reason"]


def test_verdict_normalises_case_and_whitespace():
    v = parse_verdict('{"faithful": " YES ", "relevant": "No"}')
    assert v["faithful"] == "yes"
    assert v["relevant"] == "no"


def test_a_value_off_the_scale_becomes_none_rather_than_a_guess():
    v = parse_verdict('{"faithful": "mostly", "relevant": "yes"}')
    assert v["faithful"] is None
    assert v["relevant"] == "yes"


def test_unreadable_replies_are_reported_not_invented():
    assert parse_verdict("Das kann ich nicht bewerten.") is None
    assert parse_verdict('{"faithful": broken') is None


def test_prompt_shows_the_judge_what_the_human_saw_and_nothing_more():
    row = {
        "question": "Wie hoch darf die Kaution sein?",
        "answer": "Drei Monatsmieten [§ 551 Abs. 1].",
        "reference_answer": "Höchstens das Dreifache der Monatsmiete.",
        "sections_shown": [
            {"citation": "BGB § 551 Abs. 1", "title": "Begrenzung", "text": "Hoechstens das Dreifache."}
        ],
        # present on the record, and deliberately not in the prompt
        "citation_check": {"exact": ["§ 551 Abs. 1"], "unsupported": []},
        "stratum": "clean",
    }
    prompt = build_judge_prompt(row)
    assert "Wie hoch darf die Kaution sein?" in prompt
    assert "Hoechstens das Dreifache." in prompt
    assert "Höchstens das Dreifache der Monatsmiete." in prompt
    # the automatic check must not leak, or the judge copies a verdict
    assert "citation_check" not in prompt
    assert "unsupported" not in prompt
    assert "clean" not in prompt


def test_kappa_is_zero_when_agreement_is_only_what_chance_gives():
    # both graders say yes 50% of the time, independently; they happen to agree
    # on half, which is exactly the chance rate
    pairs = [("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no")]
    assert cohens_kappa(pairs) == pytest.approx(0.0)


def test_kappa_is_one_on_perfect_agreement():
    pairs = [("yes", "yes"), ("no", "no"), ("partly", "partly")]
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_kappa_punishes_a_judge_that_always_says_the_common_answer():
    """The failure raw agreement cannot see.

    Nine of ten answers are "partly" and the judge says "partly" every time. It
    scores 90% agreement while having learned nothing, and kappa says so.
    """
    pairs = [("partly", "partly")] * 9 + [("no", "partly")]
    assert sum(1 for a, b in pairs if a == b) / len(pairs) == pytest.approx(0.9)
    assert cohens_kappa(pairs) == pytest.approx(0.0)


def test_kappa_is_negative_when_the_judge_is_worse_than_guessing():
    pairs = [("yes", "no"), ("no", "yes"), ("yes", "no"), ("no", "yes")]
    assert cohens_kappa(pairs) < 0


def test_adjacent_treats_the_scale_as_ordered():
    assert adjacent("yes", "partly")
    assert adjacent("partly", "no")
    assert adjacent("yes", "yes")
    assert not adjacent("yes", "no")


def _hand(id_, faithful, relevant, refusal=None, stratum="clean"):
    return {"id": id_, "stratum": stratum, "faithful": faithful,
            "relevant": relevant, "refusal_correct": refusal}


def _judged(id_, faithful, relevant, refusal=None):
    return {"id": id_, "stratum": "clean", "outcome": "answer",
            "verdict": {"faithful": faithful, "relevant": relevant,
                        "refusal_correct": refusal, "reason": ""}}


def test_agreement_counts_only_pairs_both_graders_scored():
    hand = [_hand("a", "yes", "yes"), _hand("b", "no", "no")]
    judged = [_judged("a", "yes", "yes"), {"id": "b", "stratum": "clean",
                                           "outcome": "answer", "verdict": None}]
    report = agreement(hand, judged)
    # b produced no readable verdict, so it is reported, not scored either way
    assert report["unparseable"] == ["b"]
    assert report["faithful"]["n"] == 1
    assert report["faithful"]["exact"] == 1.0


def test_agreement_lists_the_disagreements_for_review():
    hand = [_hand("a", "yes", "yes"), _hand("b", "no", "yes")]
    judged = [_judged("a", "yes", "yes"), _judged("b", "partly", "yes")]
    report = agreement(hand, judged)
    ids = [d["id"] for d in report["disagreements"]]
    assert ids == ["b"]
    assert report["disagreements"][0]["human"]["faithful"] == "no"
    assert report["disagreements"][0]["judge"]["faithful"] == "partly"


def test_collapsing_the_scale_rescues_a_boundary_only_disagreement():
    """The check that separates grader uncertainty from judge error.

    Here the judge is right about every answer being imperfect and only ever
    differs on yes against partly. Merging that boundary should show perfect
    agreement, which is what tells you the judge understood the answers and the
    scale was the problem.
    """
    hand = [_hand(str(i), "partly", "yes") for i in range(5)]
    judged = [_judged(str(i), "yes", "yes") for i in range(5)]
    collapsed = agreement(hand, judged)["faithful"]["collapsed"]
    assert collapsed["is it broken (yes+partly vs no)"]["exact"] == 1.0
    # the other collapse keeps the disagreement, since it is exactly that line
    assert collapsed["is it perfect (yes vs partly+no)"]["exact"] == 0.0


def test_collapsing_does_not_rescue_a_judge_that_is_simply_wrong():
    """The real case: disagreement survives the merge, so the scale is not it."""
    hand = [_hand("a", "yes", "yes"), _hand("b", "no", "yes"),
            _hand("c", "no", "yes"), _hand("d", "yes", "yes")]
    judged = [_judged("a", "no", "yes"), _judged("b", "yes", "yes"),
              _judged("c", "yes", "yes"), _judged("d", "no", "yes")]
    collapsed = agreement(hand, judged)["faithful"]["collapsed"]
    for scores in collapsed.values():
        assert scores["exact"] == 0.0


def test_agreement_reports_the_baseline_a_lazy_judge_would_hit():
    hand = [_hand(str(i), "partly", "yes") for i in range(8)] + [_hand("x", "no", "yes")]
    judged = [_judged(str(i), "partly", "yes") for i in range(8)] + [_judged("x", "partly", "yes")]
    report = agreement(hand, judged)
    assert report["faithful"]["majority_baseline"] == pytest.approx(8 / 9, abs=0.001)
    assert report["faithful"]["exact"] == pytest.approx(8 / 9, abs=0.001)
    # matching the baseline exactly means the judge added nothing
    assert report["faithful"]["kappa"] == pytest.approx(0.0)
