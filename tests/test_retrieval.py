"""Tests for the logic that could be wrong without anything failing loudly.

Metric arithmetic, rank fusion, and the domain filter all produce plausible
numbers when they are subtly broken, which is exactly the kind of bug that
quietly invalidates a results table. Everything here runs without a model, a
vector store, or the corpus.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compare_runs import mcnemar_exact  # noqa: E402
from eval_retrieval import compute_metrics, first_hit_rank  # noqa: E402
from retrievers import (  # noqa: E402
    DomainFilteredRetriever,
    HybridRetriever,
    indexed_text,
    section_number,
    tokenize,
)


class FakeRetriever:
    """Returns a fixed ranking, so fusion and filtering can be tested alone."""

    def __init__(self, ids, citations=None):
        self.ids = ids
        self.citations = citations or {}

    def search(self, query, k=10):
        return [
            {
                "record_id": i,
                "score": 1.0 / rank,
                "payload": {"citation": self.citations.get(i, "BGB § 550"), "text": i},
            }
            for rank, i in enumerate(self.ids[:k], 1)
        ]


def test_first_hit_rank_is_one_indexed():
    assert first_hit_rank(["a", "b"], {"a"}) == 1
    assert first_hit_rank(["a", "b"], {"b"}) == 2
    assert first_hit_rank(["a", "b"], {"z"}) is None


def test_metrics_match_hand_computed_values():
    results = [
        {"type": "d", "gold_ids": ["a"], "retrieved_ids": ["a", "b", "c"]},        # rank 1
        {"type": "d", "gold_ids": ["x"], "retrieved_ids": ["b", "x", "c"]},        # rank 2
        {"type": "m", "gold_ids": ["z"], "retrieved_ids": list("bcdez")},          # rank 5
        {"type": "m", "gold_ids": ["q"], "retrieved_ids": ["b", "c", "d"]},        # miss
    ]
    m = compute_metrics(results, ks=(1, 3, 5))["overall"]
    assert m["hit@1"] == pytest.approx(0.25)
    assert m["hit@3"] == pytest.approx(0.50)
    assert m["hit@5"] == pytest.approx(0.75)
    # (1 + 1/2 + 1/5 + 0) / 4
    assert m["mrr"] == pytest.approx(0.425)


def test_metrics_count_a_question_once_even_with_several_gold_hits():
    """hit@k is per question, not per matching document."""
    results = [{"type": "m", "gold_ids": ["a", "b"], "retrieved_ids": ["a", "b"]}]
    m = compute_metrics(results, ks=(1, 3))["overall"]
    assert m["hit@1"] == 1.0
    assert m["hit@3"] == 1.0
    assert m["n"] == 1


def test_per_type_breakdown_partitions_the_questions():
    results = [
        {"type": "direct", "gold_ids": ["a"], "retrieved_ids": ["a"]},
        {"type": "layperson", "gold_ids": ["b"], "retrieved_ids": ["z"]},
    ]
    m = compute_metrics(results, ks=(1,))
    assert m["by_type"]["direct"]["hit@1"] == 1.0
    assert m["by_type"]["layperson"]["hit@1"] == 0.0
    assert sum(t["n"] for t in m["by_type"].values()) == m["overall"]["n"]


def test_rrf_promotes_a_record_both_retrievers_rank_well():
    """The point of fusion: agreement should beat a single strong opinion."""
    a = FakeRetriever(["x", "shared", "p"])
    b = FakeRetriever(["y", "shared", "q"])
    out = HybridRetriever([a, b], rrf_k=1).search("q", k=1)
    assert out[0]["record_id"] == "shared"


def test_domain_filter_keeps_only_in_range_sections():
    base = FakeRetriever(
        ["out", "in", "also_out"],
        {"out": "BGB § 433", "in": "BGB § 551 Abs. 1", "also_out": "BGB § 1004"},
    )
    out = DomainFilteredRetriever(base).search("q", k=10)
    assert [h["record_id"] for h in out] == ["in"]


def test_domain_filter_escalates_rather_than_returning_short():
    """Regression: a fixed overfetch returned fewer than k when in-scope hits were sparse."""
    ids = [f"r{i}" for i in range(400)]
    # only every 50th record is in range, so a small fetch finds too few
    citations = {i: ("BGB § 551" if n % 50 == 0 else "BGB § 1004") for n, i in enumerate(ids)}
    out = DomainFilteredRetriever(FakeRetriever(ids, citations), overfetch=2).search("q", k=5)
    assert len(out) == 5


def test_section_number_parsing():
    assert section_number("BGB § 573c Abs. 1") == 573
    assert section_number("BGB § 538") == 538
    assert section_number(None) is None
    assert section_number("no section here") is None


def test_tokenizer_keeps_umlauts_inside_words():
    assert tokenize("Kündigungsfrist für Mieter") == ["kündigungsfrist", "für", "mieter"]


def test_indexed_text_matches_what_the_index_was_built_from():
    record = {"text": "body", "context": "Buch 2 > § 1 Titel"}
    assert indexed_text(record, False) == "body"
    assert indexed_text(record, True) == "Buch 2 > § 1 Titel\nbody"
    # a record with no context must not gain a stray newline
    assert indexed_text({"text": "body"}, True) == "body"


def test_mcnemar_uses_only_discordant_pairs():
    assert mcnemar_exact(0, 0) == 1.0          # no disagreement
    assert mcnemar_exact(5, 5) == 1.0          # symmetric disagreement
    assert mcnemar_exact(3, 0) == pytest.approx(0.25)
    assert mcnemar_exact(22, 2) < 0.001        # the BM25 result
