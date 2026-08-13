"""Tests for the generation logic that runs before and after the model call.

Prompt assembly, citation parsing, and the grounding check are all things that
can be wrong while still producing output that reads fine, and the grounding
check in particular is the one signal saying an answer did not invent a §. None
of this needs a model, a key, or a running Ollama.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generator import (  # noqa: E402
    REFUSAL_TOKEN,
    Answer,
    build_context,
    build_prompt,
    classify_response,
    cost_usd,
    generate,
    parse_citations,
    strip_law_prefix,
    strip_think_tags,
    verify_citations,
)


def hit(record_id: str, citation: str, text: str = "Text.", title: str | None = None) -> dict:
    return {
        "record_id": record_id,
        "score": 1.0,
        "payload": {"record_id": record_id, "citation": citation, "text": text, "title": title},
    }


CONTEXT = [
    hit("bgb-551-1", "BGB § 551 Abs. 1", "Höchstens das Dreifache der Monatsmiete.", "Begrenzung"),
    hit("bgb-551-2", "BGB § 551 Abs. 2", "Der Mieter ist zu drei Teilzahlungen berechtigt."),
    hit("bgb-573c-1", "BGB § 573c Abs. 1", "Die Kündigung ist spätestens am dritten Werktag zulässig."),
]


class StubBackend:
    """Returns a canned reply, so `generate` can be tested without a model."""

    name = "stub"
    model = "stub-model"

    def __init__(self, reply: str):
        self.reply = reply
        self.seen: tuple[str, str] | None = None

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        self.seen = (system, user)
        return self.reply, 400, 60


def test_context_gives_the_model_the_exact_citation_string():
    """The model should copy a citation, not reconstruct one from a record id."""
    text = build_context(CONTEXT)
    assert "§ 551 Abs. 1" in text
    assert "§ 573c Abs. 1" in text
    # the law prefix is the same on all 4,921 records and buys nothing per block
    assert "BGB §" not in text
    assert "bgb-551-1" not in text


def test_context_numbering_and_titles():
    text = build_context(CONTEXT)
    assert text.startswith("[1] § 551 Abs. 1 — Begrenzung")
    assert "[3] § 573c Abs. 1" in text
    # a record without a title must not trail an empty dash
    assert "[2] § 551 Abs. 2\n" in text


def test_prompt_carries_the_question_and_every_section():
    prompt = build_prompt("Wie hoch darf die Kaution sein?", CONTEXT)
    assert "Wie hoch darf die Kaution sein?" in prompt
    for h in CONTEXT:
        assert h["payload"]["text"] in prompt


def test_strip_law_prefix():
    assert strip_law_prefix("BGB § 551 Abs. 1") == "§ 551 Abs. 1"
    assert strip_law_prefix("§ 551") == "§ 551"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Nach [§ 551 Abs. 1] gilt...", [("551", "1")]),
        ("Siehe § 573c Abs. 1.", [("573c", "1")]),          # letter suffix survives
        ("§ 556 Absatz 3 Satz 2", [("556", "3")]),           # spelled out, Satz ignored
        ("Vgl. §§ 535, 536", [("535", None), ("536", None)]),        # a list needs both
        ("§§ 535 und 536 Abs. 1", [("535", None), ("536", "1")]),
        # a single § must not read the following number as a second section
        ("§ 556 Abs. 3, 12 Monate", [("556", "3")]),
        ("[§ 551 Abs. 1] und nochmals [§ 551 Abs. 1]", [("551", "1")]),  # deduplicated
        ("Keine Fundstelle hier.", []),
    ],
)
def test_citation_parsing(text, expected):
    assert parse_citations(text) == expected


def test_section_and_absatz_are_distinct():
    """§ 573 and § 573c are different sections, and must not collapse."""
    assert parse_citations("§ 573 und § 573c") == [("573", None), ("573c", None)]


def test_citation_check_separates_the_outcomes():
    answer = (
        "Die Kaution beträgt höchstens drei Monatsmieten [§ 551 Abs. 1]. "
        "Kündigung siehe [§ 573c Abs. 3]. "
        "Zur Mietminderung [§ 536 Abs. 1]."
    )
    check = verify_citations(answer, CONTEXT)

    assert check.exact == [("551", "1")]
    assert check.section == [("573c", "3")]      # right §, an Absatz that was not shown
    assert check.unsupported == [("536", "1")]   # named nowhere
    assert check.total == 3
    assert not check.grounded


def test_a_section_named_inside_the_shown_text_is_a_cross_reference():
    """Regression from the first 150-question run.

    § 559 Abs. 1 reads "Modernisierungsmaßnahmen im Sinne des § 555b", so an
    answer citing § 555b followed a pointer it was shown rather than inventing
    one. Scoring those as hallucinations overstated the rate eightfold.
    """
    context = [hit("bgb-559-1", "BGB § 559 Abs. 1",
                   "Hat der Vermieter Modernisierungsmaßnahmen im Sinne des § 555b "
                   "durchgeführt, so kann er die Miete erhöhen.")]

    check = verify_citations("Die Erhöhung setzt eine Maßnahme nach [§ 555b] voraus.", context)
    assert check.cross_referenced == [("555b", None)]
    assert check.unsupported == []
    # traceable, so not an invention, but its text was never read
    assert check.grounded
    assert not check.fully_verifiable

    # a section named nowhere is still an invention
    invented = verify_citations("Siehe [§ 902 Abs. 1].", context)
    assert invented.unsupported == [("902", "1")]
    assert not invented.grounded


def test_grounded_needs_a_citation_and_no_invented_section():
    grounded = verify_citations("Höchstens drei Monatsmieten [§ 551 Abs. 1].", CONTEXT)
    assert grounded.grounded

    invented = verify_citations("Das steht in [§ 902 Abs. 4].", CONTEXT)
    assert not invented.grounded
    assert invented.unsupported == [("902", "4")]

    # an answer with no citations at all is not grounded either, since there is
    # nothing tying it to the retrieved text
    assert not verify_citations("Ja, das ist erlaubt.", CONTEXT).grounded


def test_citation_check_serialises_back_to_readable_references():
    d = verify_citations("[§ 551 Abs. 1] und [§ 902]", CONTEXT).as_dict()
    assert d["exact"] == ["§ 551 Abs. 1"]
    assert d["unsupported"] == ["§ 902"]
    assert d["grounded"] is False


def test_think_tags_are_stripped_before_citations_are_read():
    """Inline reasoning must not be mined for citations.

    Ollama returns reasoning in its own field, so this is a guard against a
    backend that inlines it instead. Without the strip, a model reasoning aloud
    about § 536 would have that counted as a citation it never made.
    """
    reply = "<think>Vielleicht § 536 Abs. 1?</think>Höchstens drei Monatsmieten [§ 551 Abs. 1]."
    assert strip_think_tags(reply) == "Höchstens drei Monatsmieten [§ 551 Abs. 1]."
    assert parse_citations(strip_think_tags(reply)) == [("551", "1")]
    # an answer with no reasoning is returned unchanged apart from trimming
    assert strip_think_tags("  Ein Satz.  ") == "Ein Satz."


def test_an_empty_reply_is_raised_as_a_catchable_error():
    """Regression: the guard used SystemExit and killed a five-hour run.

    SystemExit derives from BaseException, so the per-question `except
    Exception` in generate_answers.py never saw it. One model hiccup at
    question 12 ended the job instead of being recorded and skipped.
    """
    import io
    import json as jsonlib
    import urllib.request

    import generator

    def fake_urlopen(request, timeout=None):
        body = jsonlib.dumps({
            "message": {"content": "   ", "thinking": "lots of reasoning"},
            "prompt_eval_count": 1100,
            "eval_count": 981,
        })
        return io.BytesIO(body.encode())

    original = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        with pytest.raises(RuntimeError) as caught:
            generator.OllamaBackend().complete("system", "user")
    finally:
        urllib.request.urlopen = original

    assert isinstance(caught.value, Exception)      # catchable, unlike SystemExit
    assert not isinstance(caught.value, SystemExit)
    assert "981 output tokens" in str(caught.value)


def test_a_bare_token_is_a_refusal_even_with_a_courtesy_sentence():
    empty = verify_citations("", CONTEXT)
    assert classify_response(REFUSAL_TOKEN, empty) == "refusal"
    assert classify_response(f"Leider {REFUSAL_TOKEN}", empty) == "refusal"


def test_an_answer_that_cites_and_then_hedges_is_not_a_refusal():
    """Regression from the first 150-question run.

    ev-139 answered the question with two grounded citations and closed by
    noting the text did not settle one detail, using the token. Counting that
    as a refusal credits the system for declining when it actually answered.
    """
    reply = (
        "Bei einer aufwendigen Maßnahme müssen Sie nur zahlen, wenn drei Angebote "
        "eingeholt wurden [§ 551 Abs. 1]. Eine abschließende Beantwortung ist "
        f"{REFUSAL_TOKEN}."
    )
    check = verify_citations(reply, CONTEXT)
    assert check.total == 1
    assert classify_response(reply, check) == "hedged"


def test_a_plain_answer_is_neither():
    reply = "Die Kaution beträgt höchstens drei Monatsmieten [§ 551 Abs. 1]."
    assert classify_response(reply, verify_citations(reply, CONTEXT)) == "answer"


def test_a_reference_to_another_statute_is_not_a_bgb_citation():
    """Also from the first run.

    The BGB cross-references the Telekommunikationsgesetz, and the model
    reproduces that when quoting § 555b. Reading it as a BGB section marked two
    faithful answers as ungrounded.
    """
    reply = (
        "Dazu zählt der Anschluss an ein Glasfasernetz gemäß § 3 Nummer 33 des "
        "Telekommunikationsgesetzes [§ 551 Abs. 1]."
    )
    assert parse_citations(reply) == [("551", "1")]
    assert verify_citations(reply, CONTEXT).grounded

    # the same shape without the qualifier is still a BGB citation
    assert parse_citations("gemäß § 3 Nummer 33 gilt") == [("3", None)]


@pytest.mark.parametrize(
    "text, expected, why",
    [
        # Found in the phase 4 audit. Skipping a foreign reference used to break
        # out of the whole list, so a real citation after it vanished. A
        # citation the checker never sees cannot be flagged as invented, which
        # is the direction that hides a mistake.
        ("§§ 3 des Telekommunikationsgesetzes, 551 Abs. 1", [("551", "1")],
         "a list survives a foreign entry"),
        # Also found in the audit: another law named by trailing abbreviation
        # rather than by "des ...gesetzes" was read as one of ours.
        ("Art. 229 § 3 EGBGB", [], "trailing abbreviation of another law"),
        ("§ 72 TKG", [], "abbreviation without a des-phrase"),
        # The exception that makes the rule usable.
        ("§ 551 Abs. 1 BGB", [("551", "1")], "our own law is not skipped"),
        ("§ 551 Abs. 1 Satz 2", [("551", "1")], "Satz is not a law abbreviation"),
        ("§ 551. Der Vermieter zahlt", [("551", None)], "a sentence boundary is not one either"),
    ],
)
def test_references_to_other_statutes(text, expected, why):
    assert parse_citations(text) == expected, why


def test_cost_is_computed_from_published_rates():
    # Haiku 4.5 at $1 in / $5 out per million tokens
    assert cost_usd("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.00)
    assert cost_usd("claude-haiku-4-5", 0, 1_000_000) == pytest.approx(5.00)
    assert cost_usd("claude-haiku-4-5", 1000, 200) == pytest.approx(0.002)
    # a model running on your own hardware has no per-token price
    assert cost_usd("qwen3.5:9b", 5000, 900) == 0.0


def test_generate_wires_prompt_verification_and_accounting_together():
    backend = StubBackend("Höchstens drei Monatsmieten [§ 551 Abs. 1].")
    answer = generate("Wie hoch darf die Kaution sein?", CONTEXT, backend)

    assert isinstance(answer, Answer)
    assert "Wie hoch darf die Kaution sein?" in backend.seen[1]
    assert answer.citations.grounded
    assert not answer.refused
    assert answer.input_tokens == 400 and answer.output_tokens == 60
    assert answer.retrieved == ["bgb-551-1", "bgb-551-2", "bgb-573c-1"]
    assert answer.latency_ms >= 0


def test_generate_marks_a_refusal_without_counting_it_as_a_hallucination():
    answer = generate("Was kostet ein Flug nach Rom?", CONTEXT, StubBackend(REFUSAL_TOKEN))
    assert answer.refused
    assert answer.citations.total == 0
    assert answer.citations.unsupported == []
