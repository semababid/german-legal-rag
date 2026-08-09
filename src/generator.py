"""Turn retrieved sections into a cited German answer.

The generation half of the system is thin on purpose. Everything that decides
answer quality happened upstream in retrieval; what is left is to constrain the
model to the sections it was handed, make it cite them, and check afterwards
that the citations it produced actually exist.

Three pieces, each usable on its own:

    build_prompt      question plus numbered sections, in German
    verify_citations  every § the answer cites, checked against what was retrieved
    generate          one call to a backend, with tokens, latency and cost

Citation verification is the cheap faithfulness signal. It cannot tell whether a
claim is supported by the section it points at, which is what the phase 4.3
judge is for, but it catches the failure that matters most in a legal setting:
an answer citing a § that was never in front of the model. That check is
deterministic, costs nothing, and does not need a judge to be calibrated first.

Backends sit behind one `.complete(system, user)` call so a run against a local
model and a run against the API differ by a flag.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# Answering a question the retrieved sections do not cover is the single worst
# thing this system can do, so the refusal is a fixed token rather than a
# sentence. Prose can be hedged and half-refused; a token either appears or it
# does not, which makes refusal measurable and lets the caller decide what the
# user actually reads.
REFUSAL_TOKEN = "NICHT_BEANTWORTBAR"

SYSTEM_PROMPT = f"""Du beantwortest Fragen zum deutschen Mietrecht ausschließlich \
auf Grundlage der Gesetzestexte, die dir im Kontext vorgelegt werden.

Regeln:
1. Nutze nur die vorgelegten Vorschriften. Ziehe kein Wissen heran, das nicht im \
Kontext steht, auch wenn du die Antwort zu kennen glaubst.
2. Belege jede Aussage mit der Vorschrift, aus der sie stammt, in eckigen Klammern \
und in der Form [§ 551 Abs. 1]. Zitiere nur Vorschriften, die im Kontext stehen.
3. Wenn die vorgelegten Vorschriften die Frage nicht beantworten, antworte \
ausschließlich mit {REFUSAL_TOKEN} und sonst nichts. Rate nicht.
4. Antworte auf Deutsch, in höchstens vier Sätzen, ohne Einleitung.
5. Du gibst Auskunft über den Gesetzeswortlaut, keine Rechtsberatung."""


def build_context(hits: list[dict]) -> str:
    """Format retrieved records as numbered, citable blocks.

    The citation is put on its own line above each text so the model has the
    exact string it is supposed to reproduce. Asking it to reconstruct "§ 551
    Abs. 1" from an id like `bgb-551-1` invites transcription errors that then
    look like hallucinated citations further down the pipeline.
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        p = hit["payload"]
        citation = strip_law_prefix(p["citation"])
        title = p.get("title") or ""
        header = f"[{i}] {citation}" + (f" — {title}" if title else "")
        blocks.append(f"{header}\n{p['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[dict]) -> str:
    return (
        "Vorschriften:\n\n"
        f"{build_context(hits)}\n\n"
        f"Frage: {question}"
    )


# German legal writing distinguishes the two sigils: a single § introduces one
# section, a doubled §§ introduces a list. That distinction is load-bearing here.
# Reading a list after a single § would swallow ordinary numbers, turning
# "§ 556 Abs. 3, 12 Monate" into a citation of § 12; not reading one after §§
# would check only the first section of "§§ 535, 536" and let the rest through
# unverified, which is the direction that hides a hallucination.
SIGIL_RE = re.compile(r"§§?")
ITEM_RE = re.compile(r"\s*(\d+[a-zA-Z]?)(?:\s*Abs(?:atz|\.)\s*(\d+))?", re.UNICODE)
# "bis" marks a range. Its endpoints are recorded and the sections between them
# are not, which is a deliberate under-count: a range is rare in a generated
# answer, and inventing intermediate citations to check would be worse.
SEPARATOR_RE = re.compile(r"\s*(?:,|;|\bund\b|\bbis\b)\s*", re.UNICODE)

LAW_PREFIX_RE = re.compile(r"^\s*BGB\s+", re.IGNORECASE)


def strip_law_prefix(citation: str) -> str:
    """`BGB § 551 Abs. 1` to `§ 551 Abs. 1`.

    Every section in the corpus is from the BGB, so repeating the law in each of
    ten context blocks spends tokens to say nothing.
    """
    return LAW_PREFIX_RE.sub("", citation).strip()


def parse_citations(text: str) -> list[tuple[str, str | None]]:
    """Pull every § reference out of a text, in order, deduplicated.

    Returns (section, absatz) pairs, where absatz is None for a bare section
    reference. Section numbers keep their letter suffix, so § 573c stays
    distinct from § 573. A trailing "Satz 2" is ignored: ground truth in this
    corpus stops at the Absatz, so a sentence number is detail nothing can be
    checked against.
    """
    seen: list[tuple[str, str | None]] = []

    for sigil in SIGIL_RE.finditer(text):
        is_list = sigil.group(0) == "§§"
        cursor = sigil.end()

        while True:
            item = ITEM_RE.match(text, cursor)
            if not item:
                break
            pair = (item.group(1).lower(), item.group(2) or None)
            if pair not in seen:
                seen.append(pair)
            cursor = item.end()

            if not is_list:
                break
            separator = SEPARATOR_RE.match(text, cursor)
            if not separator:
                break
            cursor = separator.end()

    return seen


def available_citations(hits: list[dict]) -> tuple[set[tuple[str, str | None]], set[str]]:
    """What the model was actually shown, as exact pairs and as bare sections."""
    exact: set[tuple[str, str | None]] = set()
    sections: set[str] = set()
    for hit in hits:
        for section, absatz in parse_citations(hit["payload"]["citation"]):
            exact.add((section, absatz))
            sections.add(section)
    return exact, sections


@dataclass
class CitationCheck:
    """Where every citation in an answer landed.

    exact       the Absatz was in the context
    section     the § was in the context, a different Absatz of it was cited
    unsupported the § was never shown at all
    """

    exact: list[tuple[str, str | None]] = field(default_factory=list)
    section: list[tuple[str, str | None]] = field(default_factory=list)
    unsupported: list[tuple[str, str | None]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.exact) + len(self.section) + len(self.unsupported)

    @property
    def grounded(self) -> bool:
        """No invented sections, and at least one citation to check."""
        return self.total > 0 and not self.unsupported

    def as_dict(self) -> dict:
        fmt = lambda pairs: [f"§ {s}" + (f" Abs. {a}" if a else "") for s, a in pairs]
        return {
            "exact": fmt(self.exact),
            "section_only": fmt(self.section),
            "unsupported": fmt(self.unsupported),
            "grounded": self.grounded,
        }


def verify_citations(answer: str, hits: list[dict]) -> CitationCheck:
    """Classify each citation in an answer against the retrieved context.

    A section-only match is not counted as a failure. The generator is handed a
    whole section's neighbouring Absätze often enough that pointing at the § when
    the precise Absatz is arguable is defensible; pointing at a § that was never
    retrieved is not.
    """
    exact_available, sections_available = available_citations(hits)
    check = CitationCheck()

    for pair in parse_citations(answer):
        section, _ = pair
        if pair in exact_available:
            check.exact.append(pair)
        elif section in sections_available:
            check.section.append(pair)
        else:
            check.unsupported.append(pair)
    return check


def is_refusal(answer: str) -> bool:
    """Did the model decline, rather than answer from nothing?

    Checked on the whole reply rather than the first line only: a model that
    prefixes the token with a courtesy sentence has still refused, and treating
    that as an answer would score a correct refusal as a hallucination.
    """
    return REFUSAL_TOKEN in answer.upper()


# Published rates in USD per million tokens. Kept next to the code that uses
# them so a run's reported cost can be traced to the number it was computed
# from, rather than to a figure written into a results table by hand.
PRICING_USD = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call, or 0.0 for a model that runs on your own hardware."""
    rates = PRICING_USD.get(model)
    if not rates:
        return 0.0
    per_in, per_out = rates
    return (input_tokens * per_in + output_tokens * per_out) / 1_000_000


@dataclass
class Answer:
    question: str
    text: str
    backend: str
    model: str
    citations: CitationCheck
    refused: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    retrieved: list[str]

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "backend": self.backend,
            "model": self.model,
            "refused": self.refused,
            "citations": self.citations.as_dict(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "retrieved": self.retrieved,
        }


class OllamaBackend:
    """A local model served by Ollama.

    Talks to the HTTP API through the standard library rather than the ollama
    package, because one POST does not justify a dependency that would then sit
    in requirements.txt for everyone who only wants to run retrieval.
    """

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:14b", host: str = "http://localhost:11434",
                 timeout: int = 180):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Greedy decoding. Two runs of the eval set should differ because the
            # pipeline changed, not because the sampler rolled differently.
            "options": {"temperature": 0.0},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"Ollama not reachable at {self.host} ({exc}). "
                f"Start it and pull the model:\n"
                f"  ollama serve\n  ollama pull {self.model}"
            ) from exc

        return (
            body["message"]["content"].strip(),
            body.get("prompt_eval_count", 0),
            body.get("eval_count", 0),
        )


class AnthropicBackend:
    """Claude through the official SDK.

    The key is read from the environment by the SDK itself, so it is never
    passed through this code and never ends up in a run's saved output.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError as exc:
            raise SystemExit(
                "The anthropic package is not installed. Either:\n"
                "  pip install anthropic\n"
                "or run against a local model with --backend ollama"
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text, response.usage.input_tokens, response.usage.output_tokens


def build_backend(kind: str, model: str | None = None, **kwargs):
    if kind == "ollama":
        return OllamaBackend(model or "qwen2.5:14b", **kwargs)
    if kind == "anthropic":
        return AnthropicBackend(model or "claude-haiku-4-5", **kwargs)
    raise ValueError(f"unknown backend {kind!r}")


def generate(question: str, hits: list[dict], backend) -> Answer:
    """One question, one generation, with everything needed to score it later."""
    prompt = build_prompt(question, hits)

    started = time.perf_counter()
    text, input_tokens, output_tokens = backend.complete(SYSTEM_PROMPT, prompt)
    latency_ms = (time.perf_counter() - started) * 1000

    return Answer(
        question=question,
        text=text,
        backend=backend.name,
        model=backend.model,
        citations=verify_citations(text, hits),
        refused=is_refusal(text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd(backend.model, input_tokens, output_tokens),
        retrieved=[h["record_id"] for h in hits],
    )
