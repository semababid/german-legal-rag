"""HTTP interface to the retrieval and generation pipeline.

    uvicorn api:app --app-dir src --port 8000

    POST /ask     a German question, back a cited answer
    GET  /health  is the model loaded and the corpus indexed
    GET  /search  retrieval only, for looking at what the generator was given

Two things drive the shape of this file.

The embedding model takes about ten seconds to load and a query takes about two
tenths of a second once it has. So the retriever and the backend are built once
while the app starts and reused for every request. Building them per request
would make each answer ten seconds slower for no reason.

And every request is logged as one JSON line: the question, which records came
back with what scores, how the citations checked out, tokens, and the time split
between retrieval and generation. Without that there is no way to answer "why
did it say that" about a request that already happened, which is the question
anyone actually asks about a deployed system.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import build_backend, generate  # noqa: E402
from retrievers import build_retriever  # noqa: E402

# Answering legal questions for other people in Germany runs into the
# Rechtsdienstleistungsgesetz. This is information about the wording of the
# statute, not advice, and the response says so on every request rather than
# only in a footer somebody can miss.
DISCLAIMER = (
    "Diese Auskunft gibt den Gesetzeswortlaut wieder und ist keine "
    "Rechtsberatung im Sinne des Rechtsdienstleistungsgesetzes."
)

SETTINGS = {
    "retriever": os.getenv("RETRIEVER", "domain"),
    "collection": os.getenv("COLLECTION", "absatz_ctx"),
    "backend": os.getenv("BACKEND", "ollama"),
    "model": os.getenv("MODEL") or None,
    "k": int(os.getenv("TOP_K", "6")),
    # Questions can carry personal circumstances, which is exactly the sort of
    # thing that should not land in a log by accident. On by default because the
    # log is useless for debugging retrieval without it, and switchable because
    # that trade belongs to whoever runs the service.
    "log_questions": os.getenv("LOG_QUESTIONS", "true").lower() != "false",
}


class JsonLines(logging.Formatter):
    """One JSON object per line, so the log can be queried rather than read."""

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            return json.dumps({"level": record.levelname.lower(),
                               "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                               **record.msg}, ensure_ascii=False)
        return json.dumps({"level": record.levelname.lower(),
                           "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                           "message": record.getMessage()}, ensure_ascii=False)


log = logging.getLogger("rag")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonLines())
log.addHandler(_handler)
log.setLevel(logging.INFO)
log.propagate = False


@dataclass
class Pipeline:
    retriever: object
    backend: object
    # One request at a time through the model. The embedder holds MPS state and
    # the local model serialises anyway, so letting FastAPI's threadpool run
    # several at once buys nothing and risks a race. A queue in front of a
    # single worker is the honest shape of this service.
    lock: threading.Lock


pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    started = time.perf_counter()
    retriever = build_retriever(SETTINGS["retriever"], collection=SETTINGS["collection"])
    backend = build_backend(SETTINGS["backend"], SETTINGS["model"])
    pipeline = Pipeline(retriever, backend, threading.Lock())
    log.info({"event": "startup", "seconds": round(time.perf_counter() - started, 1),
              **{k: v for k, v in SETTINGS.items() if k != "log_questions"}})
    yield
    if hasattr(retriever, "close"):
        retriever.close()
    log.info({"event": "shutdown"})


app = FastAPI(
    title="German Legal RAG",
    description="Retrieval over German tenancy law (BGB §§ 535 to 580a). "
                "Information about the statute, not legal advice.",
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    id: str
    citation: str
    title: str | None
    text: str
    score: float
    url: str


class Answer(BaseModel):
    answer: str
    outcome: str
    sources: list[Source]
    citations: dict
    disclaimer: str = DISCLAIMER
    request_id: str
    timings_ms: dict
    tokens: dict


def require_pipeline() -> Pipeline:
    if pipeline is None:
        raise HTTPException(503, "model still loading")
    return pipeline


@app.get("/health")
def health() -> dict:
    ready = pipeline is not None
    return {"ready": ready, "backend": SETTINGS["backend"],
            "collection": SETTINGS["collection"]}


@app.post("/ask", response_model=Answer)
def ask(body: Question) -> Answer:
    p = require_pipeline()
    request_id = uuid.uuid4().hex[:12]
    k = body.k or SETTINGS["k"]

    with p.lock:
        t0 = time.perf_counter()
        hits = p.retriever.search(body.question, k=k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not hits:
            # In scope for the corpus but nothing came back. Saying so is better
            # than handing the generator an empty prompt and letting it invent.
            log.info({"event": "ask", "request_id": request_id, "outcome": "no_hits",
                      **({"question": body.question} if SETTINGS["log_questions"] else {}),
                      "retrieval_ms": round(retrieval_ms, 1)})
            raise HTTPException(404, "no tenancy provisions matched that question")

        try:
            result = generate(body.question, hits, p.backend)
        except Exception as exc:
            log.error({"event": "ask", "request_id": request_id, "outcome": "error",
                       "error": f"{type(exc).__name__}: {exc}",
                       **({"question": body.question} if SETTINGS["log_questions"] else {})})
            raise HTTPException(502, "the model did not return an answer") from exc

    sources = [
        Source(
            id=h["record_id"],
            citation=h["payload"]["citation"],
            title=h["payload"].get("title"),
            text=h["payload"]["text"],
            score=round(h["score"], 4),
            url=h["payload"]["source_url"],
        )
        for h in hits
    ]

    log.info({
        "event": "ask",
        "request_id": request_id,
        **({"question": body.question} if SETTINGS["log_questions"] else {}),
        "outcome": result.outcome,
        "retrieved": [{"id": s.id, "citation": s.citation, "score": s.score} for s in sources],
        "citations": result.citations.as_dict(),
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round(result.latency_ms, 1),
        "total_ms": round(retrieval_ms + result.latency_ms, 1),
    })

    return Answer(
        answer=result.text,
        outcome=result.outcome,
        sources=sources,
        citations=result.citations.as_dict(),
        request_id=request_id,
        timings_ms={"retrieval": round(retrieval_ms, 1),
                    "generation": round(result.latency_ms, 1),
                    "total": round(retrieval_ms + result.latency_ms, 1)},
        tokens={"input": result.input_tokens, "output": result.output_tokens},
    )


@app.get("/search")
def search(q: str, k: int = 6) -> dict:
    """Retrieval without generation.

    The generator can only be as good as what it is handed, so being able to see
    that separately is what makes a bad answer diagnosable.
    """
    p = require_pipeline()
    with p.lock:
        t0 = time.perf_counter()
        hits = p.retriever.search(q, k=k)
        elapsed = (time.perf_counter() - t0) * 1000

    return {
        "query": q,
        "took_ms": round(elapsed, 1),
        "hits": [
            {"id": h["record_id"], "citation": h["payload"]["citation"],
             "title": h["payload"].get("title"), "score": round(h["score"], 4),
             "text": h["payload"]["text"]}
            for h in hits
        ],
    }
