# German Legal RAG

Retrieval over German tenancy law (Mietrecht) from the Bürgerliches Gesetzbuch,
built to be measured rather than demoed. Chaining a vector store to an LLM is
easy. Knowing whether it works, and by how much, is the harder part, and it is
what this project is about.

## Results

150 hand-written German questions, each labelled with the sections of the BGB
that answer it. Every number below is on that set.

| Level | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| Paragraph | 0.700 | 0.873 | 0.900 | 0.947 | 0.790 |
| Strict | 0.627 | 0.833 | 0.867 | 0.927 | 0.737 |

29 ms per query. bge-m3 over Absatz-level chunks, each embedded with its
structural breadcrumb, restricted to §§ 535 to 580a. No reranker, no hybrid
search, no second model in the request path.

Six changes were tried against the baseline. Two helped.

| Change | Effect on hit@1 | Holds up? |
| --- | --- | --- |
| Index the structural breadcrumb with the text | 0.533 to 0.660 | yes, p = 0.003 |
| Restrict results to the tenancy sections | 0.660 to 0.700 | yes, p = 0.031 |
| Absätze instead of whole sections | 0.613 to 0.660 | not distinguishable |
| Cross-encoder reranking | 0.660 to 0.620 | not distinguishable, 67 times slower |
| Fuse dense and BM25 with RRF | 0.533 to 0.447 | worse |
| BM25 instead of dense | 0.533 to 0.247 | much worse, p < 0.0001 |

Every comparison is significance tested, McNemar's exact test on hit@1 and a
bootstrap interval on the paired MRR difference. The two changes that survived
are both structural and cheap: one changes what gets stored, the other filters
what comes back. The shipped configuration is among the simplest tried and also
the fastest. Full working in [RESULTS.md](RESULTS.md).

The result I did not expect is that BM25 lost badly. German compound nouns
looked like exactly the case where lexical matching should win. It would have,
if the questions reused the statute's compounds, and they mostly do not, because
they were written the way a tenant would ask rather than copied from the law.

## What it looks like

![A cited answer about deposit limits](docs/img/answer.png)

The answer is German, every § in it links to the provision it came from, and
below it sits every section the model was shown with its full text and a link to
gesetze-im-internet.de. The Rechtsdienstleistungsgesetz notice is above the
search box rather than in a footer, because a notice placed after the answer is
one people read after acting on it.

![The six sections the model was given](docs/img/sources.png)

All six are listed, not only the ones that got cited, and the bottom three here
are about rent increases rather than deposits. That is what a top-6 retrieval
actually looks like and hiding it would be dishonest about the input the model
worked from. It is also the reason `/search` exists as its own endpoint: when an
answer is bad, the first question is whether the right text ever reached the
model.

## What it gets wrong

This is the same system, asked a question in everyday German instead of legal
German.

![A wrong answer that passes the citation check](docs/img/layperson.png)

The answer is wrong. A defect that serious gives you a rent reduction under
§ 536, and the page says you owe the full rent. Two separate things failed and
neither one reported a problem.

The retriever's top hit is § 559e, on rent *increases* after a heating system is
installed. The question said Heizung, so it matched Heizung. The salient noun in
a layperson question is often incidental to the legal question, and the retriever
matched the wrong concept perfectly faithfully. This is measurable and measured:
layperson phrasing scores 0.441 at hit@1 against 0.753 for direct questions, and
nothing tried in phase 3 fixed it.

The green check underneath is the second failure, and it is the more instructive
one. It is not lying. It answers a narrower question than it appears to: did this
answer cite a § that was never retrieved. § 536c was retrieved, so the check
passes, and it has no opinion on whether the sentence built around it is true.

That gap is why 30 answers were graded by hand. Of the 8 whose citations were
mechanically perfect, 4 were judged fully faithful. As a proxy for faithfulness
the citation check is right about half the time, which is worth knowing before
shipping it as a green tick.

## Honest limits

**Layperson German is where it is worst.** 0.441 at hit@1. The obvious next lever
is rewriting the query into statutory language before retrieval, and it was out
of scope here.

**Answers are not scored for faithfulness.** A local judge was built, calibrated
against the 30 hand grades, and then not used: it scored kappa 0.04 and 0.17 on
faithfulness across two prompt versions, against a 53% majority-class baseline. A
judge that ignored the answer and always guessed "partly" would have been more
accurate. So faithfulness on the other 120 answers is unmeasured rather than
measured wrongly. Running the judge over all 150 and publishing the numbers would
have looked more finished and nothing in that output would have revealed it was
worse than guessing.

**Statute text only, no case law.** Plenty of real tenancy questions turn on BGH
rulings. Those retrieve the correct § and still answer incompletely.

**One grader, 30 items, no second annotator.** So there is no inter-annotator
agreement figure, and no way to separate a judge disagreeing with the grader from
a judge disagreeing with the truth.

**The model refuses about twice as often as it should.** All 12 answers that
reached for the refusal token were judged not to have answered the question, and
6 of those 12 declined when the sections in front of them did contain the answer.

**It is not deployed.** Two targets are built and tested, a Hugging Face Space and
a free Oracle VM, and neither is pushed. Docker Spaces moved behind a $9/month
plan during the project, and the Oracle account is not set up yet. See
[Deploying it](#deploying-it).

## The corpus

The full BGB, pulled from
[gesetze-im-internet.de](https://www.gesetze-im-internet.de/bgb/) as structured
XML. German federal statutes are amtliche Werke under § 5 UrhG and carry no
copyright, so the text is free to redistribute.

`parse_bgb.py` flattens the XML into two forms:

- **4,921 Absatz records**, one per numbered paragraph. The default chunk unit.
- **2,486 section records**, one per §. Used to test whether the boundary matters.

Each record keeps its place in the hierarchy. The XML stores Buch, Abschnitt and
Titel headings as separate nodes, so the parser walks the document in order and
rebuilds the full breadcrumb for every section. § 573c Abs. 1 states a notice
period without ever using the words Mietvertrag, Wohnraum or Frist, so that
context is worth attaching before the text is embedded. Measuring that hunch is
where most of the gain in the results table came from.

## Running it

Building the corpus and the eval set needs only Python 3 and the standard library.

```bash
./scripts/fetch_corpus.sh bgb        # records builddate and checksum in provenance.txt
python src/parse_bgb.py --law bgb --stats
python src/validate_eval.py          # check every ground-truth reference resolves
```

The corpus is not committed. It regenerates from the fetch script, and
`provenance.txt` ties any run to a dated revision of the law.

Retrieval needs `requirements.txt`. Qdrant runs on-disk in local mode, so there
is no server to start.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/index_corpus.py --granularity absatz --with-context
python src/eval_retrieval.py --retriever domain --collection absatz_ctx --label v150-domain
python src/retrieve.py "Wie hoch darf die Kaution sein?"
python src/compare_runs.py --all     # is a difference between two runs real?

python -m pytest tests/ -q
```

To reproduce the raw-text baseline instead, index without `--with-context` and
score `--collection absatz_raw`. Every row in [RESULTS.md](RESULTS.md) lists the
flags that produced it.

## Answering

Retrieval hands its sections to a model, which answers in German and cites what
it used. Two backends behind a flag, because whether a system like this can run
on your own hardware is a real question in Germany and not only a budget one.

```bash
brew install ollama && ollama serve
ollama pull qwen3.5:9b
python src/answer.py "Wie hoch darf die Kaution sein?"

export ANTHROPIC_API_KEY=...
python src/answer.py --backend anthropic "Wann darf ich die Miete mindern?"

python src/generate_answers.py --backend ollama --label local-qwen   # the whole set
```

Every answer is checked against the sections it was shown. Each § it cites comes
back as one of four things: the exact Absatz retrieved, a different Absatz of a
retrieved section, a section named inside retrieved text but never shown in full,
or a section that was nowhere at all. Across 150 answers the last case happened
twice.

That four-way split took three corrections to build, and the corrections matter
more than the number. The first count was 19. Two of those were the checker
misreading references to *other* statutes: an answer faithfully quoting § 555b
reproduces its pointer to § 3 Nr. 33 of the Telekommunikationsgesetz, which is
not a BGB section at all. The remaining 17 looked like the model citing from
memory. Fifteen were not. The BGB refers to itself constantly, and § 559 Abs. 1
literally opens with "Modernisierungsmaßnahmen im Sinne des § 555b", so a model
shown § 559 and citing § 555b is repeating a pointer it read. Counting those as
inventions overstated the rate eightfold.

The model is also told to refuse. Five questions are answered by no retrieval
configuration tested, so they double as refusal cases, and a fixed token rather
than a polite sentence makes refusal countable.

Qwen 3.5 reasons before answering unless told not to, and it is expensive:
reasoning costs 22 times the latency and 30 times the output tokens, and 2 of 52 attempts
produced nothing usable at all. It buys one thing, cleaner refusals, and that is
not worth 170 seconds a question. Off by default; `--think` measures the other
side.

## Serving it

```bash
uvicorn api:app --app-dir src --port 8000

curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"Wie hoch darf die Mietkaution sein?"}'

curl "localhost:8000/search?q=Schimmel%20in%20der%20Wohnung&k=3"
```

`/ask` returns the answer, the sections it was given with their scores, the
citation check, timings and token counts. `/search` runs retrieval alone, which
is what makes a bad answer diagnosable: the generator can only work with what it
was handed, and the screenshot above is a retrieval failure wearing a generation
failure's clothes.

The embedding model takes about ten seconds to load and a query takes two tenths
of a second after that, so it loads once at startup. Requests are handled one at
a time, since the local model serialises anyway.

Every request is one JSON line:

```json
{"event": "ask", "request_id": "e7fb229c487f", "question": "Wie hoch darf die Mietkaution sein?",
 "outcome": "answer", "retrieved": [{"id": "bgb-551-1", "citation": "BGB § 551 Abs. 1", "score": 0.6305}],
 "citations": {"exact": ["§ 551 Abs. 1"], "grounded": true},
 "input_tokens": 1104, "output_tokens": 121,
 "retrieval_ms": 177.9, "generation_ms": 11984.0, "total_ms": 12162.0}
```

Questions carry personal circumstances, so `LOG_QUESTIONS=false` keeps them out
while leaving everything else in place. The public deployments set it.

`GET /` serves one HTML file. No build step, no framework, no npm, which keeps
the deployment a container with a Python process in it. Screenshots regenerate
with `python scripts/make_screenshots.py` against a running instance.

## In containers

```bash
docker compose --profile tools run --rm indexer   # once, embeds the corpus
docker compose up -d                              # qdrant and the api
```

Qdrant moves from an on-disk directory to a service, which is what `QDRANT_URL`
selects. Local mode needs no server and is what makes the experiments
reproducible on a laptop, but it takes an exclusive lock on its storage
directory, so indexing and serving cannot happen at once. Pointing at a server
removes that. Nothing above `qdrant_client()` changes either way.

The language model stays on the host. It is 6.6 GB, it wants Metal to be usable,
and a Linux container on a Mac cannot have that, so the API container reaches
back out through `OLLAMA_HOST=http://host.docker.internal:11434`. This compose
file is therefore not the whole system, which is worth saying plainly rather than
implying one command starts everything.

| Variable | Default | |
| --- | --- | --- |
| `QDRANT_URL` | unset, on-disk | a Qdrant server instead |
| `QDRANT_API_KEY` | unset | sent only when set |
| `OLLAMA_HOST` | `http://localhost:11434` | where the model is |
| `COLLECTION` | `absatz_ctx` | which index to read |
| `RETRIEVER` | `domain` | strategy, as in phase 3 |
| `TOP_K` | `6` | sections per prompt |
| `LOG_QUESTIONS` | `true` | set false to keep questions out of the log |

## Deploying it

Two targets, both built and tested, neither currently live.

```bash
python src/index_corpus.py --granularity absatz --with-context

./deploy/push_to_spaces.sh <username>/<space-name>   # Hugging Face, needs PRO
./deploy/oracle/upload.sh ubuntu@<instance-ip>       # a free Oracle VM
```

Hugging Face moved Docker Spaces behind a $9/month plan partway through this
project; only Static Spaces are free now, and those cannot run Python, so they
cannot run the embedder. The Oracle path uses Always Free, which gives 2 Arm
cores and 12 GB for the life of the account, with Caddy in front for automatic
TLS so the result is a real HTTPS link rather than an IP. Step by step, including
the two firewalls Oracle makes you open separately, is in
[deploy/oracle/SETUP.md](deploy/oracle/SETUP.md).

Either way the index travels rather than being built on arrival. Two cores and no
GPU took about thirty five minutes to embed 4,921 records, so building at startup
would leave the thing unreachable after every restart.

The language model does not travel at all. It is 6.6 GB and wants a GPU, and a
model small enough to run on two cores would answer worse than the retrieval it
sits on. So `BACKEND=none` runs retrieval alone: `/ask` returns 501, the page
asks `/health` on load and stops offering answers it cannot write, and sections
come back with their scores and text. `BACKEND=anthropic` plus a key turns
generation on with nothing else changed.

Shipping a weak generator instead would have been easy and would have
misrepresented the project, since retrieval is the measured part.

## Layout

```
scripts/fetch_corpus.sh      download and unpack a law
scripts/make_screenshots.py  regenerate the README images
src/parse_bgb.py             XML to JSONL records
src/validate_eval.py         check ground-truth references
src/embedder.py              the embedding model wrapper
src/index_corpus.py          embed records into Qdrant
src/retrieve.py              dense retrieval + query CLI
src/eval_retrieval.py        hit-rate@k and MRR against the eval set
src/compare_runs.py          significance testing between two runs
src/metrics.py               shared percentile and median
src/generator.py             prompt, model backends, citation checking
src/answer.py                ask one question, get a cited answer
src/generate_answers.py      answer the whole eval set and save it
src/make_calibration_set.py  draw the stratified sample to grade by hand
src/grade.py                 grade those answers, one at a time
src/judge.py                 LLM judge, scored against the hand grades
src/compare_backends.py      two answer runs side by side
src/api.py                   FastAPI service with structured logging
web/index.html               the page, one file, no build step
Dockerfile                   the API image, CPU-only torch
docker-compose.yml           qdrant service, api, one-shot indexer
deploy/                      Hugging Face Space: image, README, push script
deploy/oracle/               a free VM: compose, Caddy, upload script, setup guide
tests/                       100 tests over metrics, retrieval, citations, config
data/eval/mietrecht.jsonl    the evaluation questions
RESULTS.md                   every number and how it was produced
ROADMAP.md                   phases, metrics, decisions
```

## Scope

The eventual app gives information, not legal advice. Answering legal questions
for others in Germany touches the Rechtsdienstleistungsgesetz, which is why the
notice is on the page and not only in this file.
