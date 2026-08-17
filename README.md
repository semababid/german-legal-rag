# German Legal RAG

Retrieval over German tenancy law (Mietrecht) from the Bürgerliches Gesetzbuch, built to be measured rather than demoed. The focus is the evaluation: what the retrieval quality actually is, and how much each change moves it. Chaining a vector store to an LLM is easy. Knowing whether it works, and why, is the harder part, and the part this project is about.

Work in progress. The plan and the phase I am on live in [ROADMAP.md](ROADMAP.md).

## Status

| Part | State |
| --- | --- |
| Corpus and parser | Done |
| Evaluation set | 150 questions, 77 sections covered |
| Retrieval | hit@1 0.70, MRR 0.79 on the final config, see [RESULTS.md](RESULTS.md) |
| Measurement | n=150, two scoring levels, core and peripheral split, all comparisons significance tested |
| Experiments | Six run, two significant improvements. Phase complete |
| Generation | Working end to end on the local model. Answers are cited and grounding-checked |
| Judging | 30 answers graded by hand. The local judge scored below the guessing baseline on faithfulness, so answers are not model-judged |
| Serving | FastAPI with structured logging, running in containers against a Qdrant service. Frontend and deployment still to do |

## The corpus

The full BGB, pulled from [gesetze-im-internet.de](https://www.gesetze-im-internet.de/bgb/) as structured XML. German federal statutes are amtliche Werke under § 5 UrhG and carry no copyright, so the text is free to redistribute.

`parse_bgb.py` flattens the XML into two forms:

- **4,921 Absatz records**, one per numbered paragraph. This is the default chunk unit.
- **2,486 section records**, one per §. Used to test whether the chunk boundary matters.

Each record keeps its place in the hierarchy. The source XML stores the Buch, Abschnitt, and Titel headings as separate nodes, so the parser walks the document in order and rebuilds the full breadcrumb for every section. A record like § 573c Abs. 1 (notice periods) states a deadline without ever using the words Mietvertrag, Wohnraum, or Frist, so that context is worth attaching before the text is embedded.

## Running it

Building the corpus and the eval set needs only Python 3 and the standard library.

```bash
# download the BGB XML (records builddate and checksum in provenance.txt)
./scripts/fetch_corpus.sh bgb

# parse it into records
python src/parse_bgb.py --law bgb --stats

# check the eval set resolves against the corpus
python src/validate_eval.py
```

The corpus itself is not committed. It regenerates from the fetch script, and `provenance.txt` ties any run to a dated revision of the law.

Retrieval needs the packages in `requirements.txt` (sentence-transformers, torch, qdrant-client). Qdrant runs on-disk in local mode, so there is no server to start.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# embed the records with their structural breadcrumb (downloads bge-m3 once)
python src/index_corpus.py --granularity absatz --with-context

# score the final configuration against the eval set
python src/eval_retrieval.py --retriever domain --collection absatz_ctx --label v150-domain

# or query it directly
python src/retrieve.py "Wie hoch darf die Kaution sein?"

# check whether a difference between two runs is real or noise
python src/compare_runs.py --all

python -m pytest tests/ -q
```

To reproduce the raw-text baseline instead, index without `--with-context` and score
`--collection absatz_raw`. Every row in [RESULTS.md](RESULTS.md) lists the flags that
produced it.

## Answering

Retrieval hands its sections to a model, which answers in German and cites the
provisions it used. Two backends, chosen with a flag, because whether a system
like this can run on your own hardware is a real question in Germany and not
only a budget one.

```bash
# local: free, no key, nothing leaves the machine
brew install ollama && ollama serve
ollama pull qwen3.5:9b
python src/answer.py "Wie hoch darf die Kaution sein?"

# or through the API
export ANTHROPIC_API_KEY=...
python src/answer.py --backend anthropic "Wann darf ich die Miete mindern?"

# the whole eval set, saved for grading in the next step
python src/generate_answers.py --backend ollama --label local-qwen
```

Every answer is checked against the sections it was shown. Each § it cites comes
back as one of three things: the exact Absatz that was retrieved, a different
Absatz of a section that was retrieved, or a section that was never in front of
the model at all. The third case is the one worth catching, and catching it
needs no judge and no key.

The model is also told to refuse. Five questions in the eval set are answered by
no retrieval configuration tested, so they double as refusal cases: declining
those is the correct behaviour, and a fixed token rather than a hedged sentence
makes it countable.

Qwen 3.5 reasons before answering unless told not to, and it is expensive: a
one-sentence answer cost 1,444 output tokens with reasoning against 43 without.
It is off by default so a run over 150 questions finishes in minutes, and
because the API side of the comparison is measured without extended thinking
too. `--think` turns it back on and measures the other side.

## Serving it

```bash
uvicorn api:app --app-dir src --port 8000

curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"Wie hoch darf die Mietkaution sein?"}'

curl "localhost:8000/search?q=Schimmel%20in%20der%20Wohnung&k=3"
```

`/ask` returns the answer, the sections it was given with their scores and links
to gesetze-im-internet.de, the citation check, timings and token counts. Every
response carries a notice that this is the statute rather than legal advice.
`/search` runs retrieval alone, which is what makes a bad answer diagnosable:
the generator can only work with what it was handed.

The embedding model takes about ten seconds to load and a query takes two tenths
of a second afterwards, so it is loaded once at startup and reused. Requests are
handled one at a time, since the local model serialises anyway.

Every request is logged as one JSON line with the question, the records
retrieved and their scores, how the citations checked out, tokens, and the split
between retrieval and generation time:

```json
{"event": "ask", "request_id": "e7fb229c487f", "question": "Wie hoch darf die Mietkaution sein?",
 "outcome": "answer", "retrieved": [{"id": "bgb-551-1", "citation": "BGB § 551 Abs. 1", "score": 0.6305}],
 "citations": {"exact": ["§ 551 Abs. 1"], "grounded": true},
 "input_tokens": 1104, "output_tokens": 121,
 "retrieval_ms": 177.9, "generation_ms": 11984.0, "total_ms": 12162.0}
```

Questions can carry personal circumstances, so `LOG_QUESTIONS=false` keeps them
out of the log while leaving everything else in place.

## In containers

```bash
docker compose --profile tools run --rm indexer   # once, embeds the corpus
docker compose up -d                              # qdrant and the api
```

Qdrant moves from an on-disk directory to a service, which is what `QDRANT_URL`
selects. Local mode needs no server and is what makes the retrieval experiments
reproducible on a laptop, but it takes an exclusive lock on its storage
directory, so indexing and serving cannot happen at the same time. Pointing at a
server removes that. Nothing above `qdrant_client()` changes either way.

The language model stays on the host. It is 6.6 GB already pulled, it wants
Metal to be usable, and a Linux container on a Mac cannot have that, so the API
container reaches back out through `OLLAMA_HOST=http://host.docker.internal:11434`.
This compose file is therefore not the whole system, which is worth saying
plainly rather than implying one command starts everything.

The same limit applies to indexing. Inside the container the embedder runs on
CPU, so building the collection takes far longer than the few minutes it takes
on the host with MPS. Indexing on the host and pointing the container at the
resulting Qdrant is the faster route.

| Variable | Default | |
| --- | --- | --- |
| `QDRANT_URL` | unset, on-disk | a Qdrant server instead |
| `QDRANT_API_KEY` | unset | sent only when set |
| `OLLAMA_HOST` | `http://localhost:11434` | where the model is |
| `COLLECTION` | `absatz_ctx` | which index to read |
| `RETRIEVER` | `domain` | strategy, as in phase 3 |
| `TOP_K` | `6` | sections per prompt |
| `LOG_QUESTIONS` | `true` | keep questions out of the log |

## Layout

```
scripts/fetch_corpus.sh   download and unpack a law
src/parse_bgb.py          XML to JSONL records
src/validate_eval.py      check ground-truth references
src/embedder.py           the embedding model wrapper
src/index_corpus.py       embed records into Qdrant
src/retrieve.py           dense retrieval + query CLI
src/eval_retrieval.py     hit-rate@k and MRR against the eval set
src/compare_runs.py       significance testing between two runs
src/generator.py          prompt, model backends, citation checking
src/answer.py             ask one question, get a cited answer
src/generate_answers.py   answer the whole eval set and save it
src/make_calibration_set.py  draw the stratified sample to grade by hand
src/grade.py              grade those answers, one at a time
src/judge.py              LLM judge, scored against the hand grades
src/compare_backends.py   two answer runs side by side
src/api.py                FastAPI service with structured logging
src/metrics.py            shared percentile and median
Dockerfile                the API image, CPU-only torch
docker-compose.yml        qdrant service, api, one-shot indexer
src/judge.py              LLM judge, scored against the hand grades
src/compare_backends.py   two answer runs side by side
src/api.py                FastAPI service with structured logging
src/metrics.py            shared percentile and median
Dockerfile                the API image, CPU-only torch
docker-compose.yml        qdrant service, api, one-shot indexer
src/judge.py              model-graded scoring, and its agreement with the hand grades
tests/                    unit tests for the metric, retrieval and citation logic
data/eval/mietrecht.jsonl the evaluation questions
RESULTS.md                retrieval scores
ROADMAP.md                phases and metrics
```

## Scope

Statute text only, no case law. A question whose real answer turns on a court ruling will retrieve the correct § and still answer incompletely, and that limit is worth stating plainly. The eventual app gives information, not legal advice.
