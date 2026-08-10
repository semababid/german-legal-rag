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
| Generation | Working end to end on the local model. Answers are cited and grounding-checked; not yet scored for quality |
| Serving | Not started |

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

The model is also told to refuse. Eight questions in the eval set are answered
by no retrieval configuration tested, so they double as refusal cases: declining
those is the correct behaviour, and a fixed token rather than a hedged sentence
makes it countable.

Qwen 3.5 reasons before answering unless told not to, and it is expensive: a
one-sentence answer cost 1,444 output tokens with reasoning against 43 without.
It is off by default so a run over 150 questions finishes in minutes, and
because the API side of the comparison is measured without extended thinking
too. `--think` turns it back on and measures the other side.

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
tests/                    unit tests for the metric, retrieval and citation logic
data/eval/mietrecht.jsonl the evaluation questions
RESULTS.md                retrieval scores
ROADMAP.md                phases and metrics
```

## Scope

Statute text only, no case law. A question whose real answer turns on a court ruling will retrieve the correct § and still answer incompletely, and that limit is worth stating plainly. The eventual app gives information, not legal advice.
