# German Legal RAG

Retrieval over German tenancy law (Mietrecht) from the Bürgerliches Gesetzbuch, built to be measured rather than demoed. The focus is the evaluation: what the retrieval quality actually is, and how much each change moves it. Chaining a vector store to an LLM is easy. Knowing whether it works, and why, is the harder part, and the part this project is about.

Work in progress. The plan and the phase I am on live in [ROADMAP.md](ROADMAP.md).

## Status

| Part | State |
| --- | --- |
| Corpus and parser | Done |
| Evaluation set | 68 questions, expanded from error analysis |
| Retrieval | hit@1 0.74, MRR 0.82 on the final config, see [RESULTS.md](RESULTS.md) |
| Experiments | Six run, two helped. Phase complete |
| Generation and serving | Not started |

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

# embed the records into a local Qdrant collection (downloads bge-m3 once)
python src/index_corpus.py --granularity absatz

# score dense retrieval against the eval set
python src/eval_retrieval.py --collection absatz_raw --label dense-baseline

# or query it directly
python src/retrieve.py "Wie hoch darf die Kaution sein?"
```

## Layout

```
scripts/fetch_corpus.sh   download and unpack a law
src/parse_bgb.py          XML to JSONL records
src/validate_eval.py      check ground-truth references
src/embedder.py           the embedding model wrapper
src/index_corpus.py       embed records into Qdrant
src/retrieve.py           dense retrieval + query CLI
src/eval_retrieval.py     hit-rate@k and MRR against the eval set
data/eval/mietrecht.jsonl the evaluation questions
RESULTS.md                retrieval scores
ROADMAP.md                phases and metrics
```

## Scope

Statute text only, no case law. A question whose real answer turns on a court ruling will retrieve the correct § and still answer incompletely, and that limit is worth stating plainly. The eventual app gives information, not legal advice.
