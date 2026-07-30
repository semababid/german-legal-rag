# German Legal RAG

Retrieval over German tenancy law (Mietrecht) from the Bürgerliches Gesetzbuch, built to be measured rather than demoed. The focus is the evaluation: what the retrieval quality actually is, and how much each change moves it. Chaining a vector store to an LLM is easy. Knowing whether it works, and why, is the harder part, and the part this project is about.

Work in progress. The plan and the phase I am on live in [ROADMAP.md](ROADMAP.md).

## Status

| Part | State |
| --- | --- |
| Corpus and parser | Done |
| Evaluation set | 30 of 120 questions written |
| Retrieval | Not started |
| Generation and serving | Not started |

## The corpus

The full BGB, pulled from [gesetze-im-internet.de](https://www.gesetze-im-internet.de/bgb/) as structured XML. German federal statutes are amtliche Werke under § 5 UrhG and carry no copyright, so the text is free to redistribute.

`parse_bgb.py` flattens the XML into two forms:

- **4,921 Absatz records**, one per numbered paragraph. This is the default chunk unit.
- **2,486 section records**, one per §. Used to test whether the chunk boundary matters.

Each record keeps its place in the hierarchy. The source XML stores the Buch, Abschnitt, and Titel headings as separate nodes, so the parser walks the document in order and rebuilds the full breadcrumb for every section. A record like § 573c Abs. 1 (notice periods) states a deadline without ever using the words Mietvertrag, Wohnraum, or Frist, so that context is worth attaching before the text is embedded.

## Running it

Phase 0 needs only Python 3 and the standard library.

```bash
# download the BGB XML (records builddate and checksum in provenance.txt)
./scripts/fetch_corpus.sh bgb

# parse it into records
python src/parse_bgb.py --law bgb --stats

# check the eval set resolves against the corpus
python src/validate_eval.py
```

The corpus itself is not committed. It regenerates from the fetch script, and `provenance.txt` ties any run to a dated revision of the law.

## Layout

```
scripts/fetch_corpus.sh   download and unpack a law
src/parse_bgb.py          XML to JSONL records
src/validate_eval.py      check ground-truth references
data/eval/mietrecht.jsonl the evaluation questions
ROADMAP.md                phases and metrics
```

## Scope

Statute text only, no case law. A question whose real answer turns on a court ruling will retrieve the correct § and still answer incompletely, and that limit is worth stating plainly. The eventual app gives information, not legal advice.
