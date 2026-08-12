# Roadmap

A retrieval system over German tenancy law (Mietrecht, BGB §§ 535 to 580a), built to be measured rather than demoed. The evaluation harness is the point of the project. Chaining a vector store to an LLM is easy; showing what the retrieval quality actually is, and how specific changes moved it, is what almost no portfolio project does.

Assumed pace is six weeks part time, with no fixed deadline baked in. If you are aiming at a specific application window, compress phases 5 and 6 first.

## Phases

| # | Focus | Deliverable | Done when | Est. |
| --- | --- | --- | --- | --- |
| 0 | Corpus and parser | `fetch_corpus.sh`, `parse_bgb.py`, 4,921 Absatz records with breadcrumbs and citations | Done | Done |
| 1 | Evaluation set | 120 to 150 German Q/A pairs with ground-truth § references, as JSONL | Every reference resolves to a real record ID and `validate_eval.py` confirms it | Done: 150 written |
| 2 | Baseline retrieval | Dense-only pipeline, deliberately untuned | hit-rate@1/3/5 and MRR recorded against the eval set | Done: hit@1 0.62, hit@5 0.85, MRR 0.71 |
| 3 | Retrieval experiments | Six configurations, one results table | Table filled in and you can explain each result | Done: two significant wins. Final hit@1 0.700, MRR 0.790 at n=150 |
| 4 | Generation and judging | Cited German answers, calibrated LLM judge, local vs API comparison | Faithfulness and relevance scored, cost and p95 latency reported for both models | 4.1 done, 4.2 next |
| 5 | Serving | FastAPI, Docker Compose, thin frontend, structured logging, deployed | Public URL resolves and every request is logged with query, retrieved IDs, scores, tokens, latency | 1 wk |
| 6 | Write-up | README in English, German screenshots and queries | Results table leads the README and limitations are stated plainly | 3 d |

Phase 1 comes before any retrieval code on purpose. Build the pipeline first and you will tune it against your own impressions, and by the time an eval exists you have already committed to choices you cannot defend with numbers.

## Phase 1: eval set composition

| Question type | Share | Why it is in the set |
| --- | --- | --- |
| Direct lookup | ~50% | One section answers cleanly. Establishes the floor. |
| Multi-section | ~25% | Answer needs two or more provisions combined. Tests recall past rank 1. |
| Layperson phrasing | ~25% | Tenant wording with no legal vocabulary. Tests the gap between everyday and statutory German. |

Source the questions from situations you can state concretely: deposit returns, rent reduction after a defect, notice periods, operating cost statements, pets, subletting.

## Phase 3: retrieval experiments

| Experiment | What it tests | Expectation |
| --- | --- | --- |
| Dense only (baseline) | Semantic matching alone | Mediocre. German compounds are where it breaks. |
| BM25 only | Lexical matching on compound nouns | Strong on exact terms, weak on paraphrase |
| Hybrid, reciprocal rank fusion | Whether the two are complementary | Clear gain over either alone |
| Hybrid plus cross-encoder rerank | Precision from a second-stage model | Best hit-rate@1, at a latency cost |
| Absatz vs whole-section chunks | Whether the structural boundary matters | Absatz wins on precision, section on recall |
| With and without breadcrumb prefix | Whether hierarchy context helps embedding | Largest effect per line of code changed |
| Compound splitting in query expansion | Whether decompounding recovers lexical matches | Helps BM25, little effect on dense |

The breadcrumb ablation is the one to run first. BGB § 573c Abs. 1 states a notice period without containing the words Mietvertrag, Wohnraum, or Frist. A query like "Kündigungsfrist Wohnung" has almost nothing to match in the raw Absatz text, so the size of the gain from prefixing structural context is a real finding rather than a tuning detail.

## Phase 4: generation and judging

| Step | What it produces | State |
| --- | --- | --- |
| 4.1 Cited answers | German answers constrained to the retrieved sections, with every citation checked against them | Done: running on Qwen 3.5 9B, 9 s per question, 21 unit tests |
| 4.2 Calibration set | 30 answers graded by hand, before any judge sees them | Done: 30 graded, majority-class baselines 53/53/50 |
| 4.3 LLM judge | Faithfulness and relevance, agreement against the hand grades reported | Done, and the answer is no. The 9B judge scores below the guessing baseline on faithfulness, so it is not used |
| 4.4 Local vs API | Same answers from both backends, quality and cost and p95 latency side by side | Local side done. API side needs a key, and would also supply the stronger judge 4.3 wants |

Citation checking is deliberately separate from the judge. It answers a narrower
question, whether the answer cites a § that was actually retrieved, and it
answers it deterministically, for free, and before the judge exists. A judge is
needed for whether a claim is supported by the section it points at; it is not
needed to catch an invented section number, and that is the failure that matters
most in a legal setting.

Refusal is a fixed token rather than a polite sentence, so a partial refusal
cannot be scored as an answer. The eight questions no retrieval configuration
answers in the top ten are the test cases: declining those is correct.

The 30 hand grades have to be written by a person. They are the reference the
judge in 4.3 is measured against, and a reference produced by a model turns
agreement into two models agreeing, which says nothing about whether either is
right. The sample is stratified rather than random: a random 30 would contain
three or four interesting failures, too few to say anything about the judge on
the cases that matter. Agreement is therefore reported per stratum, not as one
average, for the same reason retrieval reports core and peripheral separately.

## Metrics

| Metric | Definition | Where it is reported |
| --- | --- | --- |
| hit-rate@k | Fraction of questions where a ground-truth section appears in the top k | Phases 2 and 3 |
| MRR | Mean reciprocal rank of the first correct result | Phases 2 and 3 |
| Faithfulness | Fraction of generated claims supported by the retrieved context | Phase 4 |
| Answer relevance | Whether the response addresses the question asked | Phase 4 |
| Refusal correctness | Does it decline when the corpus has no answer, instead of inventing a § | Phase 4 |
| Latency p95 | Query to complete answer, end to end | Phases 4 and 5 |
| Cost | Euros per thousand queries at observed token counts | Phase 4 |

Calibrate the judge against 30 answers you grade by hand before trusting it on the full set.

## Local vs API comparison

| | Local (Qwen 3.5 9B via Ollama) | API (Claude Haiku 4.5) |
| --- | --- | --- |
| Cost | Hardware only | $1 per Mtok in, $5 per Mtok out |
| Full eval run | Free | Under one euro |
| Reported | Quality, p95 latency, cost per 1k queries | Same |

Worth doing properly because "can this run on our own infrastructure" is a live question in most German interviews, driven by GDPR as much as by budget.

## Non-goals

| Excluded | Reason |
| --- | --- |
| Fine-tuning | Without a training set that justifies it, a LoRA here is tutorial-following and weakens the project |
| Expanding past Mietrecht before the eval works | More sections make retrieval look harder without making results more meaningful. The whole BGB is already parsed, so this is a config change later. |
| Agents and tool use | A good third project, but mixing it in dilutes the evaluation story |
| Auth, multi-user, persistence | Nothing about a portfolio project needs them |

## Compliance note

The deployed app carries a visible notice that it provides information and not legal advice. Answering legal questions for others in Germany touches the Rechtsdienstleistungsgesetz, and having thought about it reads as judgment rather than box-ticking.

State the corpus limitation in the README too: statute text only, no case law, so questions whose real answer turns on BGH rulings will retrieve the correct section and still answer incompletely.

## Stack

Chosen defaults. None are hard to change later except the vector store, where native hybrid search is what saves writing rank fusion by hand in phase 3.

| Component | Choice | Reason |
| --- | --- | --- |
| Embeddings | `bge-m3` | Strong multilingual, handles German compounds better than e5, 8k context |
| Reranker | `bge-reranker-v2-m3` | Same family, runs locally |
| Vector store | Qdrant via Docker | Dense and sparse hybrid search is native |
| Local LLM | Qwen 3.5 9B via Ollama | Strong German for its size, and 6.6 GB fits a 16 GB machine alongside the embedding model. A 14B at 9 GB does not, which is what ruled out the larger option rather than any quality argument. |
| Judge | Claude Haiku 4.5 | Cheap enough that full judge runs stay under a euro |
| Deployment | Hugging Face Spaces | Free tier, no card required |

## Cost and data status

Everything required runs at zero cost. The only money in the project is optional API usage in phase 4, capped in single-digit euros.

| Item | Status | Cost |
| --- | --- | --- |
| Corpus (BGB XML, PDF, EPUB) | On disk under `data/raw/bgb/`, 22 July 2026 revision, checksummed | Free, public domain (§ 5 UrhG) |
| Parsed records | 4,921 Absatz and 2,502 section records, regenerable from the XML | Free |
| Eval set | 150 questions written and validating | Free |
| Embeddings, reranker, vector store, local LLM | Open weights and open source, run on the local machine | Free |
| LLM judge (phase 4) | Claude Haiku 4.5, under one euro per full eval run | Optional, a few euros total |
| API side of the local vs API comparison (phase 4) | Same budget as the judge | Optional |
| Deployment (phase 5) | Hugging Face Spaces free tier | Free |

If the optional API spend is dropped entirely, phase 4 still works with a local judge; the numbers get noisier and the local vs API comparison table loses its second column, but nothing blocks. No dataset, license, GPU rental, or paid service is needed at any phase.

## Relationship to self-healing-rag

There is a second RAG project in this folder, so the two need a clear split or they read as one idea submitted twice.

| | self-healing-rag | german-legal-rag |
| --- | --- | --- |
| Question it answers | Can a pipeline catch its own bad answers and retry? | How good is retrieval, and what makes it better? |
| Corpus | Two synthetic handbook documents | 4,921 real Absätze of German statute |
| Emphasis | Architecture, LangGraph state machine, critic loop | Measurement, ablations, cost and latency |
| Evidence produced | A working control flow | A results table with numbers |

Describe this one as retrieval quality work and that one as agentic control flow. If they start converging, protect the evaluation harness here, because that is the part almost no other candidate has.

## Decisions made

| Question | Decision | Note |
| --- | --- | --- |
| Include the Betriebskostenverordnung? | No, BGB only | One corpus keeps the first build simple. The fetch script already accepts a second law, so adding it later costs one command plus a ground-truth revalidation pass. |
| Fixed application window? | None | Full scope, no compression |

## Still open

| Question | Deadline | Note |
| --- | --- | --- |
| Topic balance in the eval set | During phase 1 | Betriebskosten and Mieterhöhung have two questions each against eight for Kündigung. Worth evening out before the set is treated as final. |
| Whether phase 3 needs all seven experiments | Start of phase 3 | If the first three produce a clear story, the rest become optional rather than obligatory |
| Which backend to install first | Blocking 4.1 | Neither is on the machine yet. Ollama with Qwen 2.5 14B costs nothing and 4.4 needs it regardless, so it is the one to install first. The API side is a key and a few euros on top. |
