# Results

Retrieval quality on a hand-built Mietrecht evaluation set of 150 questions, each
pointing at the sections of the BGB that answer it.

## How scoring works

Two levels, both reported from the same run.

**Paragraph level** counts a hit when the retriever surfaces the section containing
a ground-truth Absatz. This is the primary metric, because the generator downstream
is handed the whole section: retrieving § 573 Abs. 1 when the labelled answer is
§ 573 Abs. 4 is a near miss, not a failure. It is also the only level at which an
Absatz index and a section index can be compared at all.

**Strict level** requires the exact Absatz. It stays published so the looser
reading cannot be quoted on its own.

Results are also split into **core** topics, the sections a tenant actually asks
about, and **peripheral** ones. The set deliberately covers both, but real traffic
would concentrate on the core, and the two behave differently enough that a single
average describes neither.

hit@k is the fraction of questions with a correct section in the top k. MRR is the
mean reciprocal rank of the first correct hit. Latency is the median over the run
on an Apple M5.

## Final configuration

bge-m3 embeddings over Absatz-level chunks, each embedded with its structural
breadcrumb, restricted to §§ 535 to 580a. No second retriever, no reranking.

| Level | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| Paragraph | 0.700 | 0.873 | 0.900 | 0.947 | 0.790 |
| Strict | 0.627 | 0.833 | 0.867 | 0.927 | 0.737 |

| Breakdown (paragraph level) | n | hit@1 | MRR |
| --- | --- | --- | --- |
| core topics | 76 | 0.737 | 0.816 |
| peripheral topics | 74 | 0.662 | 0.763 |
| direct | 73 | 0.753 | 0.840 |
| multi_section | 43 | 0.814 | 0.875 |
| layperson | 34 | 0.441 | 0.575 |

29 ms per query.

```bash
python src/index_corpus.py --granularity absatz --with-context
python src/eval_retrieval.py --retriever domain --collection absatz_ctx --label v150-domain
```

## All configurations

Paragraph level, with strict shown alongside.

| Run | hit@1 | hit@3 | hit@5 | hit@10 | MRR | strict hit@1 | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.247 | 0.353 | 0.400 | 0.520 | 0.321 | 0.187 | 4 |
| Hybrid, reciprocal rank fusion | 0.447 | 0.627 | 0.680 | 0.787 | 0.553 | 0.380 | 31 |
| Dense, raw Absatz text | 0.533 | 0.740 | 0.800 | 0.853 | 0.646 | 0.487 | 26 |
| Rerank on top of context | 0.620 | 0.820 | 0.887 | 0.940 | 0.730 | 0.573 | 1946 |
| Dense + breadcrumb context | 0.660 | 0.840 | 0.867 | 0.933 | 0.755 | 0.593 | 25 |
| **Dense + context + domain scope** | **0.700** | **0.873** | **0.900** | **0.947** | **0.790** | **0.627** | **29** |

## How much of this is real

McNemar's exact test on hit@1 looks only at questions where two runs disagree. The
bootstrap interval on the paired MRR difference uses the whole ranking, so it
detects smaller consistent shifts.

```bash
python src/compare_runs.py --all
python src/compare_runs.py --all --level strict
```

| Comparison | hit@1 | McNemar p | 95% CI of MRR diff | Verdict |
| --- | --- | --- | --- | --- |
| Add breadcrumb context | 0.533 to 0.660 | 0.0034 | [+0.062, +0.157] | real on both |
| BM25 instead of dense | 0.533 to 0.247 | 0.0000 | [-0.386, -0.264] | real on both, harmful |
| Add domain scoping | 0.660 to 0.700 | 0.0312 | [+0.014, +0.056] | real on both |
| Hybrid RRF | 0.533 to 0.447 | 0.0533 | [-0.152, -0.035] | real on MRR, harmful |
| Absatz instead of section | 0.613 to 0.660 | 0.248 | [-0.002, +0.082] | not distinguishable |
| Cross-encoder reranking | 0.660 to 0.620 | 0.392 | [-0.062, +0.030] | not distinguishable |

Domain scoping crosses into significance at paragraph level (p = 0.031) but not at
strict level (p = 0.0625). The reason is arithmetic rather than substantive: it
fixes 6 questions and breaks 0 at paragraph level against 5 and 0 at strict, and
five one-way discordant pairs cannot produce a p below 0.0625 whatever happens.
That is the floor of the test at that count.

The chunk-size comparison is only run at paragraph level. A section index returns
section ids, which never string-match Absatz-level ground truth, so a strict run of
it would measure the id format instead of retrieval quality.

## Breadcrumb context, the change that worked

Same model, same corpus, same queries. The only difference is what goes into the
embedding: the structural breadcrumb and section title, then the text. It moves 29
questions up and 10 down.

§ 573c Abs. 1 shows why. Its text gives a notice period without containing
Mietvertrag, Wohnraum, Frist, or Kündigung. Embedded alone it is a paragraph about
counting days. Embedded under its breadcrumb it is recognisably a rule about notice
periods in residential tenancies.

The obvious worry is that the questions were written while looking at section
titles, which would make the gain an artifact of vocabulary overlap rather than a
property of the method. Tested directly: the gain is +0.116 on the 43 questions
that share a content word with the gold section title, and +0.112 on the 107 that
share none. The effect does not depend on the overlap, and median title overlap is
zero across all three question types.

## Domain scoping

Wrong top-1 hits were landing in Sachenrecht and Familienrecht. The corpus is the
whole BGB and only 6% of it (295 of 4,921 records) is tenancy law, so a vague
question can drift out of scope entirely.

Restricting results to §§ 535 to 580a costs about 3 ms and fixes 6 questions without
breaking any. It narrows what the system can answer, which is the intended trade: a
tenancy assistant that silently returns a property-law section is worse than one
that stays in scope.

The same 6% figure means a naive post-filter starves. Fetching 50 candidates yields
roughly 3 in scope, so the filter escalates its fetch until it has enough rather
than returning fewer results than asked for.

## BM25 and hybrid fusion

BM25 is worse than dense retrieval at every rank, fixing 5 questions and breaking
48. Layperson phrasings are where it collapses, because questions written the way a
tenant would ask share almost no literal vocabulary with the statute. Lexical
scoring has nothing to match on.

This contradicts the expectation I started with, that BM25 would win on German
compound nouns. It would, if the questions reused the exact compound from the
section. They mostly do not, because they were written as natural questions rather
than copied from the statute.

Fusing the two is also harmful. Reciprocal rank fusion weights its inputs equally,
which is right when the retrievers are comparable. Here they are not, so giving the
weaker one an equal vote pulls correct results down. A weighted fusion discounting
BM25 would recover most of the loss, but tuning that weight against the same
questions used to report the result would be fitting to the eval set rather than
measuring on it. Left alone deliberately.

## Cross-encoder reranking, and a mistake worth keeping

The top 25 candidates were rescored with `BAAI/bge-reranker-v2-m3`.

The first attempt scored each candidate's raw Absatz text. That was wrong, and the
error is instructive: the candidates were found using context-enriched embeddings,
so reranking on bare text discards the breadcrumb that made them findable. Feeding
the reranker the same text the retriever matched on recovered part of the gap.

Done correctly it still does not help: it fixes 14 questions and breaks 20, p =
0.392, and the MRR interval spans zero. Interestingly it is the best configuration
on direct questions (0.849 at hit@1) and among the worst on layperson ones, so it
sharpens what is already close and cannot rescue what is not.

Latency settles it without statistics. Reranking 25 candidates costs about 1,946 ms
against 29 ms, roughly 67 times slower, for no measurable overall gain.

## Chunk size

4,921 Absatz records against 2,486 whole sections, scored at paragraph level.

| Chunk | Records | hit@1 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| Absatz | 4,921 | 0.660 | 0.867 | 0.755 |
| Whole section | 2,486 | 0.613 | 0.873 | 0.716 |

Absatz looks better and the direction is consistent, but at p = 0.248 with an MRR
interval touching zero it is not established. An earlier version of this file
reported the difference as real; that was measured on 68 questions and did not
survive the larger set. The project uses Absatz chunks anyway, because they are what
the citation-level answers in later phases need, but the retrieval evidence for that
choice is weak and should not be presented as settled.

## What still fails

Layperson phrasing, clearly and consistently: 0.441 at hit@1 against 0.753 for
direct questions, even at paragraph level. Everyday German with no legal vocabulary
is where this system is worst, and nothing tried here fixed it.

The failure mode is that the query's most salient noun is often incidental to the
legal question. Asking whether full rent is owed when the heating has been broken
for two weeks returns sections about heating installations, because the question
says Heizung while the answer is about rent reduction for defects. Indexing and
ranking cannot repair that, since the retriever is matching the wrong concept
faithfully. Rewriting the query into statutory language before retrieval is the
obvious next lever and is out of scope for this phase.

Eight questions are not answered in the top ten by any configuration tested. They
are kept in the set rather than removed, and they double as the refusal test cases
for the generation phase: a system that declines to answer those is behaving
correctly.

## A note on the evaluation set

Three questions originally had ground truth that was unanswerable by construction.
They asked whether a contract clause was valid, and the labelled answer was a
non-derogation provision whose text ("Eine zum Nachteil des Mieters abweichende
Vereinbarung ist unwirksam") appears verbatim in 34 other sections of the BGB. No
embedding can separate identical strings. Those three now also list the substantive
rule the clause protects, which genuinely answers the question and is retrievable.

Five other questions have a duplicated provision among their ground truth but also
a distinctive one, so they were already scoring correctly and were left alone.

## What actually mattered

| Change | Direction | Holds up? |
| --- | --- | --- |
| Use BM25 instead of dense | much worse | yes, p < 0.0001 |
| Fuse dense and BM25 with RRF | worse | on MRR |
| Add a cross-encoder reranker | no measurable change, 67x slower | not distinguishable |
| Use whole sections instead of Absätze | slightly worse | not distinguishable |
| Index the structural breadcrumb with the text | better | yes, p = 0.003 |
| Restrict results to the tenancy sections | better | yes, p = 0.031 |

Two changes out of six help, and both are cheap and structural. One changes what is
stored, the other filters what is returned, and neither adds a model to the request
path. Two of the rest are measurably harmful and two cannot be distinguished from
doing nothing.

The shipped configuration is among the simplest tested and also the fastest.

## Setup

Model `BAAI/bge-m3`, 1024-dimensional, cosine similarity, on Apple Silicon (MPS).
Reranking used `BAAI/bge-reranker-v2-m3`. Corpus is 4,921 Absatz records from the
BGB revision of 22 July 2026. Qdrant runs in local on-disk mode, no server needed.
Exact package versions are pinned in `requirements.txt`, since an embedding model
that shifts under a loose range would change these numbers without any code change.
