# Results

Retrieval quality on a hand-built Mietrecht evaluation set of 150 questions, each
pointing at the sections of the BGB that answer it. hit@k is the fraction of
questions with a ground-truth section in the top k. MRR is the mean reciprocal rank
of the first correct hit. Latency is the median over the run on an Apple M5.

Every comparison below is significance tested. On a set this size a five point
difference in hit@1 is seven or eight questions, which is close enough to what
chance produces that an untested table would invite conclusions it cannot support.

## Final configuration

bge-m3 embeddings over Absatz-level chunks, each embedded with its structural
breadcrumb, restricted to §§ 535 to 580a. No second retriever, no reranking.

| | hit@1 | hit@3 | hit@5 | hit@10 | MRR | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| Final | 0.627 | 0.827 | 0.860 | 0.920 | 0.733 | 28 |

```bash
python src/index_corpus.py --granularity absatz --with-context
python src/eval_retrieval.py --retriever domain --collection absatz_ctx --label v150-domain
```

## All configurations

| Run | hit@1 | hit@3 | hit@5 | hit@10 | MRR | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.180 | 0.300 | 0.353 | 0.467 | 0.259 | 4 |
| Hybrid, reciprocal rank fusion | 0.373 | 0.560 | 0.607 | 0.740 | 0.484 | 32 |
| Dense, raw Absatz text | 0.480 | 0.680 | 0.740 | 0.800 | 0.592 | 26 |
| Rerank on top of context | 0.573 | 0.780 | 0.847 | 0.920 | 0.689 | 1984 |
| Dense + breadcrumb context | 0.593 | 0.793 | 0.827 | 0.907 | 0.703 | 26 |
| **Dense + context + domain scope** | **0.627** | **0.827** | **0.860** | **0.920** | **0.733** | **28** |

Chunk granularity is compared separately below, because it needs a different
scoring rule to be fair.

## How much of this is real

McNemar's exact test on hit@1 looks only at questions where two runs disagree,
which is the right test for paired binary outcomes. The bootstrap interval on the
paired MRR difference uses the whole ranking rather than collapsing to hit or miss,
so it detects smaller consistent shifts.

```bash
python src/compare_runs.py --all
```

| Comparison | hit@1 | McNemar p | MRR | 95% CI of MRR diff | Verdict |
| --- | --- | --- | --- | --- | --- |
| Add breadcrumb context | 0.480 to 0.593 | 0.0076 | 0.592 to 0.703 | [+0.060, +0.161] | real on both |
| BM25 instead of dense | 0.480 to 0.180 | 0.0000 | 0.592 to 0.259 | [-0.397, -0.270] | real on both |
| Hybrid RRF | 0.480 to 0.373 | 0.0139 | 0.592 to 0.484 | [-0.169, -0.049] | real on both, harmful |
| Add domain scoping | 0.593 to 0.627 | 0.0625 | 0.703 to 0.733 | [+0.013, +0.053] | real on MRR |
| Absatz instead of section | 0.613 to 0.660 | 0.248 | 0.716 to 0.755 | [-0.002, +0.082] | not distinguishable |
| Cross-encoder reranking | 0.593 to 0.573 | 0.743 | 0.703 to 0.689 | [-0.062, +0.036] | not distinguishable |

Domain scoping deserves a note. It fixes 5 questions and breaks 0, which is as
one-sided as an effect can be, but five discordant pairs cannot produce a p below
0.0625 no matter how they fall. That is the floor of the test at that count, not
evidence against the change. The MRR interval excludes zero, and the mechanism is
not in doubt: returning a property-law section for a tenancy question is wrong by
construction.

## What changed when the eval set grew

The set was 68 questions before this table was rebuilt at 150. Three verdicts moved,
in both directions, which is the clearest argument for having done it.

| Comparison | At n=68 | At n=150 |
| --- | --- | --- |
| Breadcrumb context | real on MRR only | real on both |
| Hybrid RRF | not distinguishable | real on both, harmful |
| Absatz instead of section | real on MRR only | not distinguishable |

The first two firmed up, as more data usually does. The third went the other way:
the chunking advantage that looked real at 68 questions no longer separates from
noise at 150, and its MRR interval now touches zero. The earlier result was
probably part signal and part small-sample luck, and the honest position is that
Absatz chunks are not shown to beat whole sections on this evidence.

Absolute scores also fell, from 0.735 to 0.627 at hit@1 for the same configuration.
Nothing regressed. The larger set reaches 77 distinct sections instead of 35 and
includes far more layperson phrasing, so it is simply a harder test. Numbers from
the two sets are not comparable, and only the 150-question ones are quoted here.

## Breadcrumb context, the change that worked

Same model, same corpus, same queries. The only difference is what goes into the
embedding: the structural breadcrumb and section title, then the text.

| Type | n | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 76 | 0.658 | 0.868 | 0.895 | 0.934 | 0.765 |
| multi_section | 40 | 0.675 | 0.875 | 0.900 | 0.950 | 0.783 |
| layperson | 34 | 0.353 | 0.529 | 0.588 | 0.794 | 0.468 |

It moves 27 questions up and 10 down, hit@1 from 0.480 to 0.593, MRR from 0.592 to
0.703. This is the only change tested here that is significant on both metrics and
positive.

§ 573c Abs. 1 shows why. Its text gives a notice period without containing
Mietvertrag, Wohnraum, Frist, or Kündigung. Embedded alone it is a paragraph about
counting days. Embedded under its breadcrumb it is recognisably a rule about notice
periods in residential tenancies.

The breadcrumb comes from the document's own structure, not from the eval
questions, so no answer leaks into the index. Part of the gain does come from
section titles carrying topic vocabulary the Absatz body omits, which is the
mechanism rather than a confound.

## Domain scoping

Error analysis found wrong top-1 hits landing in Sachenrecht and Familienrecht. The
corpus is the whole BGB and only 6% of it (295 of 4,921 records) is tenancy law, so
a vague question can drift out of scope entirely.

Restricting results to §§ 535 to 580a costs about 1 ms and fixes 5 questions without
breaking any. It narrows what the system can answer, which is the intended trade: a
tenancy assistant that silently returns a property-law section is worse than one
that stays in scope.

Only 6% of records being in range also means a naive post-filter starves. Fetching
50 candidates yields roughly 3 in scope, so the filter escalates its fetch until it
has enough, rather than returning fewer results than asked for.

## BM25, the clearest loser

BM25 is worse than dense retrieval by every measure and at every rank, fixing 2
questions and breaking 47. Layperson phrasings are where it collapses, because
questions written the way a tenant would ask share almost no literal vocabulary
with the statute that answers them. Lexical scoring has nothing to match on.

This contradicts the expectation I started with, that BM25 would win on German
compound nouns. It would, if the questions reused the exact compound from the
section. They mostly do not, because they were written as natural questions rather
than copied from the statute.

## Hybrid fusion

Reciprocal rank fusion of dense and BM25 is significantly worse than dense alone,
fixing 11 questions and breaking 27. RRF weights its inputs equally, which is right
when the retrievers are comparable. Here they are not, so giving the weaker one an
equal vote pulls correct results down the list.

A weighted fusion discounting BM25 would recover most of the loss, but tuning that
weight against the same questions used to report the result would be fitting to the
eval set rather than measuring on it. Left alone deliberately.

## Cross-encoder reranking, and a mistake worth keeping

The top 25 candidates were rescored with `BAAI/bge-reranker-v2-m3`.

The first attempt scored each candidate's raw Absatz text. That was wrong, and the
error is instructive: the candidates were found using context-enriched embeddings,
so reranking on bare text discards the breadcrumb that made them findable. Feeding
the reranker the same text the retriever matched on recovered part of the gap.

Done correctly it still does not help. At 150 questions it fixes 17 and breaks 20,
p = 0.743, and the MRR interval spans zero. The accuracy difference is not
established in either direction.

Latency settles it without needing statistics. Reranking 25 candidates costs about
1,984 ms against 26 ms, roughly 76 times slower, for no measurable gain.

## Chunk size

The corpus parses into 4,921 Absatz records or 2,486 whole sections. Both runs are
scored at paragraph level, where a hit means the retriever surfaced the section
containing a gold Absatz, since a section index can never string-match an
Absatz-level id.

| Chunk | Records | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Absatz | 4,921 | 0.660 | 0.840 | 0.867 | 0.933 | 0.755 |
| Whole section | 2,486 | 0.613 | 0.787 | 0.873 | 0.920 | 0.716 |

Absatz looks better and the direction is consistent, but at p = 0.248 with an MRR
interval touching zero it is not established. The project uses Absatz chunks
anyway, because they are also what the citation-level answers in later phases need,
but the retrieval evidence for that choice is weak and should not be presented as
settled.

## What still fails

Layperson phrasing, clearly and consistently. At hit@1 the final configuration gets
0.671 on direct questions, 0.750 on multi-section ones, and 0.382 on layperson
ones. Everyday German with no legal vocabulary is where this system is worst, and
nothing tried here fixed it.

The failure mode is that the query's most salient noun is often incidental to the
legal question. Asking whether full rent is owed when the heating has been broken
for two weeks returns sections about heating installations, because the question
says Heizung while the answer is about rent reduction for defects. Indexing and
ranking cannot repair that, since the retriever is matching the wrong concept
faithfully. Rewriting the query into statutory language before retrieval is the
obvious next lever and is out of scope for this phase.

## What actually mattered

| Change | Direction | Holds up? |
| --- | --- | --- |
| Use BM25 instead of dense | much worse | yes, p < 0.0001 |
| Fuse dense and BM25 with RRF | worse | yes, p = 0.014 |
| Add a cross-encoder reranker | no measurable change, 76x slower | not distinguishable |
| Use whole sections instead of Absätze | slightly worse | not distinguishable |
| Index the structural breadcrumb with the text | better | yes, p = 0.008 |
| Restrict results to the tenancy sections | better | on MRR |

One change out of six is significant, positive, and cheap. It costs nothing at
query time because it changes what is stored rather than what happens per request.
Two of the others are significantly harmful, and two cannot be distinguished from
doing nothing at all.

The shipped configuration is the simplest one tested and also the fastest.

## Setup

Model `BAAI/bge-m3`, 1024-dimensional, cosine similarity, on Apple Silicon (MPS).
Reranking used `BAAI/bge-reranker-v2-m3`. Corpus is 4,921 Absatz records from the
BGB revision of 22 July 2026. Qdrant runs in local on-disk mode, no server needed.
Exact package versions are pinned in `requirements.txt`, since an embedding model
that shifts under a loose range would change these numbers without any code change.
