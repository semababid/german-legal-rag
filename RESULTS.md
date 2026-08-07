# Results

Retrieval quality on a hand-built Mietrecht evaluation set. hit@k is the fraction
of questions with a ground-truth section in the top k. MRR is the mean reciprocal
rank of the first correct hit. Latency is the median over the whole run on an
Apple M5.

The eval set grew from 60 to 68 questions partway through, after error analysis
showed which failure modes were under-represented. Runs are grouped by which set
they were scored on, because numbers across the two are not comparable.

## Final configuration

bge-m3 embeddings over Absatz-level chunks, each embedded with its structural
breadcrumb, restricted to §§ 535 to 580a. No second retriever, no reranking.

| | hit@1 | hit@3 | hit@5 | hit@10 | MRR | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| Final | 0.735 | 0.882 | 0.912 | 0.941 | 0.815 | 27 |

```bash
python src/index_corpus.py --granularity absatz --with-context
python src/eval_retrieval.py --retriever domain --collection absatz_ctx --label domain-scoped
```

## Experiments, 60-question set

| Run | hit@1 | hit@3 | hit@5 | hit@10 | MRR | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| Dense baseline (raw Absatz text) | 0.617 | 0.767 | 0.850 | 0.900 | 0.709 | 26 |
| BM25 | 0.333 | 0.483 | 0.550 | 0.650 | 0.427 | 4 |
| Hybrid, reciprocal rank fusion | 0.500 | 0.667 | 0.733 | 0.817 | 0.601 | 32 |
| Rerank on context, raw pair text | 0.683 | 0.867 | 0.933 | 0.950 | 0.787 | 1551 |
| Rerank on context, matched pair text | 0.700 | 0.900 | 0.917 | 0.983 | 0.798 | 1984 |
| **Dense + breadcrumb context** | **0.767** | **0.917** | **0.933** | **0.983** | **0.850** | **26** |

## Experiments, 68-question set

| Run | hit@1 | hit@3 | hit@5 | hit@10 | MRR | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| Dense + breadcrumb context | 0.691 | 0.853 | 0.868 | 0.941 | 0.780 | 26 |
| **Dense + context + domain scope** | **0.735** | **0.882** | **0.912** | **0.941** | **0.815** | **27** |

## Breadcrumb context, the change that worked

Same model, same corpus, same queries. The only difference is what goes into the
embedding: the structural breadcrumb and section title, then the text.

| Type | n | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 32 | 0.844 | 0.969 | 1.000 | 1.000 | 0.912 |
| multi_section | 16 | 0.875 | 1.000 | 1.000 | 1.000 | 0.938 |
| layperson | 12 | 0.417 | 0.667 | 0.667 | 0.917 | 0.565 |

Overall hit@1 rises from 0.617 to 0.767 and MRR from 0.709 to 0.850. Per question,
21 improved, 34 were unchanged, 5 slipped by one rank.

The gain lands where the baseline was weakest. Multi-section questions double at
rank 1, from 0.438 to 0.875. That fits the mechanism: a question spanning three
sections needs each of them to look relevant alone, and an Absatz stripped of its
heading often does not.

§ 573c Abs. 1 is the clearest case. Its text gives a notice period without
containing Mietvertrag, Wohnraum, Frist, or Kündigung. Embedded alone it is a
paragraph about counting days. Embedded under its breadcrumb it is recognisably a
rule about notice periods in residential tenancies.

The breadcrumb comes from the document's own structure, not from the eval
questions, so no answer leaks into the index. Part of the gain does come from
section titles carrying topic vocabulary the Absatz body omits, which is the
mechanism rather than a confound.

## Domain scoping

Error analysis showed 6 of 21 wrong top-1 hits coming from outside tenancy law
entirely, four from Sachenrecht and two from Familienrecht. The corpus is the whole
BGB, so a vague tenancy question can drift into property or family law.

Restricting results to §§ 535 to 580a lifts hit@1 from 0.691 to 0.735 and MRR from
0.780 to 0.815, for about 1 ms of extra latency. It is a filter over candidates the
retriever already produced, so there is no model involved.

This narrows what the system can answer, which is the intended trade. A tenancy
assistant that silently returns a property-law section is worse than one that stays
in scope.

## BM25, and why it lost

| Type | n | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 32 | 0.438 | 0.688 | 0.719 | 0.750 | 0.552 |
| multi_section | 16 | 0.375 | 0.438 | 0.562 | 0.750 | 0.460 |
| layperson | 12 | 0.000 | 0.000 | 0.083 | 0.250 | 0.049 |

BM25 loses on every type, and on layperson questions it fails outright: no correct
section in the top three for any of the twelve. Those questions are written the way
a tenant would ask, so they share almost no literal vocabulary with the statute.
Lexical scoring has nothing to match on.

This contradicts the expectation I started with, that BM25 would win on German
compound nouns. It would, if the questions reused the exact compound from the
section. They mostly do not, because they were written as natural questions rather
than copied from the statute.

## Hybrid fusion

Before building it, worth checking whether the two retrievers even find different
things, since fusion can only reorder what one of them already found.

| Outcome across 60 questions | Count | Share |
| --- | --- | --- |
| Both find a gold section in the top 10 | 38 | 63% |
| Dense only | 16 | 27% |
| BM25 only | 1 | 2% |
| Neither | 5 | 8% |

The union ceiling for hit@10 is 0.917 against dense's 0.900. On rank quality dense
is better on 33 questions, BM25 on 3.

The run confirmed the prediction: hybrid scores 0.500 at hit@1 against dense's
0.617. Reciprocal rank fusion weights its inputs equally, which is right when the
retrievers are comparable. Here they are not, so giving the weaker one an equal vote
pulls correct results down the list.

A weighted fusion discounting BM25 would recover most of the loss, but tuning that
weight against the same questions used to report the result would be fitting to the
eval set rather than measuring on it. Left alone deliberately.

## Cross-encoder reranking, and a mistake worth keeping

The top 25 candidates were rescored with `BAAI/bge-reranker-v2-m3`.

The first run scored each candidate's raw Absatz text. That was wrong, and the error
is instructive: the candidates were found using context-enriched embeddings, so
reranking on bare text discards the breadcrumb that made them findable. Feeding the
reranker the same text the retriever matched on moved hit@1 from 0.683 to 0.700.

It still loses to no reranking at all, which scores 0.767. Reranking improved the
rank of 7 questions and worsened 12, with ev-052 dropping from 1 to 5. Layperson
questions fell from 0.417 to 0.333, the opposite of the intent.

There is very little left to fix. Dense plus context already places a gold section
in the top three for 92% of questions. A general-purpose multilingual reranker
reordering an already good list of short, formulaic statutory paragraphs mostly
introduces noise, and it has no German legal tuning.

Latency settles it: 1,984 ms against 26 ms, about 76 times slower, to lose seven
points of hit@1.

## Chunk size

The corpus parses into 4,921 Absatz records or 2,486 whole sections. Both runs are
scored at paragraph level, where a hit means the retriever surfaced the section
containing a gold Absatz, since a section index can never string-match an
Absatz-level id. That makes these Absatz numbers slightly higher than elsewhere.

| Chunk | Records | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| Absatz | 4,921 | 0.783 | 0.917 | 0.933 | 0.983 | 0.858 |
| Whole section | 2,486 | 0.683 | 0.850 | 0.933 | 0.967 | 0.788 |

Absatz wins by 10 points at hit@1. The gap closes by rank 5, so this is about
ranking precision rather than reachability. The likely mechanism is dilution: a
section such as § 543 spans several Absätze covering distinct rules, and averaging
them into one vector blurs what the section is about.

I predicted Absatz would win on precision and section on recall. The first half held
and the second did not. Section chunks are no better at hit@10, so there is no
recall argument for them either.

## Error analysis and how the eval set grew

With the winning config, five of the original 60 questions fell outside the top
three. They fall into three patterns.

**A salient noun pointing at the wrong area of law.** ev-024 asks whether full rent
is owed when the heating has been broken for two weeks. The answer is § 536 Abs. 1
on rent reduction. The top five results were all about heating installations:

```
1. § 559e Abs. 3  Mieterhöhung nach Einbau einer Heizungsanlage
2. § 559e Abs. 5  Mieterhöhung nach Einbau einer Heizungsanlage
3. § 559e Abs. 2  Mieterhöhung nach Einbau einer Heizungsanlage
4. § 556c Abs. 1  Kosten der Wärmelieferung als Betriebskosten
5. § 559e Abs. 1  Mieterhöhung nach Einbau einer Heizungsanlage
```

It matched on Heizung when the question is not about heating at all. The heating is
incidental to a question about what a tenant owes when something is broken.

**The right cluster, the wrong rule.** ev-045 asks about leaving because of an
announced modernisation. It returned § 559 Abs. 4, § 555c and § 555d, all correct
territory, but the special termination right is § 555e. ev-049 returned § 566 where
the deposit rule is the adjacent § 566a.

**Too little signal.** ev-030, "Das Haus wurde verkauft. Muss ich jetzt ausziehen?",
is short and generic enough that the retriever left tenancy law altogether.

Eight questions were then written to probe these patterns deliberately, taking the
set to 68. Five of the eight land outside the top three, so they are doing their
job. The headline number drops from 0.767 to 0.691 as a result, which is the eval
set getting harder rather than the system getting worse.

## What still fails

Ten of 68 questions sit outside the top three with the final configuration, and
seven of those ten are layperson phrasings. That is the honest weak spot: everyday
German with no legal vocabulary, where the query's most salient word is often
incidental to the legal question being asked.

None of the techniques tried here address it, because it is a query understanding
problem rather than an indexing or ranking one. Rewriting the query into statutory
language before retrieval is the obvious next thing to try, and it is out of scope
for this phase.

## What actually mattered

| Change | Effect on hit@1 |
| --- | --- |
| Add BM25 as a second retriever | worse |
| Fuse dense and BM25 with RRF | worse |
| Add a cross-encoder reranker | worse |
| Use whole sections instead of Absätze | worse |
| Index the structural breadcrumb with the text | better, 0.617 to 0.767 |
| Restrict results to the tenancy sections | better, 0.691 to 0.735 |

Four of six changes made retrieval worse. Both that worked are cheap and structural:
one changes what is stored, the other filters what is returned. Neither adds a model
to the request path. The four that failed all bolted machinery onto retrieval that
was already working, or coarsened the unit it works on.

The shipped configuration is therefore the simplest one tested, and also the
fastest at 27 ms per query.

## Setup

Model `BAAI/bge-m3`, 1024-dimensional, cosine similarity, on Apple Silicon (MPS).
Reranking used `BAAI/bge-reranker-v2-m3`. Corpus is 4,921 Absatz records from the
BGB revision of 22 July 2026. Qdrant runs in local on-disk mode, no server needed.
