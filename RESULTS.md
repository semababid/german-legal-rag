# Results

Retrieval quality on the 60-question Mietrecht eval set. Each run scores the same
questions, so the rows are directly comparable. hit@k is the fraction of questions
where a ground-truth section is in the top k; MRR is the mean reciprocal rank of
the first correct hit.

Reproduce a row with:

```bash
python src/index_corpus.py --granularity absatz        # build the index
python src/eval_retrieval.py --collection absatz_raw --label dense-baseline
```

## Overall

| Run | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| Dense baseline (bge-m3, Absatz, raw text) | 0.617 | 0.767 | 0.850 | 0.900 | 0.709 |
| BM25 (Absatz, raw text) | 0.333 | 0.483 | 0.550 | 0.650 | 0.427 |
| Hybrid, reciprocal rank fusion | 0.500 | 0.667 | 0.733 | 0.817 | 0.601 |
| **Dense + breadcrumb context** | **0.767** | **0.917** | **0.933** | **0.983** | **0.850** |

Later rows add hybrid fusion, reranking, the breadcrumb ablation, and the Absatz
versus section comparison.

## Dense baseline by question type

| Type | n | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 32 | 0.781 | 0.875 | 0.938 | 0.938 | 0.839 |
| multi_section | 16 | 0.438 | 0.812 | 0.875 | 1.000 | 0.632 |
| layperson | 12 | 0.417 | 0.417 | 0.583 | 0.667 | 0.468 |

The split is where the interesting part is.

Direct questions, where the wording sits close to the statute, retrieve well. The
two weak spots each point at a specific fix:

- **Layperson phrasing is the hardest.** Everyday questions with no legal vocabulary
  land far from the formal wording of the section that answers them. A tenant asking
  whether a dripping pipe has to be reported does not phrase it as an Anzeigepflicht.
  This is the gap the hybrid and breadcrumb experiments are meant to close.
- **Multi-section questions miss at rank 1 but not at rank 10.** hit@1 is 0.44 while
  hit@10 is a perfect 1.00, so the right sections are in the pool, just not at the
  top. That is the case reranking is built for.

## BM25 by question type

| Type | n | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 32 | 0.438 | 0.688 | 0.719 | 0.750 | 0.552 |
| multi_section | 16 | 0.375 | 0.438 | 0.562 | 0.750 | 0.460 |
| layperson | 12 | 0.000 | 0.000 | 0.083 | 0.250 | 0.049 |

BM25 loses to the dense baseline on every type, and on layperson questions it fails
outright: no correct section in the top three for any of the twelve, and an MRR of
0.049. Those questions are written the way a tenant would actually ask, so they
share almost no literal vocabulary with the statute that answers them. Lexical
scoring has nothing to match on.

This contradicts the expectation I started with, which was that BM25 would win on
German compound nouns. It would, if the questions reused the exact compound from
the section text. They mostly do not, because they were written as natural
questions rather than copied from the statute. The compound-noun advantage is real
but it needs a query that contains the compound.

## Are dense and BM25 complementary?

Worth checking directly before building a hybrid, since fusion can only reorder
what at least one retriever already found.

| Outcome across the 60 questions | Count | Share |
| --- | --- | --- |
| Both find a gold section in the top 10 | 38 | 63% |
| Dense only | 16 | 27% |
| BM25 only | 1 | 2% |
| Neither | 5 | 8% |

The union ceiling for hit@10 is 0.917 against dense's 0.900. BM25 contributes one
question that dense misses (ev-050, on the tacit renewal of a tenancy under § 545).
On rank quality dense is better on 33 questions, BM25 on 3.

So hybrid fusion has almost nothing to add here. It is still worth running as a
measured row rather than an assumption, but the honest expectation is a marginal
change, not a gain. The 8% that neither retriever finds is the more useful target,
and reranking cannot help there either, because those sections never enter the
candidate pool at all.

## Hybrid fusion, and why it lost

| Run | hit@1 | hit@5 | MRR |
| --- | --- | --- | --- |
| Dense | 0.617 | 0.850 | 0.709 |
| Hybrid (RRF) | 0.500 | 0.733 | 0.601 |
| BM25 | 0.333 | 0.550 | 0.427 |

Fusing the two retrievers is worse than using the better one alone, on every metric
and every question type. The complementarity table above predicted this, and the
run confirms it.

Reciprocal rank fusion weights its inputs equally. That is the right default when
the retrievers are of comparable quality, because it lets agreement between them
outvote any single confident but wrong result. Here they are not comparable: dense
wins on rank for 33 questions against BM25's 3. Giving a much weaker retriever an
equal vote pulls correct dense results down the list, and BM25 contributes only one
question dense did not already find. The arithmetic does not work out.

A weighted fusion that discounted BM25 would recover most of the loss, but tuning
that weight against the same 60 questions used to report the result would be
fitting to the eval set rather than measuring on it. Left alone deliberately.

The useful conclusion is that on this corpus the ranking is not the bottleneck, the
candidate pool is. That points at what gets indexed rather than how results are
reordered, which is the breadcrumb ablation below.

## Breadcrumb context, the change that worked

Same model, same corpus, same queries. The only difference is what goes into the
embedding: instead of the Absatz text alone, each record is embedded as its
structural breadcrumb and section title followed by the text.

| Type | n | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 32 | 0.844 | 0.969 | 1.000 | 1.000 | 0.912 |
| multi_section | 16 | 0.875 | 1.000 | 1.000 | 1.000 | 0.938 |
| layperson | 12 | 0.417 | 0.667 | 0.667 | 0.917 | 0.565 |

Against the raw-text baseline, overall hit@1 goes from 0.617 to 0.767 and MRR from
0.709 to 0.850. Per question, 21 improved, 34 were unchanged, and 5 slipped by a
single rank position.

The effect is concentrated where the baseline was weakest. Multi-section questions
double at rank 1, from 0.438 to 0.875, and reach 1.000 by rank 3. That fits the
mechanism: a question spanning three sections needs each of them to look relevant
on its own, and an Absatz stripped of its heading often does not.

§ 573c Abs. 1 is the clearest case. Its text gives a notice period without
containing Mietvertrag, Wohnraum, Frist, or Kündigung anywhere. Embedded alone it
is a paragraph about counting days. Embedded under its breadcrumb and title it is
recognisably a rule about notice periods in residential tenancies.

Worth being precise about what this is. The breadcrumb and title come from the
document's own structure, not from the eval questions, so nothing about the answers
leaks into the index. It does mean part of the gain comes from section titles
carrying topic vocabulary the Absatz body omits, which is the mechanism, not a
confound.

## What still fails

Only one question has no gold section in the top ten, and four of the five worst
are layperson phrasings.

| Question | Type | Rank |
| --- | --- | --- |
| ev-024 heating broken two weeks, still pay full rent? | layperson | not found |
| ev-025 partner wants to move in, landlord refuses | layperson | 9 |
| ev-030 house was sold, do I have to move out? | layperson | 6 |
| ev-045 large modernisation announced, can I leave? | layperson | 6 |
| ev-049 what happens to my deposit if the flat is sold | direct | 5 |

ev-024 is worth reading closely, because it shows the limit of the approach. The
gold section is § 536 Abs. 1, on reducing rent when the property is defective. The
top five results are all about heating systems:

```
1. § 559e Abs. 3  Mieterhöhung nach Einbau einer Heizungsanlage
2. § 559e Abs. 5  Mieterhöhung nach Einbau einer Heizungsanlage
3. § 559e Abs. 2  Mieterhöhung nach Einbau einer Heizungsanlage
4. § 556c Abs. 1  Kosten der Wärmelieferung als Betriebskosten
5. § 559e Abs. 1  Mieterhöhung nach Einbau einer Heizungsanlage
```

The retriever matched on Heizung, the concrete noun in the question, when the
question is not about heating at all. It is about what a tenant owes when something
is broken, and the heating is incidental. Adding structural context cannot fix this,
because the failure is that the query's most salient word points at the wrong area
of law. That is a query understanding problem, not an indexing one.

## Setup

Model `BAAI/bge-m3`, 1024-dimensional, cosine similarity, run on Apple Silicon (MPS).
Corpus is 4,921 Absatz records. Qdrant runs in local on-disk mode, no server needed.
