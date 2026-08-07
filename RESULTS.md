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

## Setup

Model `BAAI/bge-m3`, 1024-dimensional, cosine similarity, run on Apple Silicon (MPS).
Corpus is 4,921 Absatz records. Qdrant runs in local on-disk mode, no server needed.
