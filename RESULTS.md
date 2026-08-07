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

Later phases add BM25, hybrid, reranking, the breadcrumb ablation, and the Absatz
versus section comparison, one row each.

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

## Setup

Model `BAAI/bge-m3`, 1024-dimensional, cosine similarity, run on Apple Silicon (MPS).
Corpus is 4,921 Absatz records. Qdrant runs in local on-disk mode, no server needed.
