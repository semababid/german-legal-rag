---
title: Mietrecht-Auskunft
emoji: 📕
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Retrieval over German tenancy law, measured rather than demoed
---

# Mietrecht-Auskunft

Retrieval over German tenancy law (BGB §§ 535 to 580a). Ask a question in
German and see which provisions the system finds, with their similarity scores
and the statute text itself.

This Space is the front end of a project about measuring retrieval rather than
demonstrating it. The retrieval configuration running here is the one that won a
set of six experiments against a hand-built evaluation set of 150 questions:

| | |
| --- | --- |
| hit@1 | 0.700 |
| hit@5 | 0.900 |
| MRR | 0.790 |
| Median query | 29 ms locally |

The two changes that produced that, out of six tried, were embedding each
Absatz together with its position in the statute's hierarchy (p = 0.003) and
restricting results to the tenancy sections (p = 0.031). Cross-encoder
reranking, BM25 and hybrid fusion were all tested and none of them helped; two
were measurably harmful. Every comparison is significance tested, because a five
point difference across 150 questions is seven or eight questions and well
inside what chance produces.

## What this Space does not do

It does not write answers. The full project generates cited German answers from
the retrieved sections using a 9B model, but that model is 6.6 GB and wants a
GPU, and a model small enough to run on two cores would answer worse than the
retrieval it sits on. Shipping a weak generator and letting it speak for the
project would misrepresent it, so this deployment returns the provisions and
says so.

It also has no case law, which for many tenancy questions is where the answer
actually lives, and it is not legal advice. Answering legal questions for other
people in Germany runs into the Rechtsdienstleistungsgesetz.

## Where the work is

The evaluation harness, the retrieval experiments, the citation checking, the
hand-graded calibration set and the write-up of what did and did not work are in
the repository. The most useful thing in it is probably the record of
measurement bugs found along the way: nine so far, almost all of which made the
system look worse or better than it was until they were fixed.
