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

## Generation, first run

150 answers from Qwen 3.5 9B, run locally, six retrieved sections per prompt and
reasoning switched off. Only the countable things are here. Whether an answer is
correct is what the hand-graded set and the judge are for, and putting a quality
number here before those exist would be inventing one.

| | |
| --- | --- |
| Answers | 150 |
| Cited a § named nowhere in the text shown | 2 |
| Cited a § named in the text but never shown | 15 |
| Cited nothing at all | 1 |
| Used the refusal token | 12 |
| Latency p50 / p95 | 7.9 s / 11.0 s |
| Cost | none, runs on the machine |

Those two citation rows started as one, and separating them took three
corrections to the checker rather than to the model.

The first count was 19. Two were a bug: the BGB cross-references other statutes,
so an answer quoting § 555b faithfully reproduces its reference to § 3 Nr. 33 of
the Telekommunikationsgesetz, and the checker read that as a BGB section that
was never retrieved. Citations qualified by another law's name are now skipped.

The remaining 17 looked like the model citing sections from memory, which would
be the failure a legal assistant cannot have. Fifteen of them were not that. The
statute refers to itself constantly, and § 559 Abs. 1 opens with
"Modernisierungsmaßnahmen im Sinne des § 555b", so a model shown § 559 and
citing § 555b is repeating a pointer it read. Counting those as inventions
overstated the rate eightfold.

That leaves 2 answers in 150 citing a section that appears nowhere at all. The
15 are a real but lesser problem: the model asserts what § 560 provides while
having seen only its name, so the claim is traceable to the prompt and still
unverified. They are counted separately rather than forgiven, and
`fully_verifiable` is the stricter reading that excludes them.

The refusal count needs reading carefully. Not one of the 12 is a clean refusal.
Every one cites something alongside the token, and they split into two
behaviours that look the same to any rule: answering and then caveating a
detail, and declining while citing the sections examined to explain why they do
not answer. The prompt asks for the bare token and nothing else; the model
prefers to argue its case. Those 12 go into the hand-graded set rather than
being counted either way here.

## The hand-graded set

30 of the 150 answers, graded by hand before any judge existed. Stratified
rather than random, so the interesting failures are represented: all 12 that
used the refusal token, both that cited a section named nowhere, 8 of the 15
that cited a cross-reference, and 8 whose citations were mechanically perfect.

| Judgment | yes | partly | no |
| --- | --- | --- | --- |
| Every claim supported by the sections shown | 11 | 16 | 3 |
| Answers the question asked | 16 | 2 | 12 |
| Declining was right (the 12 token cases) | 4 | 2 | 6 |

The grader found "no" clear to apply and the line between "yes" and "partly"
frequently not, so the split between the first two columns carries more
uncertainty than the third. What that does and does not affect is worked through
under the judge results below.

The middle column is where the answers live. Outright invention is rare and
complete faithfulness is not the norm either: two thirds of answers mix
supported claims with unsupported ones, which is the failure mode a citation
count cannot see.

### The citation check does not measure faithfulness

Of the 8 answers where every citation pointed at text the model had actually
read, 4 were judged fully faithful. The mechanical check is right about half the
time as a proxy for the thing it looks like it measures.

The check is not broken. It answers a narrower question than its name suggests:
did this answer cite a section it was never shown. An answer can clear that bar
and still claim something the cited section does not say. Catching that means
reading the answer against the text, which is why these 30 were graded by a
person.

### Refusal is the weak spot

Every one of the 12 answers that reached for the refusal token was judged not to
have answered the question, and 6 of those 12 declined when the sections in
front of them did contain the answer. So the model gives up about twice as often
as it should. A wrong answer is easier to notice than an unnecessary refusal,
which is what makes this one worth measuring rather than assuming.

### What this set cannot support

One grader, 30 items, no second annotator, so there is no inter-annotator
agreement figure and no way to separate a judge disagreeing with the grader from
a judge disagreeing with the truth. The counts are small enough that the
per-stratum splits are indicative and not much more: 4 against 4 in a stratum of
8 would move to 5 against 3 on one changed mind.

What it does support is the thing it was built for. A judge that guesses the
majority class every time scores 53% on faithfulness, 53% on relevance and 50%
on refusal, so a judge worth trusting has to beat those and can be seen not to.

## The judge, and why it is not being used

A model was asked to grade the same 30 answers on the same three points, shown
exactly what the human grader saw and nothing else. Agreement was then measured
against the hand grades. Kappa is agreement above chance: 0 means a judge that
learned nothing, 1 means perfect. The baseline column is what a judge scores by
guessing the commonest human grade every time.

| Judge | | exact | within one | kappa | baseline |
| --- | --- | --- | --- | --- | --- |
| v1 | faithful | 33% | 87% | 0.04 | 53% |
| v1 | relevant | 77% | 83% | 0.54 | 53% |
| v1 | refusal (n=4) | 50% | 50% | 0.00 | 50% |
| v2 | faithful | 43% | 90% | 0.17 | 53% |
| v2 | relevant | 67% | 77% | 0.33 | 53% |
| v2 | refusal (n=10) | 40% | 80% | 0.12 | 40% |

Qwen 3.5 9B cannot grade faithfulness. Both versions score below the baseline,
so on this evidence a judge that ignored the answer entirely and always guessed
"partly" would be more accurate.

The cause is visible without reference to the human grades at all. v1 returned
yes 23 times, no 7 times and partly not once across 30 answers, against a human
who used the middle grade for 16 of 30. Two thirds of a three-point scale cannot
match a grader who lives in the middle of it. v1 was adjacent-correct 87% of the
time, which is the same finding from the other side: the judge knows roughly
where an answer sits and will not say so.

v2 defined each grade explicitly, including a sentence saying that a mix of
supported and unsupported claims is the normal case rather than the exception.
It moved the middle grade from 0 uses to 2. The refusal field went from 4
verdicts to 10 after being made mandatory when the token is present, which is
the one clear improvement. Relevance fell from 0.54 to 0.33, though at n=30 that
is three answers and inside the noise.

The prompt was revised once, and the reason is worth stating because the
distinction is easy to lose. A judge emitting zero "partly" verdicts is a defect
readable in the judge's own output, with the human grades covered up, so fixing
it is not tuning against them. Measuring the fix on the same 30 answers is still
optimistic, and no further revision was made: iterating a prompt until the
agreement number rises fits the judge to the calibration set and produces a
figure that says nothing about the other 120 answers.

### Was it the grader or the judge?

The grader reported afterwards that "no" felt clear every time, while the line
between "yes" and "partly" often did not. That is worth checking rather than
noting, because if most of the disagreement sat on a boundary the human was
unsure of, the judge would be taking the blame for the scale.

Merging that boundary, both directions, on faithfulness:

| Collapse | v1 kappa | v2 kappa | above baseline |
| --- | --- | --- | --- |
| three-point, as graded | +0.04 | +0.17 | no |
| is it broken (yes+partly vs no) | +0.07 | +0.07 | no |
| is it perfect (yes vs partly+no) | +0.07 | +0.28 | no |

The judge stays at or under the baseline under every reading of the scale. Had
the disagreement been about where the middle grade starts, merging the two
grades would have shown it, and it does not. The conclusion survives the
grader's own doubt about the grades it rests on, which is a stronger position
than it started in. The check runs on every comparison rather than being
remembered as a one-off.

The doubt still bites elsewhere. Any figure that turns on the yes-against-partly
line inherits it: the 11 fully faithful against 16 partly above, and the finding
that 4 of 8 mechanically perfect answers were fully faithful. Both would move if
the boundary moved. The count of 3 answers with essentially nothing supported is
the one that does not, since that grade is the one the grader trusted. A second
annotator would settle it and there was not one.

Relevance under v1 is the one usable result: 0.54 is moderate agreement, clear
of the baseline, on the simplest of the three judgments. It is not being quoted
as the judge's score, because picking v1 for relevance and v2 for refusal is
choosing the flattering half of each run after seeing both.

So the answers in this project are not judged by a model. Faithfulness on the
other 120 stays unmeasured rather than measured wrongly. Without the 30 hand
grades the natural next step would have been to run the judge across all 150 and
publish the scores, and nothing in that output would have revealed that the
judge was worse than guessing.

Two routes out, neither taken here. A stronger judge, which is what the API side
of the comparison was for and needs a key. Or a second hand-graded sample, held
out, which is the only way to get an honest number for a revised prompt.

## Setup

Model `BAAI/bge-m3`, 1024-dimensional, cosine similarity, on Apple Silicon (MPS).
Reranking used `BAAI/bge-reranker-v2-m3`. Corpus is 4,921 Absatz records from the
BGB revision of 22 July 2026. Qdrant runs in local on-disk mode, no server needed.
Exact package versions are pinned in `requirements.txt`, since an embedding model
that shifts under a loose range would change these numbers without any code change.
