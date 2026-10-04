# Eval glossary

Plain definitions for the vocabulary used in the inference docs (`docs/evaluation.md`, `docs/reranker-selection.md`, `eval/README.md`) and `LORA.md`. What the words mean, not how to compute them.

## Statistics

**p** — a probability. The symbol means different things by context: in a binomial test p = 0.5 is the win chance under the null; a p-value is the chance of data at least this extreme under the null.

**p-value** — the probability of a result at least this lopsided if the null hypothesis were true. Not the probability that the null is true, and not the probability that the difference is "real."

**α (significance level)** — the false-positive rate you accept before running the test; you reject the null when p < α. Convention: 0.05. Same symbol, three different jobs: stats α, LoRA's alpha (a scaling factor), hybrid retrieval's alpha (a blend weight).

**Null hypothesis** — the default of no difference: every discordant case is a coin flip.

**Type I error** — a false positive: declaring a difference that isn't there. Its rate is α.

**Type II error** — a false negative: missing a difference that is there. Its rate falls as power rises.

**Power** — the chance of detecting a real difference of a given size. More cases and bigger effects raise it. One case of 500 moves recall by 0.002, so a small suite cannot see a small effect.

**Effect size** — how big the difference is in its own units (e.g. 5.5 points of recall@1). Significance says "probably not zero"; effect size says "worth acting on."

**Confidence interval** — a range that would contain the true value in X% of repeated experiments. The Wilson interval is the version that behaves at small n and near 0/1.

**Multiple comparisons** — testing many models inflates the chance of a lucky p. Bonferroni divides α by the number of tests. FDR (Benjamini–Hochberg) accepts a controlled fraction of false positives among discoveries — less conservative.

**Discordant / concordant pairs** — in a paired comparison, cases where the two models disagree (discordant) or agree (concordant). McNemar uses only the discordant ones; concordant pairs cancel.

**McNemar's test** — the paired test for two models on the same cases with a binary outcome. Chi-squared form: χ² = (|b − c| − 1)² / (b + c). Exact form: a binomial test on b out of b + c with p = 0.5. It answers: are the disagreements lopsided?

**Continuity correction** — the −1 in the chi-squared form; nudges the approximation toward exact at small counts. The exact test doesn't need it.

**Binomial test** — the test behind exact McNemar: given n trials with win chance p, how surprising is the observed count?

**Paired vs unpaired** — paired: the same cases through both models (what we do). Unpaired: separate samples; you need more data to see the same difference.

**Relatives** — paired t-test (continuous outcomes), Wilcoxon signed-rank (ordinal), permutation and bootstrap tests (resample the data, assume no distribution).

## Evaluation practice

**Held-out set** — cases never used to build, tune, or index the system, kept only for measurement. The 500-question set is held out.

**Golden set / golden data** — cases with known correct answers. The 105-case suite is hand-built golden data; "synthetic golden data" is generated with models as partners, then reviewed. It gates changes.

**Regression gate** — a suite that must pass before a change ships. The 105-case suite is the fast gate; the 500-question set decides serving.

**Candidate pool** — the documents a query retrieves before reranking (~30 here). Rerankers are compared on real pools, not hand-picked distractors.

## Retrieval metrics

**Top-k** — the k highest-scoring candidates a stage returns. Everything downstream sees only those k.

**Recall@k** — share of cases where the relevant document is in the top k. recall@1: the reranker put it first.

**MRR (mean reciprocal rank)** — the average of 1/rank of the relevant document. A near miss at rank 2 scores 0.5, so MRR rewards almost-right rankings that recall@1 discards.

**Precision** — share of returned documents that are relevant. Recall asks "did we find it"; precision asks "how much junk came with it."

**nDCG / MAP** — ranking metrics for graded relevance. nDCG weights top positions more; MAP averages precision at each relevant hit. Not used in the current eval; know them when relevance is graded.

**Rank correlation (Spearman, Kendall)** — how similarly two rankings order the same items. The two eval suites correlated ρ = +0.80: related, not identical.

**Reranker (cross-encoder)** — reads query and document together and scores the pair; slower, more accurate. A bi-encoder (embedding) scores them separately; fast, less precise.

**Hybrid retrieval** — mixing lexical and vector search; alpha sets the blend (0.5 = even).

## Calibration

**Calibration** — whether stated probabilities match observed frequencies: of the times a model says 0.8, does it happen 80% of the time?

**Brier score** — mean squared error of probabilistic predictions; lower is better.

**ECE (expected calibration error)** — the average gap between stated confidence and observed accuracy, in buckets. The number to quote when probabilities gate actions.

**Temperature scaling** — dividing logits by one learned number per question type to fix overconfidence; the standard first fix.
