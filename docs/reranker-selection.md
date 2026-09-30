# Reranker selection

Which reranker should serve. This supersedes the ordering in [evaluation.md](evaluation.md) for the local field, and records why that ordering changed.

The comparison below was run on 2026-09-29 against a **private corpus that is not reproduced here**. What is published is the method, the code, and the model comparison; the questions, documents, node identifiers and corpus version stay in the private repository. Where a number depends on corpus specifics this document says so rather than rounding it away.

## Method

Two things separate this from a naive bake-off, and both are the reason the result can be trusted at the margin.

**Real candidate pools, not authored distractors.** Every case is scored against the full candidate set the real retrieval stage actually returned for that query — roughly thirty documents, the production candidate count — with one of them the document that answers the question. Ranking among documents a real retrieval system surfaced is the product task. Ranking among hand-picked obvious non-answers is not, and it does not discriminate between models.

**Paired testing, not aggregate comparison.** All models see identical cases, so the comparison is paired. Each model is compared to the incumbent with an exact two-sided McNemar test over the discordant cases. This matters because an aggregate cannot distinguish a two-case gap from a twenty-case gap — and a two-case gap is what a marginal serving decision actually looks like.

**Two denominators, and the difference matters.** The harness scores every row of the file: 370 rows, and it reports `248/370 (0.670)` for the incumbent. The paired analysis keys on case id, and the file holds only **365 unique ids** — see *Known defect*, which is why. So the table below is on **365** (`248/365 = 0.679`) and the McNemar counts are on 365. The harness's own output and this table are both correct on their own basis; they differ because one counts rows and the other counts unique cases. The dedup lives in the analysis step, not the harness.

```
p = min(1, 2 · Σ_{j=0}^{min(a_only,b_only)} C(n,j) / 2^n),  n = a_only + b_only
```

The harness writes per-case ranks (`EVAL_PERCASE=<file>`), which is the input this test needs. Without it the harness only emits aggregates, and a close call cannot be settled.

## Result

Latency is mean per request, single-shot, measured on an otherwise idle machine — a floor, not a production forecast.

| model | recall@1 | MRR | mean_ms | vs incumbent |
| --- | --- | --- | --- | --- |
| `rerank-3` (Voyage, paid) | **0.756** | 0.829 | 544 | **leads by 28, p=0.0003** |
| `rerank-3-lite` (Voyage, paid) | 0.737 | 0.817 | 543 | **leads by 21, p=0.0065** |
| `Alibaba-NLP/gte-reranker-modernbert-base` (local, incumbent) | 0.679 | 0.779 | 104 | — |
| `jev-latest` (TypeSafe, paid) | 0.644 | 0.753 | 475 | gte leads by 13, p=0.18 — not significant |
| `BAAI/bge-reranker-v2-m3` (local) | 0.625 | 0.727 | 212 | gte leads by 20, p=0.0078 |
| `Alibaba-NLP/gte-multilingual-reranker-base` (local) | 0.625 | 0.733 | 134 | gte leads by 20, p=0.0265 |
| `cross-encoder/ms-marco-MiniLM-L6-v2` (local) | 0.578 | 0.693 | 16 | gte leads by 37, p<0.0001 |
| `ibm-granite/granite-embedding-reranker-english-r2` (local) | 0.559 | 0.689 | 123 | gte leads by 44, p<0.0001 |
| `convaiinnovations/laya/typed-decisions` (local) | 0.151 | 0.317 | 868 | gte leads by 193, p<0.0001 |

`SupersonicLabs/Julia-1` and `tinnel123/OmniJev-0.8B` are excluded: they serve `/v1/predict` and `/v1/systemone` and answer `/rerank` with 404, so they are not rerankers on this surface and their scores would not be comparable.

### `bge-reranker-v2-m3` does not displace the incumbent

This was the question the run was commissioned to settle. It is settled against bge: **36 discordant cases to 16, p = 0.0078.** A 2.25:1 split — bge is significantly *worse*, by 5.5 points of recall@1, and it is also twice as slow (212 ms against 104 ms). It loses on both axes at once, so there is no accuracy/latency trade to weigh. Keep `Alibaba-NLP/gte-reranker-modernbert-base`.

### The paid option is the real question, and it is not a modelling one

Not asked, and worth more than the answer that was asked: both Voyage models beat every local option significantly. `rerank-3-lite` leads by 21 cases (p=0.0065), `rerank-3` by 28 (p=0.0003). At the production candidate count that is roughly $0.0003–$0.0007 per query, inside the 200M-token free allowance for about 13,000 queries. The real cost is **+439 ms** on the retrieval path.

`rerank-3` and `rerank-3-lite` are statistically indistinguishable from each other here — 13 discordant cases to 6, p=0.167 — at 2.5× the price. If a paid reranker is taken, `rerank-3-lite` is the one the evidence supports.

The third paid option, TypeSafe `jev-latest`, was included for completeness because it is the one contender whose standing changes. It is not: it places fourth, is statistically indistinguishable from the free incumbent (p=0.18), and costs 4.5× the latency to do it. Its earlier second place was an artifact of that table's build.

**Recommendation: leave the local incumbent in place.** The case for changing is now "switch to a paid model", which is a latency and cost decision with a budget attached, not a consequence of a bake-off.

## What changed relative to `evaluation.md`

Only the top-two direction reproduces, and only just. The middle of that ordering does not: `ms-marco-MiniLM-L6-v2` was second there and is fourth here; `gte-multilingual-reranker-base` was last and is tied third. `jev-latest` was the second-best model in that table, ahead of the incumbent; here it lands fourth and is **not significantly different from it** (p=0.18) while taking 4.5× the latency. Its position there does not reproduce either. **The local ordering in `evaluation.md` should not be cited as a standing result** — it was measured on a different index build against a different question selection, and it does not survive re-measurement. The `gte > bge` direction does survive, which is the part the serving decision rests on.

On the previously hand-authored 105-case suite: it correlates ρ = +0.80 with this measurement, so it is not signal-free, but it ranked bge *above* gte and would have given the wrong answer to this exact question. It remains a regression gate. Do not use it to choose a model.

## Harness defect found and fixed

`relevantRank` validated the returned index permutation with a bare `.sort()`:

```js
const order = [...results.map((entry) => entry?.index)].sort();
if (order.join(",") !== [...Array(count).keys()].join(",")) return null;
```

With no comparator, JavaScript sorts lexicographically. For indices 0–29 that produces `0,1,10,11,…,2,20,…`, which never equals `0,1,2,3,…`, so **every well-formed response was rejected as malformed** and the run failed on the first case for every model.

The bug is invisible at ten documents or fewer, where `0,1,2,…,9` sorts identically either way. The 105-case suite used six documents per case, so it could not have caught this. The first run against a full thirty-document pool failed all 370 cases at once. Fixed as `.sort((a, b) => a - b)`.

The scorer is [`../eval/rank.mjs`](../eval/rank.mjs), and [`../eval/rank.test.mjs`](../eval/rank.test.mjs) pins that case so it cannot regress silently. Worth checking any other harness that validates a per-question index set for the same pattern — a lexicographic sort over a fixed-width range is the shape to look for.

## Known defect in the delivered cases

The case file holds 370 rows but only **365 unique case ids**, and the cause is worse than a duplicated label.

Six rows share one id *and* one question text *and* differ only in which document they designate. Within each of those rows, **five distractor slots are byte-identical copies of the relevant document** — that pool holds 4 distinct texts across 30 slots, five of them the answer. No reranker can separate identical strings, so the designated index's rank is a lottery, not a measurement. Those six rows are unwinnable in expectation for every model. On the 365-case basis used here they collapse to a single case, so they depress the absolute numbers by well under a point rather than the ~1.4 an earlier draft of this document claimed; the paired comparisons are unaffected because every model draws the same lottery.

The root cause is upstream and broader. In the question set, the template question `What legal issues were addressed in Bulletin ?, Item 8?` — a bulletin number stripped from the template — appears **14 times with 14 different expected documents**. Ids are derived from question text, so all 14 collide. A second question text appears twice. So the fix is not "dedupe one id": template questions whose placeholders were stripped need to be either regenerated with their placeholders intact or excluded, and the gold should be a **set** of acceptable documents across the whole affected family rather than one id. Fewer still: 16 of the 370 pools contain duplicate distractor texts, one with ten copies of a single text, so a nominal 30-document pool is sometimes ~20 distinct documents and ranking is easier than the slot count suggests.

Reported to the corpus owners; not fixable from here, since the case file is theirs.

## Reproducing

The private corpus is not in this repository, so the case file cannot be committed. The harness is, and the local field runs against it directly. What produced this document:

- [`../eval/README.md`](../eval/README.md) — the suite, its knobs, and what the metrics mean
- [`../eval/run.mjs`](../eval/run.mjs) — the runner: every advertised reranker through the router, paid models through the gateway
- [`../eval/rank.mjs`](../eval/rank.mjs) — the scorer, where the sort defect lived
- [`../eval/rank.test.mjs`](../eval/rank.test.mjs) — its tests, including the 30-document regression; runs in `make test`
- `../eval/golden.json` — the 105-case suite in this repo, superseded by the real-corpus file but still the fast gate (`make eval-golden`)

```bash
# the whole local field
LAYA_SUBFOLDER=typed-decisions EVAL_PAID="" node eval/run.mjs

# a subset, with per-case ranks for a paired test
LAYA_SUBFOLDER=typed-decisions EVAL_PAID="" \
  EVAL_LOCAL="BAAI/bge-reranker-v2-m3,Alibaba-NLP/gte-reranker-modernbert-base" \
  EVAL_PERCASE="percase.jsonl" node eval/run.mjs

# the paid field, through the gateway
EVAL_PAID="rerank-3,rerank-3-lite" node eval/run.mjs
```

`EVAL_PERCASE` writes `{model, id, rank, ms}` per case. It carries no question text, no document text, and no node identifier — but `id` **is** a corpus-derived value: for the delivered file it is an md5 of the question text, so it is a stable fingerprint of a private question rather than a neutral row number. Treat the per-case file as private-repo material. If a row key is needed for auditing, a per-model rank histogram is enough to recompute every p-value in this document and leaks nothing at all. The case file itself stays out: drop it in from the private repository as `eval/golden.json` to run the real comparison.

## Serving topology

The TEI rerankers (`text-embeddings-inference:86-1.9.1`) behind the router on `:8100` advertise `max_client_batch_size = 64`, and the Python services 64 except `agentjev` at 32 — all above the ~30 documents per request, so the bound never bound here. The bound is per model class, not one number: rerankers are 64, embedders 128. The four `rerankers`-profile models sit on GPU 0; the incumbent `reranker` pair is in the default stack. Paid models go through the gateway's `/v1/rerank` with Cohere's `documents` field rather than TEI's `texts`, and need a curl-like user-agent.

See [operations.md](operations.md) for bringing the profile models up and for the batch-token gotchas that make a pool overflow.
