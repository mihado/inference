# eval/

Fast reranker regression, not a serving decision.

- `golden.json` — 20 hand-built cases in the codex domain (alcohol beverage
  regulation). Each case has one relevant document and distractors that share
  vocabulary, so a lexical shortcut fails and only relevance ordering passes.
  The full 500-Q held-out set in `docs/evaluation.md` stays authoritative for
  serving; this file catches breakage, model swaps, and shape drift in seconds.
- `run.mjs` — the bake-off runner (`make eval-golden`): every advertised
  reranker runs the same 20 cases through the router, and the paid models
  (`rerank-3-lite`, `jev-latest`, `EVAL_PAID` to change) through the product
  gateway's v1-compat `/v1/rerank`. Quality deltas never fail the run; request
  errors do.

## Running

```sh
make eval-golden                     # locals via the router, paid via the gateway
EVAL_VERBOSE=1 make eval-golden      # list every miss
EVAL_PAID=rerank-3-lite,jev-latest make eval-golden
```

The paid adapter needs `LEXLAB_OPENAPI_KEY` in the environment; the gateway
base defaults to `https://lexlab-api.rter.cc` (`EVAL_V1_COMPAT` to override).
Cloudflare answers that host with a bot-403 unless the request carries a
`curl` user agent, so the runner sets one. Without a key the paid rows SKIP.

## Reading the numbers

`recall@1` is share of cases whose relevant document the reranker puts first;
`MRR` rewards a near-miss. Both are on 20 cases, so one case is worth 0.05
recall — read the miss list (`EVAL_VERBOSE`) before believing a gap, and
treat the 500-Q table in `docs/evaluation.md` as the decision.

Placement defeats position bias: `relevant_first: true` puts the relevant
document at index 0, false puts it last. Listwise rerankers (jina) read
documents jointly, so order must not decide the ranking.
