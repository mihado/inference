# eval/

Fast reranker regression, not a serving decision.

- `golden.json` — 20 hand-built cases in the codex domain (alcohol beverage
  regulation). Each case has one relevant document and distractors that share
  vocabulary, so a lexical shortcut fails and only relevance ordering passes.
  The full 500-Q held-out set in `docs/evaluation.md` stays authoritative for
  serving; this file catches breakage, model swaps, and shape drift in seconds.
- `run.mjs` — the bake-off runner (`make eval-golden`): every advertised
  reranker runs the same 20 cases through the router; unadvertised profiles
  SKIP, as do the paid APIs until the v1-compat adapter is filled in.
  Quality deltas never fail the run; request errors do.

Placement defeats position bias: `relevant_first: true` puts the relevant
document at index 0, false puts it last. Listwise rerankers (jina) read
documents jointly, so order must not decide the ranking.
