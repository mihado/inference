# Laya as the codex evidence judge (typed-decisions)

Bring-up for the GPU machine. The product calls Laya through its native surface
(`POST /v1/decisions`); this is the checkpoint to serve, why, and how to verify it before
the product points at it. Measured context lives in
`lexica-internal/codex/evaluations.md` (the Jev-as-judge benchmark).

## Why `typed-decisions`

The English base checkpoint is near chance on typed decisions zero-shot — 0.362 against a
0.318 random and 0.461 majority-class baseline, per its own model card — and its `noul`
can follow its `false:`/`true:` labels instead of the state
([laya#156](https://github.com/NandhaKishorM/laya/issues/156)). The `typed-decisions`
checkpoint is the fine-tuned one (0.766 overall; `noul` 0.857) and is what a decision
judge needs. The base checkpoint stays right for guardrails and email triage.

## Bring-up

In the inference repo root, `.env`:

```sh
LAYA_SUBFOLDER=typed-decisions
```

Then:

```sh
make up-laya
# underneath: docker compose -f compose.yml -f laya/compose.laya.yml --profile laya up -d --build laya
```

One service serves one checkpoint: a change is a recreate. The served model id becomes
`convaiinnovations/laya/typed-decisions`, and clients must send that name. The product's
Laya adapter keys on the `convaiinnovations/laya` prefix, so the suffixed id routes.

## Verify before the product points at it

1. `/info` answers `{"model_id":"convaiinnovations/laya/typed-decisions"}` once the load
   finishes. It answers `503` until then, so the router does not route to it yet.

2. A noul must separate a clear yes from a clear no. Through the router port:

```sh
curl -s localhost:8100/v1/decisions -H 'content-type: application/json' -d '{
  "model": "convaiinnovations/laya/typed-decisions",
  "state": {"question": "When may a shipment of alcohol be refused?",
            "excerpt": "The department shall refuse any shipment that lacks a certificate of compliance."},
  "questions": {"answers_question": {"type": "noul",
    "instructions": "Does `state.excerpt` contain the text that answers `state.question`?"}}}'
```

   Expect a high noul for that excerpt (~0.9) and a low one (~0.1) for the same question
   against `"This chapter governs the transportation of alcoholic beverages."` If both land
   mid-range or the sign flips, use the documented workaround: ask the same judgment as a
   two-option `choice` with neutral keys (one key holding the yes wording, the other the no
   wording) instead of a noul, and/or fit a temperature on our labeled pairs before gating.

3. Rerank-shaped evidence scoring, one forward pass over the candidates:

```sh
curl -s localhost:8100/rerank -H 'content-type: application/json' -d '{
  "model": "convaiinnovations/laya/typed-decisions",
  "query": "When may a shipment of alcohol be refused?",
  "texts": ["The department shall refuse a shipment that lacks a certificate.",
            "This chapter governs the transportation of alcoholic beverages."]}'
```

## What the product sends

- `POST /v1/decisions` with `{model, state, questions}` — the neutral question shape;
  wording and noul criteria are rendered as text by the adapter.
- The evidence judgment is one noul per excerpt: rubric `answerability.v1`, state
  `{question, excerpt}`; the caller compares nouls in code.
- `POST /rerank` with `{query, texts}` is Laya's designed ranking path (one fixed
  two-option noul per query-document pair, all pairs in one forward pass). Prefer it when
  the job is ordering candidates rather than scoring one excerpt.

## Caveats that matter for gating

- Context: 1024 tokens on this checkpoint (~768 for the state after the head budget). Our
  excerpts are ~400 characters; keep questions short — the question head shares the budget.
- It ships over-confident: refit one temperature per (question type, option count) on our
  own labeled pairs before gating on a probability (the card moves mean ECE 0.466 → 0.081).
- The `typed-decisions` family is English; other languages need `multilingual`.

## After bring-up

Re-run the evidence benchmark — the same 100 query-window-vs-head pairs the `jev-latest`
row used — with the suffixed model id, and record the row next to `jev-latest` in
`lexica-internal/codex/evaluations.md`. The rubric (`answerability.v1`) is in the product;
the benchmark harness is the lab session's scratch script (`noul-bench.mjs`), pointed at
`--models convaiinnovations/laya/typed-decisions`.
