# Laya

Laya is a decision model — ModernBERT plus a typed decision head. It answers typed questions about one state in one forward pass, with calibrated probabilities, and never generates text. It is not an embedder and not a cross-encoder, so neither TEI nor vLLM can load it: it runs its own Python service (this directory), in the `laya` profile, on GPU 0. It is the candidate for the one rerank row that is currently a paid API (TypeSafe `jev-latest`).

| Surface | Shape | How the router treats it |
| --- | --- | --- |
| `GET /info` | TEI's `{model_id}` | discovery; `503` until the model is loaded, so an unloaded model is never routed |
| `GET /health` | liveness | compose healthcheck only |
| `POST /rerank` | TEI's `{query, texts}` | found and proxied like any TEI slot, no router change |
| `POST /v1/decisions` | `{state, questions}` or `{state, preset}` | forwarded verbatim |

Start it alone, or with everything (from the repo root):

```sh
make up-laya                               # the default stack plus Laya
docker compose -f compose.yml -f laya/compose.laya.yml \
  --profile laya up -d --build laya        # the compose command underneath
make up-all                                # every profile, everything
```

Clients keep using the router port. The `model` field names the checkpoint:

```sh
# Rerank: every document is one noul question, all documents in one forward pass.
curl -s localhost:8100/rerank -H 'content-type: application/json' -d '{
  "model": "convaiinnovations/laya",
  "query": "Can a brewer sell beer directly at the taproom?",
  "texts": ["A brewer may sell beer on the licensed premises.",
            "The label must carry a health warning.",
            "Tax returns are due quarterly."]}'

# Typed decisions: ask your own questions, or take a preset question set.
curl -s localhost:8100/v1/decisions -H 'content-type: application/json' -d '{
  "model": "convaiinnovations/laya",
  "state": {"message": "We were billed twice for March. Refund today or we cancel."},
  "preset": "triage"}'
```

Presets: `triage`, `email`, `guard`, `moderation`, `router`. A response carries one answer per question — `choice` with the full `probabilities` map, `score` with its legend, `noul` as a probability — plus `confidence` and an `act_probability`.

### Checkpoints

One service serves one checkpoint. It is loaded at start, so a change is a recreate, and the served model id changes with it:

| `MODEL_LAYA` + `LAYA_SUBFOLDER` | Params | Context | Best at |
| --- | --- | --- | --- |
| `convaiinnovations/laya` (root, default) | 421M | 512 | English: guardrails, email triage |
| `…` + `multilingual` | 322M | 1024 | 100+ languages |
| `…` + `typed-decisions` | 421M | 1024 | the four typed-decisions workflows (0.766) |

```sh
# .env: LAYA_SUBFOLDER=typed-decisions  ->  the served id becomes
# convaiinnovations/laya/typed-decisions, and clients must send that name.
docker compose -f compose.yml -f laya/compose.laya.yml \
  --profile laya up -d laya
```

### Why one noul per document

A choice question scores every option at its own marker, and all options of a question share one `head_max_len` budget (192 tokens on the root checkpoint). Thirty documents as choice options leave roughly five tokens each, and options that short stop being distinguishable — the collapse the model card measures on Banking77 (0.425 accuracy). So `/rerank` sends one fixed two-option `noul` question per (query, document) pair instead: each pair keeps the whole state budget, and the same `collate_items` batch that `system_one` uses for many questions puts all thirty pairs in one forward pass. Full documents, one pass, no shortlist.

### Honest limits

- The root checkpoint is English only. Anything else needs `multilingual`.
- The base checkpoints are near chance on typed-decisions zero-shot (0.362 against a 0.318 random baseline). The 0.766 that beats Jev belongs to the `typed-decisions` checkpoint, fine-tuned on that benchmark's own split.
- It ships over-confident. Refit a temperature on your own data before gating on a probability; the model card moves ECE 0.466 to 0.081 after the refit.
- The root context is 512 tokens: the question head takes its share first and the state keeps the rest (~440), truncated at the tail.
- The first start downloads ~842 MB into the shared cache. `/info` answers 503 until the load finishes, so the router simply does not route to it yet.
- Not yet measured here. To place it in the rerank table in [docs/evaluation.md](../docs/evaluation.md), point the codex retrieval evaluation at `model=convaiinnovations/laya` and compare on the same 500 questions.
