# AgentJev

- Qwen3-0.6B with the LM head removed and a trained candidate head: boolean/choice/score answers with full distributions, zero decoded tokens.
- Own Python service (this directory), in the `agentjev` profile, on GPU 0 — like Laya, TEI and vLLM cannot host it.
- Reuses the KV prefix across a question's candidates (Laya is one forward pass per call): up to 255 choice options in one call, 2048-token state budget.
- Upstream is young (tree cloned at a pinned commit in the Dockerfile); weights and code are Apache-2.0.

| Surface | Shape | How the router treats it |
| --- | --- | --- |
| `GET /info` | TEI's `{model_id}` | discovery; `503` until the model is loaded, so an unloaded model is never routed |
| `GET /health` | liveness | compose healthcheck only |
| `POST /rerank` | TEI's `{query, texts}` | one boolean per document, sent as one native batch call (32 max — the upstream batch limit, so no chunking) |
| `POST /api/evaluate` | `{state, questions}` or batched `{requests}` | forwarded verbatim |

```sh
make up-agentjev                          # the default stack plus AgentJev
make up-agentjev GPU=1                    # same, on the other card
```

`make up-agentjev CONCURRENCY=2` also starts `agentjev-b` on the other card (port 8046): one instance per card, and the router alternates requests between them.

Clients keep using the router port. The `model` field names the checkpoint:

```sh
# Rerank: every document is one boolean, all documents in one batch call.
curl -s localhost:8100/rerank -H 'content-type: application/json' -d '{
  "model": "aimeigaoshou/agent-jev",
  "query": "Do all tests pass on this patch?",
  "texts": ["Tests run: 14, Failures: 1.",
            "Tests run: 25, Failures: 0.",
            "The office is closed on Friday."]}'

# Native decisions: ask your own questions, or batch states.
curl -s localhost:8100/api/evaluate -H 'content-type: application/json' -d '{
  "model": "aimeigaoshou/agent-jev",
  "state": "Tests run: 14, Failures: 1.",
  "questions": [{"id": "done", "type": "boolean",
                 "question": "Are all tests passing?"}]}'
```

### Honest limits

- Over-length input is a hard refusal at 2048 tokens, mapped to 400 — never a silent truncation.
- Temperatures ship fitted on held-out calibration cases; refit on your own data before gating on a probability (same advice as Laya).
- The 79.25% typed-decisions figure is a specialist fit on that benchmark's split (their protocol holds out dev and calibration and discloses it); Jev's 72.7% is zero-shot. Different measurements, not a leaderboard.
- Not yet measured here. Same eval, `model=aimeigaoshou/agent-jev`, same 500 questions ([method](../docs/evaluation.md)).
