# Spec: router discovers POST paths via `/info`

Status: implemented (router unions `/info` `paths` with its static set; all six
Python services advertise).

## Problem

The router allowlists POST paths in code (`router/index.mjs`, `POST_PATHS`).
Every new surface needs a router edit plus a router redeploy, or calls 404:
Julia-1 served `POST /v1/predict` and its README advertised it via the
router, but the router had no such entry — fixed after the fact in
`da56d29`. The next custom head will repeat this.

## Proposal

Backends advertise the POST paths they serve in their existing `/info`
answer, which the router already reads every 30s for discovery:

```json
{ "model_id": "SupersonicLabs/Julia-1", "paths": ["/v1/predict"] }
```

The router unions advertised paths per model and routes any POST whose body
carries a string `model`/`model_id` to the matching backend path. The only
code-kept mapping stays the `/rerank` + `/v1/rerank` Cohere
`documents`→`texts` normalization; everything else forwards verbatim, as
today.

## Contract

- `paths` is optional. A backend that omits it (TEI, vLLM — both unchangeable
  upstream) keeps today's static set: `/v1/embeddings`, `/rerank`,
  `/v1/rerank`, `/v1/decisions`, `/v1/systemone`, `/v1/predict`,
  `/api/evaluate`.
- Entries must be strings starting with `/`; anything else is ignored, never
  an error — a misbehaving backend degrades to unroutable paths, not a broken
  catalogue.
- Duplicates collapse; discovery order still defines rotation.
- An unlisted path still 404s with `invalid_request_error`, as today. No
  fail-open: the allowlist remains a boundary, it just moves from code to
  catalogue.

## Advertised surfaces

One source of truth (each service's `/info` carries its row):

| Service | `paths` |
| --- | --- |
| `laya` | `/rerank`, `/v1/decisions` |
| `agentjev` | `/rerank`, `/api/evaluate` |
| `jina` | `/rerank` |
| `jina-embed` | `/v1/embeddings` |
| `omnijev` | `/v1/systemone` |
| `julia` | `/v1/predict` |

## Rollout

1. Router: build the routable set as static defaults ∪ advertised paths;
   extend `router/dispatch.test.mjs` (stub advertises a novel path, assert it
   forwards; assert a malformed `paths` entry is ignored).
2. One key added to each Python service's `/info` return (`laya`, `agentjev`,
   `jina`, `jina-embed`, `omnijev`, `julia`): 1 line each, no behavior change
   — the router already serves those paths, so this is advertise-only.
3. Acceptance: add a new surface to one Python service (or rename one in a
   stub) and confirm it routes with no router change.

## Out of scope

- Auth/TLS in front of the router (unchanged).
- Per-model counters (`/health` metrics) and the eval harness: separate items.
- `scripts/smoke.mjs` needs no change: it already probes fixed endpoints per
  model kind and SKIPs what is absent. Iterating advertised paths instead is
  an optional follow-up, not required by this spec.
