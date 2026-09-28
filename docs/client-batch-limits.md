# Advertise each backend's client batch bound

Status: implemented. Each `/v1/models` entry carries the backend's
`max_client_batch_size` when the backend reports one; with several replicas the
router advertises the strictest. The client side is `mihado/base-stack#193`.

## The failure this fixes

The lab's TEI provider declared `MAX_RERANK_DOCUMENTS = 1_000` (a value copied
from the Voyage rerank descriptor, where 1,000 *is* documented) while the
deployed reranker enforces `--max-client-batch-size 64`:

```
embeddings -> 400: batch size 256 > maximum allowed batch size 64
```

A 256-document request passed the client's own validation and died upstream with
the backend's raw message; the caller then silently fell back to an un-reranked
order. Nothing in the router's catalogue said the bound existed.

The client side is being fixed in `mihado/base-stack#193` (the TEI rerank adapter
now chunks at the bound). This document is the other half: **make the bound
discoverable**, so the next client reads it instead of assuming.

## What to build

Each model entry in the router's catalogue (`GET /v1/models`) carries the
backend's client batch bound when the backend has one:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Alibaba-NLP/gte-reranker-modernbert-base",
      "object": "model",
      "owned_by": "tei",
      "max_client_batch_size": 64
    }
  ]
}
```

Additive: existing clients read `data[].id` and ignore extra fields (the lab's
`openAiModelIds` already does).

## Where the number comes from

1. **First, check the backend.** `curl -s localhost:8021/info | jq` — recent TEI
   builds report their own limits. If `max_client_batch_size` is present, carry
   it straight through; that is the authoritative source.
2. **Else read the container.** The router already inspects containers for
   discovery; the same inspect data carries the command that set the flag
   (`--max-client-batch-size 64` in `compose.yml`). A per-backend label (e.g.
   `tei.max-client-batch-size=64`) is a fine fallback if parsing the command is
   awkward — but prefer what the server reports.
3. **Several replicas, one id.** When two backends serve the same model, report
   the **minimum** bound — a client must respect the strictest replica.

## Shape of the change

- `router/backends.mjs`: carry the bound alongside the model id (the parsing
  helpers are pure and tested; extend `backends.test.mjs`).
- `router/index.mjs`: include the field in the `/v1/models` union. If the router
  also serves an `/info`-shaped union for raw-TEI clients, include it there too.
- `docs/operations.md`: document the advertised field next to the
  `--max-client-batch-size` / `--max-batch-tokens` tuning note it already has.

## What shipped

- `router/backends.mjs`: `infoMaxClientBatchSize` (parse) and
  `minClientBatchSize` (strictest replica), pure and covered in
  `backends.test.mjs`.
- `router/index.mjs`: `/v1/models` adds the field per model; the scan collects
  bounds alongside the advertised paths. Backends reporting none are omitted
  rather than advertised as unlimited.
- `docs/operations.md`: a "Client batch bounds" section with the curl.
- `router/dispatch.test.mjs`: two replicas reporting 64 and 32 advertise 32.

## Acceptance

```sh
curl -s localhost:8100/v1/models | jq '.data[] | select(.id | test("reranker"))'
# -> { "id": "Alibaba-NLP/gte-reranker-modernbert-base", ..., "max_client_batch_size": 64 }

curl -s localhost:8021/info          # a raw backend still answers as before
make test                            # router + py-common unit tests
```

The lab's provider can now stop hardcoding 64 and read the field per model.
