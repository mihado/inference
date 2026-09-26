# inference

This repository runs model servers on a 2x RTX A4000 box. A router puts them behind one address. Every model has a replica on each GPU, so the router can rotate a model's requests across both cards. The default stack is the MVP — `reranker` and `voyage-embed`; every optional service is a profile in its own compose file (see "Profiles").

The rules for measurement and tuning are in [TUNING.md](TUNING.md); the measured numbers are in [docs/evaluation.md](docs/evaluation.md).

The two servers are:

- `reranker` — the local reranker.
- `voyage-embed` — the embedder for the `voyage-4-nano` model.

## Servers

| Name | Port | GPU | Runtime | Model |
| --- | --- | --- | --- | --- |
| `reranker` | 8021 | 1 | TEI | `Alibaba-NLP/gte-reranker-modernbert-base` |
| `voyage-embed` | 8001 | 1 | vLLM | `voyageai/voyage-4-nano` |
| `router` | 8100 | — | Node.js | — |

`voyage-embed` uses vLLM, not TEI. `voyage-4-nano` is a bf16 model:

- At fp16, TEI returns NaN vectors for some inputs.
- At fp32, TEI runs out of memory.
- vLLM runs the model at bf16, and the vectors are correct.

## Profiles

Five services start by default: `router`, plus a pair for each of the two models — `reranker` and `voyage-embed` — with the second replica on the other GPU. That is the MVP. Everything optional is a profile in its own compose file, and every profile has a pair of targets:

| Profile | File | Services | Ports | GPU |
| --- | --- | --- | --- | --- |
| `rerankers` | `compose.rerankers.yml` | `bge-reranker`, `ms-marco`, `gte-multilingual`, `granite-reranker` | 8099, 8098, 8097, 8096 | 0 |
| `qwen-embed` | `compose.qwen-embed.yml` | `qwen-embed`, `qwen-embed-b` | 8003, 8004 | 0, 1 |
| `laya` | `laya/compose.laya.yml` | `laya` (own Python runtime) | 8043 | 0 |
| `agentjev` | `agentjev/compose.agentjev.yml` | `agentjev` (own Python runtime) | 8045 | 0 |
| `jina` | `jina/compose.jina.yml` | `jina` (own Python runtime, non-commercial) | 8095 | 0 |
| `jina-embed` | `jina-embed/compose.jina-embed.yml` | `jina-embed` (own Python runtime, non-commercial) | 8093 | 0 |
| `omnijev` | `omnijev/compose.omnijev.yml` | `omnijev` (own Python runtime, Apache-2.0) | 8041 | 0 |
| `ollama` | `compose.ollama.yml` | `ollama` (own runtime, beside the router, no label) | 11434 | 1 |

| Target | Effect |
| --- | --- |
| `make up-rerankers` / `make down-rerankers` | the four bake-off rerankers |
| `make up-qwen-embed` / `make down-qwen-embed` | the Qwen3 embedder pair (the index build model) |
| `make up-laya` / `make down-laya` | the Laya decision service |
| `make up-agentjev` / `make down-agentjev` | the AgentJev decision service |
| `make up-jina` / `make down-jina` | the Jina reranker (non-commercial, local dev) |
| `make up-jina-embed` / `make down-jina-embed` | the Jina embedder (non-commercial, local dev) |
| `make up-omnijev` / `make down-omnijev` | the OmniJev decision service (Apache-2.0) |
| `make up-ollama` / `make down-ollama` | the Ollama GGUF runner (reference only) |
| `make up-all` / `make down-all` | every profile, everything |

Port bands, so each kind of server lives in its own mental space: 8001–8004 embeds, 8021–8022 rerankers, 8041 upward Python services, 8080–8099 optionals and ad-hoc slots (pinned top-down from 8099, ad-hoc bottom-up from 8080), 8100 router.

Each `up-*` starts the default stack as well, so one command always leaves a router in front of what it started. Each `down-*` removes only its own profile's services — the router and the default models keep running. `GPU=` moves a profile's services to another card (`make up-laya GPU=1`); empty means each profile's default. The compose command underneath, for the rerankers profile:

```sh
docker compose -f compose.yml -f compose.rerankers.yml \
  --profile rerankers up -d --build
```

Every Python profile takes a second replica on the other card, with an adjacent port pair: `omnijev` 8041/8042, `laya` 8043/8044, `agentjev` 8045/8046, `jina-embed` 8093/8092, `jina` 8095/8094. `make up-omnijev CONCURRENCY=2` enables the `-b` service (default `CONCURRENCY=1` keeps one instance), and the router alternates requests between each pair — the same `-b` shape the default stack uses.

Every profiled service carries the `tei.backend=1` label, so the router finds it with no config change — `make models` is the live union. Use the rerankers profile to compare rerankers with one another and with the paid APIs.

## Quick start

`make help` (or bare `make`) lists every target and the make flags.

```sh
# 1. Create the model cache.
mkdir -p "$HOME/.hf-cache"

# 2. Build the router image and start the default stack (no `.env` needed,
# every variable has a default in the compose files).
make up

# 3. Show the model of each server.
make status

# 4. Show the model list of the router.
make models

# 5. Show the token rate and the queue of the vLLM engines while they work.
make throughput
```

One port must be open to the network: the router on 8100. Clients use the router only. They do not use a server port.

## Router

`router/` is a proxy with no dependencies. It does three tasks:

- It reports the model list of all servers.
- It sends each `/v1/embeddings` call to the server with the requested model.
- It sends each `/rerank` call to the server with the requested model.

The router reads the model of each server. It reads the TEI `/info` data, or the OpenAI `/v1/models` list of a vLLM server. It repeats the read every 30 seconds. So the router finds a new or changed server with no restart.

```sh
curl -s localhost:8100/v1/models
curl -s localhost:8100/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"voyageai/voyage-4-nano","input":["A brewer may sell beer."]}'
```

## Change a model

The defaults live in the compose files — `MODEL_VOYAGE_EMBED` for voyage-embed (vLLM) and `MODEL_RERANKER` for the reranker (TEI). To change one, set it in `.env` (optional overrides only) and recreate. Each server loads one model at start, so a change is a recreate:

```sh
echo 'MODEL_RERANKER=BAAI/bge-reranker-v2-m3' >>.env
docker compose up -d voyage-embed reranker
```

For a short test, do not change the compose file. Start a free slot instead. The script finds a free GPU and a free port. The router finds the new slot.

```sh
make run MODEL=BAAI/bge-reranker-base
make status
make stop NAME=tei-baai-bge-reranker-base
```

## Services

Each optional service documents itself beside its compose file:

- [Laya](laya/README.md) — the decision model (`laya` profile, GPU 0, port 8043). Typed questions with calibrated probabilities; the candidate for the one rerank row that is currently a paid API.
- [AgentJev](agentjev/README.md) — the decision model on Qwen3-0.6B (`agentjev` profile, GPU 0, port 8045). Boolean, choice, and score answers with full distributions and zero decoded tokens.
- [Jina](jina/README.md) — the listwise reranker plus the `jina-embed` embedder sibling (`jina` and `jina-embed` profiles, ports 8095 and 8093). Non-commercial weights: local dev and eval only.
- [OmniJev](omnijev/README.md) — the omni-modal decision model (`omnijev` profile, GPU 0, port 8041). Typed questions about one image with calibrated probabilities and explicit abstention; Apache-2.0, the one decision service that may serve.
- [Ollama](ollama/README.md) — the GGUF runner beside the router (port 11434, no discovery). Reference only.

## Measured results

The codex retrieval evaluation numbers live in [docs/evaluation.md](docs/evaluation.md): embedding models at 1024 vs 2048 dims, and the reranker bake-off, where the best local reranker adds 16 points to recall@1.

## Operations

Box setup and troubleshooting live in [docs/operations.md](docs/operations.md): first start, NVIDIA prerequisites, image tags, the shared model cache, diagnostics, and gotchas.
