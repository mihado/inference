# inference

This repository runs model servers on a 2x RTX A4000 box. A router puts them behind one address. Every model has a replica on each GPU, so the router can rotate a model's requests across both cards. The default stack is the MVP — `reranker` and `nano`; every optional service is a profile in its own compose file (see "Profiles").

The rules for measurement and tuning are in [TUNING.md](TUNING.md).

The two servers are:

- `reranker` — the local reranker.
- `nano` — the embedder for the `voyage-4-nano` model.

## Servers

| Name | Port | GPU | Runtime | Model |
| --- | --- | --- | --- | --- |
| `reranker` | 8080 | 1 | TEI | `Alibaba-NLP/gte-reranker-modernbert-base` |
| `nano` | 8086 | 1 | vLLM | `voyageai/voyage-4-nano` |
| `router` | 8100 | — | Node.js | — |

`nano` uses vLLM, not TEI. `voyage-4-nano` is a bf16 model:

- At fp16, TEI returns NaN vectors for some inputs.
- At fp32, TEI runs out of memory.
- vLLM runs the model at bf16, and the vectors are correct.

## Profiles

Five services start by default: `router`, plus a pair for each of the two models — `reranker` and `nano` — with the second replica on the other GPU. That is the MVP. Everything optional is a profile in its own compose file, and every profile has a pair of targets:

| Profile | File | Services | Ports | GPU |
| --- | --- | --- | --- | --- |
| `rerankers` | `compose.rerankers.yml` | `bge-reranker`, `ms-marco`, `gte-multilingual`, `granite-reranker` | 8081–8084 | 0 |
| `qwen-embed` | `compose.qwen-embed.yml` | `qwen-embed`, `qwen-embed-b` | 8085, 8088 | 0, 1 |
| `laya` | `laya/compose.laya.yml` | `laya` (own Python runtime) | 8090 | 0 |
| `ollama` | `compose.ollama.yml` | `ollama` (own runtime, beside the router, no label) | 11434 | 1 |

| Target | Effect |
| --- | --- |
| `make up-rerankers` / `make down-rerankers` | the four bake-off rerankers |
| `make up-qwen-embed` / `make down-qwen-embed` | the Qwen3 embedder pair (the index build model) |
| `make up-laya` / `make down-laya` | the Laya decision service |
| `make up-ollama` / `make down-ollama` | the Ollama GGUF runner (reference only) |
| `make up-all` / `make down-all` | every profile, everything |

Each `up-*` starts the default stack as well, so one command always leaves a router in front of what it started. Each `down-*` removes only its own profile's services — the router and the default models keep running. `GPU=` moves a profile's services to another card (`make up-laya GPU=1`); empty means each profile's default. The compose command underneath, for the rerankers profile:

```sh
docker compose -f compose.yml -f compose.rerankers.yml \
  --profile rerankers up -d --build
```

Every profiled service carries the `tei.backend=1` label, so the router finds it with no config change — `make models` is the live union. Use the rerankers profile to compare rerankers with one another and with the paid APIs.

## Quick start

1. Create the model cache.

   ```sh
   mkdir -p "$HOME/.hf-cache"
   ```

2. Build the router image and start the default stack — no `.env` needed, every variable has a default in the compose files.

   ```sh
   make up
   ```

3. Show the model of each server.

   ```sh
   make status
   ```

4. Show the model list of the router.

   ```sh
   make models
   ```

5. Show the token rate and the queue of the vLLM engines while they work.

   ```sh
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

The defaults live in the compose files — `MODEL_NANO` for nano (vLLM) and `MODEL_RERANKER` for the reranker (TEI). To change one, set it in `.env` (optional overrides only) and recreate. Each server loads one model at start, so a change is a recreate:

```sh
echo 'MODEL_RERANKER=BAAI/bge-reranker-v2-m3' >>.env
docker compose up -d nano reranker
```

For a short test, do not change the compose file. Start a free slot instead. The script finds a free GPU and a free port. The router finds the new slot.

```sh
make run MODEL=BAAI/bge-reranker-base
make status
make stop NAME=tei-baai-bge-reranker-base
```

## Laya

Laya is a decision model — ModernBERT plus a typed decision head. It answers typed questions about one state in one forward pass, with calibrated probabilities, and never generates text. It is not an embedder and not a cross-encoder, so neither TEI nor vLLM can load it: it runs its own Python service in `laya/`, in the `laya` profile, on GPU 0. It is the candidate for the one rerank row that is currently a paid API (TypeSafe `jev-latest`).

| Surface | Shape | How the router treats it |
| --- | --- | --- |
| `GET /info` | TEI's `{model_id}` | discovery; `503` until the model is loaded, so an unloaded model is never routed |
| `GET /health` | liveness | compose healthcheck only |
| `POST /rerank` | TEI's `{query, texts}` | found and proxied like any TEI slot, no router change |
| `POST /v1/decisions` | `{state, questions}` or `{state, preset}` | forwarded verbatim |

Start it alone, or with everything:

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
- Not yet measured here. To place it in the rerank table above, point the codex retrieval evaluation at `model=convaiinnovations/laya` and compare on the same 500 questions.

## Ollama

Ollama runs GGUF models behind its own API, beside the router rather than behind it. One Ollama server holds many models with on-demand loading, while the router assumes one model per server with a static id — and the router buffers full responses, so token streaming would break passing through it. No label, no discovery, no streaming work: clients use port 11434 directly with the full `/api/*` surface. Reference only; rarely used these days.

```sh
make up-ollama                                # the default stack plus Ollama
make ollama-deps                              # pull every model in ollama/ollama.mk
docker compose -f compose.yml -f ollama/compose.ollama.yml --profile ollama exec -T ollama ollama run gemma4:e4b
```

Residency is bounded because the box is shared: `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_KEEP_ALIVE` (30m — covers a work session's gaps; any call extends it per-request), plus the `GPU_OLLAMA` pin (GPU 1 by default: GPU 0 collects the primaries plus every extras profile, and one card keeps each model whole instead of split across PCIe). Models live in Ollama's own blob store (`OLLAMA_HOME`); nothing dedupes with the HF cache. Upgrade by bumping `OLLAMA_IMAGE_TAG`.

## Measured results

The numbers below come from the codex retrieval evaluation.

- The data set has 7,277 nodes from New York, DC, and Maryland.
- The question set has 500 held-out questions.
- The retrieval is hybrid with alpha 0.5 and 30 candidates.
- The index is `voyage-4-nano` at 1024 dimensions.
- The metrics are recall and MRR. A higher number is better.

### Embedding model, no rerank

| Model | Dimensions | recall@1 | recall@5 | MRR | Index speed |
| --- | --- | --- | --- | --- | --- |
| `voyage-4-nano` | 1024 | 0.564 | 0.906 | 0.710 | ~214 nodes/s |
| `voyage-4-nano` | 2048 | 0.560 | 0.902 | 0.709 | ~214 nodes/s |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | 0.562 | 0.880 | 0.696 | ~123 nodes/s |
| `Qwen/Qwen3-Embedding-4B` | 2560 | 0.566 | 0.882 | 0.702 | ~18 nodes/s |

The 1024 and 2048 results are equal. So the 1024 index is the default. It is half the size.

### Reranker

A reranker reads the 30 candidates and puts them in a new order.

| Reranker | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| none | 0.564 | 0.904 | 0.710 |
| Voyage `rerank-3-lite` | 0.796 | 0.962 | 0.873 |
| TypeSafe `jev-latest` | 0.752 | 0.960 | 0.848 |
| `Alibaba-NLP/gte-reranker-modernbert-base` | 0.724 | 0.934 | 0.822 |
| `BAAI/bge-reranker-v2-m3` | 0.676 | 0.930 | 0.788 |
| `ibm-granite/granite-embedding-reranker-english-r2` | 0.660 | 0.928 | 0.780 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 0.654 | 0.916 | 0.768 |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 0.684 | 0.906 | 0.780 |

Results:

- You must use a reranker. The best local reranker adds 16 points to recall@1 and 3 points to recall@5.
- The best local reranker is `Alibaba-NLP/gte-reranker-modernbert-base`. The `reranker` server runs this model.
- Voyage `rerank-3-lite` is the best reranker. It is the MVP choice.
- A difference of 0.002 is noise. One question of 500 causes it.

## First start

The first start is slow. These events are normal:

- The small files download in a few seconds. The log shows each file.
- The message `Could not download model.safetensors: 404` is not a fault. The model has shards. TEI uses the shard files.
- The shard download writes no log line. It is not stopped. To watch it, run `docker stats --no-stream reranker` and `du -sh "$HF_CACHE"`.
- A server does not listen before the model is loaded. A connection error at this time is normal. Wait for the word `Ready` in the log.

A cold start can take minutes. The download is the slow part.

## Prerequisites

- An NVIDIA driver with CUDA 12.2 or later.
- The NVIDIA Container Toolkit.

You must install the toolkit. Without it, Docker fails at container start with the message `could not select device driver "" with capabilities: [[gpu]]`. Ubuntu 24.04 does not have the package in its default repositories. Add the NVIDIA source first:

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # verify
```

## Image tag

Use the image tag for the compute capability of the GPU. The RTX A4000 is Ampere 8.6. So the tag is `86-1.9.1`.

| Compute capability | Tag |
| --- | --- |
| sm80 (A100, A30) | `1.9.1` |
| sm86 (RTX A4000) | `86-1.9.1` |
| sm89 (RTX 4090) | `89-1.9.1` |
| Hopper | `hopper-1.9.1` |
| Turing | `turing-1.9.1` |

The 1.9.x images are CUDA 12.9. They must have driver 575 or later. If TEI writes `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` and then uses the CPU, the tag does not match the driver. Use `cuda-1.9.1` instead.

## Model cache

Each server uses the same host directory at `/data`. The default is `$HOME/.hf-cache`. The Hugging Face cache is safe for more than one process, because it uses file locks and atomic renames. So the share is safe, also during the first download. You can download the models one time, then use a read-only mount.

## Diagnostics

| Symptom | Command | Cause |
| --- | --- | --- |
| `could not select device driver ""` | `docker info \| grep -i runtime` | The NVIDIA Container Toolkit is not installed, or not connected to Docker. |
| `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`, then CPU use | `nvidia-smi` (driver version and CUDA version) | The image CUDA is newer than the driver. Use a matching tag. |
| HTTP `000`, or connection refused | `docker compose ps`; `make logs SVC=reranker` | The server does not listen yet, or the container stopped. |
| No progress after `Starting ... model on Cuda` | `docker stats`; `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv` | GPU use near 0% and CPU use near 100% is a slow cold start, not a stop. |
| The download does not move | `du -sh "$HF_CACHE"`; `docker stats` (NET I/O) | A growing value is a download. A static value is a stop. A restart continues from the cache. |
| Why did the container stop? | `docker inspect --format 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' reranker` | The exit code, and the out-of-memory flag. |

## Gotchas

- The message `429 Model is overloaded` comes from TEI. It is backpressure, not a rate limit.
  - One 64-input request is about 12k tokens.
  - The limit `--max-batch-tokens` is 8192. So the queue can overflow.
  - Lower `--max-client-batch-size`, or raise `--max-batch-tokens`.
- `--max-batch-tokens` must be the largest value that the model accepts. TEI cannot calculate this value alone.
- `--served-model-name` sets the model name for the OpenAI surface. If you do not set it, the name is the Hugging Face id.
- A Matryoshka model can serve more dimensions than you index.
  - `voyage-4-nano` serves 2048 dimensions.
  - The client takes the first 1024 values, then normalizes them.
  - So the local index can use the vectors of the paid API.

## Model types

- **Embeddings:** TEI serves text-embedding models such as Nomic, BERT, XLM-RoBERTa, GTE, Qwen2, Qwen3, and Gemma3.
- **Rerankers:** TEI serves sequence-classification cross-encoders such as BERT, XLM-RoBERTa, GTE, and ModernBERT. An example is `BAAI/bge-reranker-v2-m3`.
- **Not supported on TEI:** generative rerankers (BAAI `v2-gemma`, `v2-minicpm-layerwise`, `Qwen/Qwen3-Reranker`) and custom architectures (`nvidia/llama-nemotron-rerank-1b-v2`, `JinaForRanking`). Run these models on a generation server.
  - TEI support for Qwen3 is for embeddings only. So no Qwen3 reranker loads.
  - `jinaai/jina-reranker-v2-base-multilingual` fails the TEI config parse. The field `model_type` is absent.
