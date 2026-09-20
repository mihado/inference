# inference

Inference slots for a 2× RTX A4000 box: TEI for the rerankers and Qwen3 embedder, vLLM for `voyage-4-nano` (which needs bf16). A small router fronts them all as one endpoint. One host weight cache is shared.

## Slots

| slot | port | GPU | runtime | default model |
| --- | --- | --- | --- | --- |
| `hf-1` | 8080 | 0 | TEI | `Alibaba-NLP/gte-reranker-modernbert-base` |
| `hf-2` | 8081 | 0 | TEI | `BAAI/bge-reranker-v2-m3` |
| `hf-3` | 8082 | 0 | TEI | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `hf-4` | 8083 | 0 | TEI | `Alibaba-NLP/gte-multilingual-reranker-base` |
| `hf-5` | 8084 | 0 | TEI | `ibm-granite/granite-embedding-reranker-english-r2` |
| `hf-6` | 8085 | 1 | TEI | `Qwen/Qwen3-Embedding-0.6B` |
| `nano` | 8086 | 1 | vLLM | `voyageai/voyage-4-nano` |

Slots `hf-1`–`hf-5` pin GPU 0 (the rerankers); `hf-6` and `nano` pin GPU 1 (the embedders). Placement is per-slot (`GPU_<n>`): any assignment works as long as the models fit the card. More slots can be added freely — see Ad-hoc slots.

`nano` is vLLM rather than TEI because `voyage-4-nano` is a bf16 checkpoint whose custom bidirectional pooling overflows to all-NaN vectors at TEI's fp16 and OOMs at fp32. vLLM runs it at its native bfloat16 with a bounded memory budget; it is configured in `docker-compose.yml` (not a `MODEL_<n>` TEI slot — see Changing models).

## Quick start

```sh
mkdir -p "$HOME/.hf-cache"
cp .env.example .env
make up            # builds the router image, starts every slot
make models        # the router's union of served models
make status        # live model per slot
```

Only **one** port needs to be reachable from outside the box: the router on **8100**. Clients point at it; they never address a slot.

## Router

`router/` is a dependency-free proxy that fronts every slot as one base URL (`:8100`): `GET /v1/models` returns the union of the slots' models, and `POST /v1/embeddings` / `/rerank` are dispatched by the requested model to the slot serving it. Model→slot is read from each container's TEI `/info`, or a vLLM slot's `/v1/models`, on a TTL — so a swapped or newly-started slot is picked up automatically.

```sh
curl -s localhost:8100/v1/models
curl -s localhost:8100/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":["A brewer may sell beer."]}'
```

Traefik and other HTTP proxies cannot dispatch on a JSON body, which is why this exists; a plain reverse proxy in front is enough if you need TLS/ingress.

## Changing models

TEI loads one model at startup, so a swap is a recreate.

```sh
make load SLOT=2 MODEL=BAAI/bge-reranker-v2-m3   # set + recreate hf-2
make unload SLOT=2                               # stop hf-2, freeing VRAM
```

For one-off experiments, skip the compose file — start an ad-hoc slot on the compose network, pinned to the GPU with the most free VRAM and labelled so the router discovers it without a restart:

```sh
make run MODEL=BAAI/bge-reranker-base            # free GPU + a free debug port
make status
make stop NAME=tei-baai-bge-reranker-base
```

The router discovers labelled containers (`tei.backend=1`) over the Docker socket; the static `BACKENDS` list is a fallback.

## Make targets

| target | does |
| --- | --- |
| `make up` / `make down` | build + start / stop the stack |
| `make status` | live model per slot |
| `make models` | the router's union of models |
| `make health` | per-slot `/health` |
| `make logs SVC=hf-3` | follow one service |
| `make load SLOT=2 MODEL=…` | set a slot's model and recreate it |
| `make unload SLOT=2` | stop a slot |
| `make run MODEL=…` | start an ad-hoc slot |
| `make stop NAME=tei-…` | remove an ad-hoc slot |

## Prerequisites

- NVIDIA driver with **CUDA ≥ 12.2**.
- **NVIDIA Container Toolkit.** Without it, Docker fails `--gpus` at container creation with `could not select device driver "" with capabilities: [[gpu]]`, and `docker info | grep -i runtime` shows only `runc`. Ubuntu 24.04 does **not** carry the package in its default repos — add NVIDIA's apt source first, then configure the runtime:

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

- **Image tag = GPU compute capability.** RTX A4000 is Ampere 8.6 → `86-1.9.1` (the default). Map: `1.9`=sm80 (A100/A30), `86-1.9.1`=sm86, `89-1.9.1`=sm89, `hopper-1.9.1`, `turing-1.9.1` (needs `USE_FLASH_ATTENTION=True`). The 1.9.x images are CUDA 12.9-based (driver ≥ 575). If TEI logs `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` and falls back to CPU, the tag's CUDA/driver combo is wrong — `cuda-1.9.1` is a known-good fallback.

## First start (what looks broken but isn't)

```sh
# 1. Small artifacts download in seconds, each logged by name.
docker compose logs -f hf-5

# 2. "Could not download model.safetensors: 404" is benign — the model is
#    sharded, so TEI falls back to model.safetensors.index.json + model-0000N shards.

# 3. The shard download logs nothing while it runs — it is NOT stuck. Watch it grow:
docker stats --no-stream hf-5      # NET I/O climbing
du -sh "$HOME/.hf-cache"           # size growing; a restart resumes from the cache

# 4. TEI does not listen until the model is loaded, so curl returns
#    connection-refused / HTTP 000 meanwhile. Wait for "Ready" in the logs.

# 5. Confirm it runs on the GPU. A driver/CUDA mismatch is SILENT — it logs a
#    warning and falls back to CPU, then chokes warming up a large model there:
#      CUDA_ERROR_SYSTEM_DRIVER_MISMATCH -> "Using CPU instead" -> "on Cpu"
docker compose logs | grep -E "on Cuda|on Cpu"
```

A large model's cold init can take minutes; the shard download dominates the first run.

## Shared cache

Every instance bind-mounts the same host directory to `/data` (`HF_CACHE`, default `$HOME/.hf-cache`). The Hugging Face hub cache is built for concurrent access (file locks and atomic renames), so sharing is safe — including a concurrent first download. Pre-download once, then later mounts may be read-only.

## Diagnostics

| Symptom | Check | Cause |
| --- | --- | --- |
| `could not select device driver ""` | `docker info \| grep -i runtime` | NVIDIA Container Toolkit missing/not wired into Docker |
| `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`, `Using CPU instead` | `nvidia-smi` (Driver vs CUDA Version) | Image CUDA newer than the driver — use a matching tag |
| HTTP `000` / connection refused | `docker compose ps`, `logs` | Not listening yet (loading/warming) or the container exited |
| Looks stuck after `Starting ... model on Cuda` | `docker stats`, `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv` | GPU util ~0% + CPU ~100% = slow cold init, not a hang |
| Is the download progressing? | `du -sh "$HF_CACHE"`, `docker stats` NET I/O | Growing = downloading; frozen = stalled (restart resumes) |
| Why did it exit? | `docker inspect --format 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' hf-5` | Exit code / OOM flag |

## Gotchas

- **`429 Model is overloaded`** is TEI backpressure, not a rate limiter: a 128-input request is ~25k tokens against `--max-batch-tokens 32768`, so only one fits per step and extras overflow the queue. Lower `--max-client-batch-size` or raise `--max-batch-tokens`.
- `--max-batch-tokens` should be the largest value the model tolerates before going compute-bound; TEI cannot infer it. Defaults: `--max-batch-tokens 16384`, `--max-client-batch-size 32`.
- `--served-model-name` sets the OpenAI-compatible model alias; unset means the Hugging Face id is the served name.
- **Dims can differ from the API.** `voyageai/voyage-4-nano` serves `num_labels: 2048` locally while Voyage's API defaults to 1024 (Matryoshka); truncate + renormalise client-side to match.

## Models

- **Embeddings:** any TEI text-embeddings model (Nomic, BERT, XLM-RoBERTa, GTE, Qwen2/3, Gemma3, …).
- **Rerankers:** TEI serves sequence-classification cross-encoders (BERT, XLM-RoBERTa, GTE, ModernBERT), e.g. `BAAI/bge-reranker-v2-m3`. Generative/decoder rerankers — BAAI `v2-gemma`, `v2-minicpm-layerwise`, `Qwen/Qwen3-Reranker` — and custom architectures (`nvidia/llama-nemotron-rerank-1b-v2`, `JinaForRanking`) are **not** served by TEI; run those on a generation server. TEI's Qwen3 support is embeddings-only, so no Qwen3 reranker loads; `jinaai/jina-reranker-v2-base-multilingual` fails TEI's config parse (`missing field model_type`).
