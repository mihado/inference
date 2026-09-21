# inference

This repository runs six model servers on a 2x RTX A4000 box. A router puts
them behind one address. Every model has a replica on each GPU, so the router can
rotate a model's requests across both cards.

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

Seven services start by default: `router`, plus a pair for each of the three
models — `reranker`, `nano`, and `qwen-embed` — with the second replica on the
other GPU. Four more are optional, and they live in `docker-compose.full.yml`:

| Service | Port | GPU | Model |
| --- | --- | --- | --- |
| `bge-reranker` | 8081 | 0 | `BAAI/bge-reranker-v2-m3` |
| `ms-marco` | 8082 | 0 | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| `gte-multilingual` | 8083 | 0 | `Alibaba-NLP/gte-multilingual-reranker-base` |
| `granite-reranker` | 8084 | 0 | `ibm-granite/granite-embedding-reranker-english-r2` |


The optional services stay off until you enable the `full` profile:

```sh
make up-all                                   # both defaults, plus all extras
docker compose -f docker-compose.yml -f docker-compose.full.yml --profile full up -d --build
docker compose -f docker-compose.yml -f docker-compose.full.yml --profile full stop  # the extras only
```

They use GPU 0. The router finds them with no config change, because it lists
all labelled containers. Use them to compare rerankers. The MVP needs only
`reranker` and `nano`.

## Quick start

1. Create the model cache.

   ```sh
   mkdir -p "$HOME/.hf-cache"
   ```

2. Copy the example environment file.

   ```sh
   cp .env.example .env
   ```

3. Build the router image and start both servers.

   ```sh
   make up
   ```

4. Show the model of each server.

   ```sh
   make status
   ```

5. Show the model list of the router.

   ```sh
   make models
   ```

6. Show the token rate and the queue of the vLLM engines while they work.

   ```sh
   make throughput
   ```

One port must be open to the network: the router on 8100. Clients use the
router only. They do not use a server port.

## Router

`router/` is a proxy with no dependencies. It does three tasks:

- It reports the model list of all servers.
- It sends each `/v1/embeddings` call to the server with the requested model.
- It sends each `/rerank` call to the server with the requested model.

The router reads the model of each server. It reads the TEI `/info` data, or
the OpenAI `/v1/models` list of a vLLM server. It repeats the read every 30
seconds. So the router finds a new or changed server with no restart.

```sh
curl -s localhost:8100/v1/models
curl -s localhost:8100/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"voyageai/voyage-4-nano","input":["A brewer may sell beer."]}'
```

## Change a model

The `nano` model is a vLLM service, not a TEI slot. To change it, set
`MODEL_NANO` in `.env`, then run this command:

```sh
docker compose up -d nano
```

The `reranker` model is a TEI slot. TEI loads one model at start. A change is
a recreate. Use these commands:

```sh
make load MODEL=BAAI/bge-reranker-v2-m3   # set the model and recreate
make unload                               # stop the server and free its VRAM
```

For a short test, do not change the compose file. Start a free slot instead.
The script finds a free GPU and a free port. The router finds the new slot.

```sh
make run MODEL=BAAI/bge-reranker-base
make status
make stop NAME=tei-baai-bge-reranker-base
```

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

The 1024 and 2048 results are equal. So the 1024 index is the default. It is
half the size.

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

- You must use a reranker. The best local reranker adds 16 points to recall@1
  and 3 points to recall@5.
- The best local reranker is `Alibaba-NLP/gte-reranker-modernbert-base`. The
  `reranker` server runs this model.
- Voyage `rerank-3-lite` is the best reranker. It is the MVP choice.
- A difference of 0.002 is noise. One question of 500 causes it.

## First start

The first start is slow. These events are normal:

- The small files download in a few seconds. The log shows each file.
- The message `Could not download model.safetensors: 404` is not a fault. The
  model has shards. TEI uses the shard files.
- The shard download writes no log line. It is not stopped. To watch it, run
  `docker stats --no-stream reranker` and `du -sh "$HF_CACHE"`.
- A server does not listen before the model is loaded. A connection error at
  this time is normal. Wait for the word `Ready` in the log.

A cold start can take minutes. The download is the slow part.

## Prerequisites

- An NVIDIA driver with CUDA 12.2 or later.
- The NVIDIA Container Toolkit.

You must install the toolkit. Without it, Docker fails at container start with
the message `could not select device driver "" with capabilities: [[gpu]]`.
Ubuntu 24.04 does not have the package in its default repositories. Add the
NVIDIA source first:

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

Use the image tag for the compute capability of the GPU. The RTX A4000 is
Ampere 8.6. So the tag is `86-1.9.1`.

| Compute capability | Tag |
| --- | --- |
| sm80 (A100, A30) | `1.9.1` |
| sm86 (RTX A4000) | `86-1.9.1` |
| sm89 (RTX 4090) | `89-1.9.1` |
| Hopper | `hopper-1.9.1` |
| Turing | `turing-1.9.1` |

The 1.9.x images are CUDA 12.9. They must have driver 575 or later. If TEI
writes `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` and then uses the CPU, the tag does
not match the driver. Use `cuda-1.9.1` instead.

## Model cache

Each server uses the same host directory at `/data`. The default is
`$HOME/.hf-cache`. The Hugging Face cache is safe for more than one process,
because it uses file locks and atomic renames. So the share is safe, also
during the first download. You can download the models one time, then use a
read-only mount.

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

- The message `429 Model is overloaded` comes from TEI. It is backpressure,
  not a rate limit.
  - One 64-input request is about 12k tokens.
  - The limit `--max-batch-tokens` is 8192. So the queue can overflow.
  - Lower `--max-client-batch-size`, or raise `--max-batch-tokens`.
- `--max-batch-tokens` must be the largest value that the model accepts. TEI
  cannot calculate this value alone.
- `--served-model-name` sets the model name for the OpenAI surface. If you do
  not set it, the name is the Hugging Face id.
- A Matryoshka model can serve more dimensions than you index.
  - `voyage-4-nano` serves 2048 dimensions.
  - The client takes the first 1024 values, then normalizes them.
  - So the local index can use the vectors of the paid API.

## Model types

- **Embeddings:** TEI serves text-embedding models such as Nomic, BERT,
  XLM-RoBERTa, GTE, Qwen2, Qwen3, and Gemma3.
- **Rerankers:** TEI serves sequence-classification cross-encoders such as
  BERT, XLM-RoBERTa, GTE, and ModernBERT. An example is
  `BAAI/bge-reranker-v2-m3`.
- **Not supported on TEI:** generative rerankers (BAAI `v2-gemma`,
  `v2-minicpm-layerwise`, `Qwen/Qwen3-Reranker`) and custom architectures
  (`nvidia/llama-nemotron-rerank-1b-v2`, `JinaForRanking`). Run these models on
  a generation server.
  - TEI support for Qwen3 is for embeddings only. So no Qwen3 reranker loads.
  - `jinaai/jina-reranker-v2-base-multilingual` fails the TEI config parse. The
    field `model_type` is absent.
