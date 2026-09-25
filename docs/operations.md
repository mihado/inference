# Operations

Box setup and troubleshooting: first start, NVIDIA prerequisites, image tags, the shared model cache, diagnostics, and gotchas.

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
# 1. Add the NVIDIA toolkit source and install it.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 2. Connect the toolkit to Docker and verify.
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
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
