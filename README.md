# inference

Text Embeddings Inference (TEI) servers for a 2× RTX A4000 box. Two instances run one embedding model each — `Qwen/Qwen3-Embedding-4B` and `Qwen/Qwen3-Embedding-0.6B` — and share a single Hugging Face weight cache on the host.

| service | model | dims | port |
| --- | --- | --- | --- |
| `tei-4b` | `Qwen/Qwen3-Embedding-4B` | 2560 | 8080 |
| `tei-06b` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 8081 |

Both default to **GPU 1**; the 4B (~8 GB fp16) plus the 0.6B (~1.2 GB) fit in one 16 GB A4000, leaving GPU 0 free.

## Prerequisites

- NVIDIA driver with **CUDA ≥ 12.2**.
- **NVIDIA Container Toolkit.** Without it Docker fails `--gpus` at container creation with `could not select device driver "" with capabilities: [[gpu]]`, and `docker info | grep -i runtime` shows only `runc`. Ubuntu 24.04 does **not** carry the package in its default repos — add NVIDIA's apt source first:

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

- **Image tag = GPU compute capability.** RTX A4000 is Ampere 8.6 → `86-1.9`. The generic `1.9` is sm_80 (A100/A30): it runs on sm_86 but is not tuned. Map: `1.9`=sm80, `86-1.9`=sm86, `89-1.9`=sm89, `hopper-1.9`, `turing-1.9` (needs `USE_FLASH_ATTENTION=True`).

## Run

```sh
mkdir -p "$HOME/.hf-cache"
cp .env.example .env        # adjust tag/GPU/ports if needed
docker compose up -d
docker compose logs -f
```

Verify once ready:

```sh
curl -s http://127.0.0.1:8080/info | head -c 400
curl -s http://127.0.0.1:8081/info | head -c 400
curl -s http://127.0.0.1:8080/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-4B","input":["A brewer may sell beer."]}' | head -c 200
```

## First start (what looks broken but isn't)

```sh
# 1. Small artifacts download in seconds, each logged by name.
docker compose logs -f tei-4b

# 2. "Could not download model.safetensors: 404" is benign — the model is
#    sharded, so TEI falls back to model.safetensors.index.json + model-0000N shards.

# 3. The shard download logs nothing while it runs — it is NOT stuck. Watch it grow:
docker stats --no-stream tei-4b       # NET I/O climbing
du -sh "$HOME/.hf-cache"              # size growing; a restart resumes from the cache

# 4. TEI does not listen until the model is loaded, so curl returns
#    connection-refused / HTTP 000 meanwhile. Wait for "Ready" in the logs.

# 5. Confirm it runs on the GPU. A driver/CUDA mismatch is SILENT — it logs a
#    warning and falls back to CPU, then chokes warming up a 4B model there:
#      CUDA_ERROR_SYSTEM_DRIVER_MISMATCH -> "Using CPU instead" -> "on Cpu"
docker compose logs | grep -E "on Cuda|on Cpu"
```

A 4B cold init can take ~6 minutes; the shard download dominates the first run.

## Shared cache

Every instance bind-mounts the same host directory to `/data` (`HF_CACHE`, default `$HOME/.hf-cache`). The Hugging Face hub cache is built for concurrent access (file locks and atomic renames), so sharing is safe — including a concurrent first download. Pre-download once, then later mounts may be read-only.

## Diagnostics

| Symptom | Check | Cause |
| --- | --- | --- |
| `could not select device driver ""` | `docker info \| grep -i runtime` | NVIDIA Container Toolkit missing/not wired into Docker |
| `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`, `Using CPU instead` | `nvidia-smi` (Driver vs CUDA Version) | Image CUDA newer than the driver — use a matching tag |
| HTTP `000` / connection refused | `docker compose ps`, `logs` | Not listening yet (loading/warming) or the container exited |
| Looks stuck after `Starting FlashQwen3 model on Cuda` | `docker stats`, `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv` | GPU util ~0% + CPU ~100% = slow cold init, not a hang |
| Is the download progressing? | `du -sh "$HF_CACHE"`, `docker stats` NET I/O | Growing = downloading; frozen = stalled (restart resumes) |
| Why did it exit? | `docker inspect --format 'exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' tei-4b` | Exit code / OOM flag |

## Gotchas

- **`429 Model is overloaded`** is TEI backpressure, not a rate limiter: a 128-input request is ~25k tokens against `--max-batch-tokens 32768`, so only one fits per step and extras overflow the queue. Lower `--max-client-batch-size` or raise `--max-batch-tokens`.
- `--max-batch-tokens` should be the largest value the model tolerates before it becomes compute-bound; TEI cannot infer it. Defaults: `--max-batch-tokens 16384`, `--max-client-batch-size 32`.
- `--served-model-name` sets the OpenAI-compatible model alias; unset means the Hugging Face id is the served name.

## Other models

`voyageai/voyage-4-nano` (340M) is TEI-supported under the Qwen3 type, but its config carries `num_labels: 2048` and bidirectional attention — check the served dims via `/info` before assuming 1024. Re-rankers (e.g. `BAAI/bge-reranker-large`) run the same way on a third port if needed.
