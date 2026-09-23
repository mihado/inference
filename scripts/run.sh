#!/usr/bin/env bash
# Start an ad-hoc TEI container for a model, on the compose network and
# labelled so the router discovers it. Picks a free GPU and a free host port
# (for reaching it from outside for debugging). No compose edits needed.
#
#   scripts/run.sh <model-id> [--name NAME] [--gpu N] [--port N]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${HF_CACHE:=$HOME/.hf-cache}"
: "${TEI_IMAGE_TAG:=86-1.9.1}"

MODEL="${1:?usage: run.sh <model-id> [--name N] [--gpu G] [--port P]}"
shift || true
NAME="tei-$(printf '%s' "$MODEL" | tr '/.:' '---' | tr '[:upper:]' '[:lower:]')"
GPU=""
PORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$GPU" ]]; then
  GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
if [[ -z "$PORT" ]]; then
  # Ad-hoc shares the 808x band with the optionals: pinned services sit at the
  # top (8099 downward), so take the lowest free port scanning up from 8080 —
  # bottom-up stays clear of them. The live-bindings check below still applies,
  # so a slot never takes an occupied port; if a profile starts onto a taken
  # port, compose fails loudly — bring the slot down first (`make stop`).
  for candidate in $(seq 8080 8099); do
    if ! docker ps --format '{{.Ports}}' | grep -q ":$candidate->"; then
      PORT="$candidate"
      break
    fi
  done
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --network inference_default \
  --label tei.backend=1 \
  --restart unless-stopped \
  --gpus "device=$GPU" \
  -p "${PORT}:80" \
  -v "${HF_CACHE}:/data" \
  "ghcr.io/huggingface/text-embeddings-inference:${TEI_IMAGE_TAG}" \
  --model-id "$MODEL" --dtype float16 --max-client-batch-size 128 --max-batch-tokens 32768

echo "$NAME  model=$MODEL  gpu=$GPU  host_port=$PORT  (router picks it up on the next scan)"
