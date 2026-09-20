#!/usr/bin/env bash
# Show the live model of each server. Load or unload the reranker model.
#
#   scripts/model.sh status
#   scripts/model.sh load <model-id>   # set the reranker model and recreate it
#   scripts/model.sh unload            # stop the reranker and free its VRAM
#
# A swap is a recreate: TEI loads one model at start. The reranker model lives
# in .env as MODEL_RERANKER. The nano embedder is a vLLM service, not a TEI
# slot: set MODEL_NANO in .env, then run `docker compose up -d nano`. See
# README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_FILE="$ROOT/.env"

usage() {
  echo "usage: scripts/model.sh status | load <model-id> | unload" >&2
  exit 2
}

set_env() {
  local key="$1" value="$2"
  touch "$ENV_FILE"
  if grep -qE "^${key}=" "$ENV_FILE"; then
    local tmp
    tmp="$(mktemp)"
    sed "s|^${key}=.*|${key}=${value}|" "$ENV_FILE" >"$tmp" && mv "$tmp" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$value" >>"$ENV_FILE"
  fi
}

# Print one row: <service>  <model>  <state>. The live container is the truth.
# A stopped service falls back to .env, then to the compose default.
print_service() {
  local service="$1" model_flag="$2" env_key="$3"
  local cid model state
  cid="$(docker compose ps -q "$service" 2>/dev/null || true)"
  if [[ -n "$cid" ]]; then
    model="$(docker inspect "$cid" --format '{{join .Config.Cmd " "}}' 2>/dev/null |
      sed -E "s/.*${model_flag} ([^ ]+).*/\1/")"
    state="$(docker inspect "$cid" --format '{{.State.Status}}' 2>/dev/null)"
  else
    model="$(grep -E "^${env_key}=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2- || true)"
    state="stopped"
  fi
  printf '%-9s %-52s %s\n' "$service" "${model:-<compose default>}" "${state:-unknown}"
}

case "${1:-}" in
  status)
    print_service reranker '--model-id' MODEL_RERANKER
    print_service nano '--model' MODEL_NANO
    ;;
  load)
    [[ $# -eq 2 ]] || usage
    set_env MODEL_RERANKER "$2"
    docker compose up -d --force-recreate reranker
    echo "reranker -> $2"
    ;;
  unload)
    [[ $# -eq 1 ]] || usage
    docker compose stop reranker
    echo "reranker stopped"
    ;;
  *)
    usage
    ;;
esac
