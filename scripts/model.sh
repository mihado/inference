#!/usr/bin/env bash
# Show the live model of each server. A model change is a compose recreate:
# edit .env, then `docker compose up -d <service>`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_FILE="$ROOT/.env"

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
  printf '%-15s %-52s %s\n' "$service" "${model:-<compose default>}" "${state:-unknown}"
}

case "${1:-}" in
  status)
    print_service reranker '--model-id' MODEL_RERANKER
    print_service voyage-embed '--model' MODEL_VOYAGE_EMBED
    ;;
  *)
    echo "usage: scripts/model.sh status" >&2
    exit 2
    ;;
esac
