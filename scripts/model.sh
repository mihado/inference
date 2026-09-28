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

# Print one row for a running backend container: <service>  <model>  <state>.
# The model comes from the live /info (all backends serve TEI's shape there);
# a backend that does not answer yet falls back to its container command.
print_container() {
  local cid="$1" service model state port cmd
  service="$(docker inspect "$cid" --format '{{index .Config.Labels "com.docker.compose.service"}}' 2>/dev/null)"
  if [[ -z "$service" || "$service" == "<no value>" ]]; then
    service="$(docker inspect "$cid" --format '{{.Name}}' 2>/dev/null | sed 's#^/##')"
  fi
  case "$service" in
    reranker | voyage-embed) return 0 ;; # the two rows above already cover these
  esac
  state="$(docker inspect "$cid" --format '{{.State.Status}}' 2>/dev/null)"
  port="$(docker port "$cid" 80 2>/dev/null | sed -E -n 's/.*:([0-9]+)$/\1/p' | head -1)"
  model=""
  if [[ -n "$port" ]]; then
    model="$(curl -s -m 3 "localhost:$port/info" 2>/dev/null | sed -E -n 's/.*"model_id": *"([^"]+)".*/\1/p')"
  fi
  if [[ -z "$model" ]]; then
    cmd="$(docker inspect "$cid" --format '{{join .Config.Cmd " "}}' 2>/dev/null)"
    model="$(printf '%s' "$cmd" | sed -E -n 's/.*--model-id ([^ ]+).*/\1/p')"
    if [[ -z "$model" ]]; then
      model="$(printf '%s' "$cmd" | sed -E -n 's/.*--model ([^ ]+).*/\1/p')"
    fi
  fi
  printf '%-15s %-52s %s\n' "$service" "${model:-<loading>}" "${state:-unknown}"
}

case "${1:-}" in
  status)
    print_service reranker '--model-id' MODEL_RERANKER
    print_service voyage-embed '--model' MODEL_VOYAGE_EMBED
    # Every other labelled backend, by its live /info (localhost ports answer
    # from the box; backends bind localhost). The two rows above already cover
    # the default pair, including when stopped.
    for cid in $(docker ps -q --filter label=tei.backend=1 2>/dev/null); do
      print_container "$cid"
    done
    ;;
  *)
    echo "usage: scripts/model.sh status" >&2
    exit 2
    ;;
esac
