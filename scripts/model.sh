#!/usr/bin/env bash
# Load / unload / swap the model on one of the six TEI inference slots.
#
#   scripts/model.sh status
#   scripts/model.sh load <slot 1-6> <model-id>     # set model + (re)start the slot
#   scripts/model.sh unload <slot 1-6>              # stop the slot, freeing VRAM
#
# A swap is a recreate: TEI loads one model at startup, so changing the model
# means stopping the container and starting it again with a new --model-id.
# The model per slot lives in .env (MODEL_<slot>), which docker-compose reads.
# The `nano` vLLM embedder is not a TEI slot: it has its own service and args,
# so edit MODEL_NANO in .env and `docker compose up -d nano` to change it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_FILE="$ROOT/.env"

usage() {
  echo "usage: scripts/model.sh status | load <slot 1-6> <model-id> | unload <slot 1-6>" >&2
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

case "${1:-}" in
  status)
    for n in 1 2 3 4 5 6; do
      cid="$(docker compose ps -q "hf-${n}" 2>/dev/null || true)"
      if [[ -n "$cid" ]]; then
        # The live container's --model-id is the truth, not .env.
        model="$(docker inspect "$cid" --format '{{join .Config.Cmd " "}}' 2>/dev/null |
          sed -E 's/.*--model-id ([^ ]+).*/\1/')"
        state="$(docker inspect "$cid" --format '{{.State.Status}}' 2>/dev/null)"
      else
        model="$(grep -E "^MODEL_${n}=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2- || true)"
        state="stopped"
      fi
      printf 'hf-%s  %-52s  %s\n' "$n" "${model:-<compose default>}" "${state:-unknown}"
    done
    nano_cid="$(docker compose ps -q nano 2>/dev/null || true)"
    if [[ -n "$nano_cid" ]]; then
      # vLLM takes --model; the live container is the truth, not .env.
      nano_model="$(docker inspect "$nano_cid" --format '{{join .Config.Cmd " "}}' 2>/dev/null |
        sed -E 's/.*--model ([^ ]+).*/\1/')"
      nano_state="$(docker inspect "$nano_cid" --format '{{.State.Status}}' 2>/dev/null)"
    else
      nano_model="$(grep -E "^MODEL_NANO=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2- || true)"
      nano_state="stopped"
    fi
    printf 'nano  %-52s  %s\n' "${nano_model:-<compose default>}" "${nano_state:-unknown}"
    ;;
  load)
    [[ $# -eq 3 ]] || usage
    slot="$2"
    model="$3"
    [[ "$slot" =~ ^[1-6]$ ]] || { echo "slot must be 1-6" >&2; exit 2; }
    set_env "MODEL_${slot}" "$model"
    docker compose up -d --force-recreate "hf-${slot}"
    echo "hf-${slot} -> ${model}"
    ;;
  unload)
    [[ $# -eq 2 ]] || usage
    slot="$2"
    [[ "$slot" =~ ^[1-6]$ ]] || { echo "slot must be 1-6" >&2; exit 2; }
    docker compose stop "hf-${slot}"
    echo "hf-${slot} stopped"
    ;;
  *)
    usage
    ;;
esac
