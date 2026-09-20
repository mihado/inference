#!/usr/bin/env bash
# Remove an ad-hoc slot started by scripts/run.sh (or any container by name).
set -euo pipefail
NAME="${1:?usage: stop.sh <container-name>}"
docker rm -f "$NAME"
