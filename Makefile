# Wrappers over docker compose and scripts/model.sh.
ROUTER_PORT ?= 8100

# Service-owned targets live with what they serve; included, not duplicated.
include ollama/ollama.mk

# Every compose file; only the "all" targets name them.
COMPOSE_ALL := docker compose -f compose.yml -f compose.rerankers.yml -f laya/compose.laya.yml -f ollama/compose.ollama.yml -f compose.qwen-embed.yml

# GPU= moves a profile's services to another card: make up-laya GPU=1. Empty
# means the compose default (GPU 0 for every profile below).
GPU ?= $(GPU_EXTRAS)

up: ## build (router) and start the default stack
	docker compose up -d --build

up-all: ## the default stack plus every profile
	$(COMPOSE_ALL) --profile "*" up -d --build

down: ## stop and remove the default-profile stack
	docker compose down

# Loads every profile file, so it removes the router too — nothing survives.
down-all: ## stop and remove EVERY profile, plus every ad-hoc slot
	$(COMPOSE_ALL) --profile "*" down --remove-orphans
	@ids="$$(docker ps -aq --filter label=tei.backend=1)"; \
	if [ -n "$$ids" ]; then docker rm -f $$ids; else echo "no ad-hoc slots"; fi

up-rerankers: ## start the default stack plus the four bake-off rerankers (GPU 0)
	GPU_EXTRAS=$(GPU) docker compose -f compose.yml -f compose.rerankers.yml --profile rerankers up -d --build

down-rerankers: ## stop and remove the bake-off rerankers only
	docker compose -f compose.rerankers.yml --profile rerankers down

up-qwen-embed: ## start the default stack plus the Qwen3 embedder pair
	docker compose -f compose.yml -f compose.qwen-embed.yml --profile qwen-embed up -d --build

down-qwen-embed: ## stop and remove the Qwen3 embedder pair only
	docker compose -f compose.qwen-embed.yml --profile qwen-embed down

up-laya: ## start the default stack plus the Laya decision service (GPU 0)
	GPU_EXTRAS=$(GPU) docker compose -f compose.yml -f laya/compose.laya.yml --profile laya up -d --build

down-laya: ## stop and remove the Laya service only
	docker compose -f laya/compose.laya.yml --profile laya down

status: ## live model of each server
	scripts/model.sh status

models: ## the router's union of models
	curl -s localhost:$(ROUTER_PORT)/v1/models

health: ## per-server /health (curl, not wget: the TEI image has no wget)
	@printf 'reranker  '; docker compose exec -T reranker curl -fsS http://127.0.0.1:80/health 2>/dev/null || printf 'down'; echo
	@printf 'voyage-embed '; docker compose exec -T voyage-embed python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:80/health')" >/dev/null 2>&1 && echo ok || echo down

logs: ## follow one service: make logs SVC=reranker
	docker compose logs -f $(SVC)

throughput: ## vLLM's tokens/s and queue depth while it works
	@out="$$(docker compose logs --since 5m voyage-embed voyage-embed-b 2>/dev/null | grep -E 'Avg prompt throughput|Running:' | tail -20)"; \
	if [ -n "$$out" ]; then echo "$$out"; else echo "no throughput lines yet - vLLM writes them per interval while it works"; fi

run: ## start a free ad-hoc TEI slot: make run MODEL=... [NAME=] [GPU=] [PORT=]
	scripts/run.sh "$(MODEL)" $(if $(NAME),--name $(NAME)) $(if $(GPU),--gpu $(GPU)) $(if $(PORT),--port $(PORT))

stop: ## remove an ad-hoc slot: make stop NAME=tei-...
	scripts/stop.sh "$(NAME)"
