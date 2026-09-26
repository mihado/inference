# Wrappers over docker compose and scripts/model.sh.
ROUTER_PORT ?= 8100

# Service-owned targets live with what they serve; included, not duplicated.
include ollama/ollama.mk

# Every compose file; only the "all" targets name them.
COMPOSE_ALL := docker compose -f compose.yml -f compose.rerankers.yml -f laya/compose.laya.yml -f agentjev/compose.agentjev.yml -f ollama/compose.ollama.yml -f jina/compose.jina.yml -f jina-embed/compose.jina-embed.yml -f omnijev/compose.omnijev.yml -f compose.qwen-embed.yml

# Concurrency is instances per Python profile, not workers: CONCURRENCY=2 adds
# the profile's second replica (-b) on the other card, with its own port; the
# router alternates requests between the two, exactly like the default stack's
# -b services. Default 1 keeps a single instance.
CONCURRENCY ?= 1

# GPU= moves a profile's first replica to another card: make up-laya GPU=1.
# Empty means the compose default (first replica GPU 0). The second replica
# takes the other card unless GPU_B is set explicitly.
GPU ?= $(GPU_EXTRAS)
GPU_B ?= $(if $(filter 0,$(GPU)),1,$(if $(filter 1,$(GPU)),0,1))

# --profile for a Python service, adding its -b replica when CONCURRENCY=2.
py-profiles = --profile $(1)$(if $(filter 2,$(CONCURRENCY)), --profile $(1)-b)

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

up-laya: ## start the default stack plus the Laya service (CONCURRENCY=2: second replica)
	GPU_EXTRAS=$(GPU) GPU_EXTRAS_B=$(GPU_B) docker compose -f compose.yml -f laya/compose.laya.yml $(call py-profiles,laya) up -d --build

down-laya: ## stop and remove the Laya service, both replicas
	docker compose -f laya/compose.laya.yml --profile laya --profile laya-b down

up-agentjev: ## start the default stack plus the AgentJev service (CONCURRENCY=2: second replica)
	GPU_EXTRAS=$(GPU) GPU_EXTRAS_B=$(GPU_B) docker compose -f compose.yml -f agentjev/compose.agentjev.yml $(call py-profiles,agentjev) up -d --build

down-agentjev: ## stop and remove the AgentJev service, both replicas
	docker compose -f agentjev/compose.agentjev.yml --profile agentjev --profile agentjev-b down

up-jina: ## start the default stack plus the Jina reranker (CONCURRENCY=2: second replica, non-commercial)
	GPU_EXTRAS=$(GPU) GPU_EXTRAS_B=$(GPU_B) docker compose -f compose.yml -f jina/compose.jina.yml $(call py-profiles,jina) up -d --build

down-jina: ## stop and remove the Jina service, both replicas
	docker compose -f jina/compose.jina.yml --profile jina --profile jina-b down

up-jina-embed: ## start the default stack plus the Jina embedder (CONCURRENCY=2: second replica, non-commercial)
	GPU_EXTRAS=$(GPU) GPU_EXTRAS_B=$(GPU_B) docker compose -f compose.yml -f jina-embed/compose.jina-embed.yml $(call py-profiles,jina-embed) up -d --build

down-jina-embed: ## stop and remove the Jina embedder, both replicas
	docker compose -f jina-embed/compose.jina-embed.yml --profile jina-embed --profile jina-embed-b down

up-omnijev: ## start the default stack plus the OmniJev service (CONCURRENCY=2: second replica, Apache-2.0)
	GPU_EXTRAS=$(GPU) GPU_EXTRAS_B=$(GPU_B) docker compose -f compose.yml -f omnijev/compose.omnijev.yml $(call py-profiles,omnijev) up -d --build

down-omnijev: ## stop and remove the OmniJev service, both replicas
	docker compose -f omnijev/compose.omnijev.yml --profile omnijev --profile omnijev-b down

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
