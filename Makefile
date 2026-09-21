# Wrappers over docker compose and scripts/model.sh.
ROUTER_PORT ?= 8100

.PHONY: up up-all down down-all status models health logs load unload run stop

up: ## build (router) and start both servers
	docker compose up -d --build

up-all: ## start both servers plus every extra slot (bake-off)
	docker compose --profile "*" up -d --build

down: ## stop and remove the default-profile stack
	docker compose down

# `up`/`down` act on the default profile; `up-all`/`down-all` act on every
# profile, so the two pairs read as one scale of scope. `down` only removes
# services in the active profiles, so anything started with `up-all` (or an
# older run with other profiles) survives it, and the ad-hoc
# TEI slots from `make run` were never compose services at all. This is the
# target that leaves nothing running.
down-all: ## stop and remove EVERY profile, plus every ad-hoc slot
	docker compose --profile "*" down --remove-orphans
	@ids="$$(docker ps -aq --filter label=tei.backend=1)"; \
	if [ -n "$$ids" ]; then docker rm -f $$ids; else echo "no ad-hoc slots"; fi

status: ## live model of each server
	scripts/model.sh status

models: ## the router's union of models
	curl -s localhost:$(ROUTER_PORT)/v1/models

health: ## per-server /health (curl, not wget: the TEI image has no wget)
	@printf 'reranker  '; docker compose exec -T reranker curl -fsS http://127.0.0.1:80/health 2>/dev/null || printf 'down'; echo
	@printf 'nano      '; docker compose exec -T nano python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:80/health')" >/dev/null 2>&1 && echo ok || echo down

logs: ## follow one service: make logs SVC=reranker
	docker compose logs -f $(SVC)

load: ## swap the reranker model: make load MODEL=BAAI/bge-reranker-v2-m3
	scripts/model.sh load $(MODEL)

unload: ## stop the reranker and free its VRAM
	scripts/model.sh unload

run: ## start a free ad-hoc TEI slot: make run MODEL=... [NAME=] [GPU=] [PORT=]
	scripts/run.sh "$(MODEL)" $(if $(NAME),--name $(NAME)) $(if $(GPU),--gpu $(GPU)) $(if $(PORT),--port $(PORT))

stop: ## remove an ad-hoc slot: make stop NAME=tei-...
	scripts/stop.sh "$(NAME)"
