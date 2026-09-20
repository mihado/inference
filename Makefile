# Convenience wrappers over docker compose + scripts/model.sh.
ROUTER_PORT ?= 8090
SLOTS := 1 2 3 4 5 6

.PHONY: up down status models health logs load unload

up: ## build (router) and start every slot
	docker compose up -d --build

down: ## stop and remove the stack
	docker compose down

status: ## live model per slot
	scripts/model.sh status

models: ## the router's union of models
	curl -s localhost:$(ROUTER_PORT)/v1/models

health: ## per-slot /health
	@for n in $(SLOTS); do printf 'hf-%s  ' $$n; docker compose exec -T hf-$$n wget -qO- http://127.0.0.1:80/health 2>/dev/null || echo down; done

logs: ## follow one service: make logs SVC=hf-3
	docker compose logs -f $(SVC)

load: ## swap a slot's model: make load SLOT=3 MODEL=BAAI/bge-reranker-v2-m3
	scripts/model.sh load $(SLOT) $(MODEL)

unload: ## stop a slot: make unload SLOT=3
	scripts/model.sh unload $(SLOT)
