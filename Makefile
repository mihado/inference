# Convenience wrappers over docker compose + scripts/model.sh.
ROUTER_PORT ?= 8100
# TEI slots. The `nano` vLLM service is not a slot (see README).
SLOTS := 1 2 3 4 5 6

.PHONY: up down status models health logs load unload run stop

up: ## build (router) and start every slot
	docker compose up -d --build

down: ## stop and remove the stack
	docker compose down

status: ## live model per slot
	scripts/model.sh status

models: ## the router's union of models
	curl -s localhost:$(ROUTER_PORT)/v1/models

health: ## per-slot /health (curl, not wget: the TEI image ships no wget)
	@for n in $(SLOTS); do printf 'hf-%s  ' $$n; docker compose exec -T hf-$$n curl -fsS http://127.0.0.1:80/health 2>/dev/null || printf 'down'; echo; done
	@printf 'nano   '; docker compose exec -T nano python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:80/health')" >/dev/null 2>&1 && echo ok || echo down

logs: ## follow one service: make logs SVC=hf-3
	docker compose logs -f $(SVC)

load: ## swap a slot's model: make load SLOT=3 MODEL=BAAI/bge-reranker-v2-m3
	scripts/model.sh load $(SLOT) $(MODEL)

unload: ## stop a slot: make unload SLOT=3
	scripts/model.sh unload $(SLOT)

run: ## start an ad-hoc slot: make run MODEL=... [NAME=] [GPU=] [PORT=]
	scripts/run.sh "$(MODEL)" $(if $(NAME),--name $(NAME)) $(if $(GPU),--gpu $(GPU)) $(if $(PORT),--port $(PORT))

stop: ## remove an ad-hoc slot: make stop NAME=tei-...
	scripts/stop.sh "$(NAME)"
