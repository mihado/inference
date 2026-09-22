# Ollama service targets and model set, included by the root Makefile. Paths
# below are relative to the repo root: make always runs from there, and every
# command loads the root file first (the project dir), so -f paths resolve.
# Models Ollama keeps resident. `make ollama-deps` pulls every one; a singular
# pull stays manual: docker compose -f compose.yml -f ollama/compose.ollama.yml --profile ollama exec -T ollama ollama pull <model>.
MODELS = \
	gemma4:e4b \
	medgemma1.5:4b \
	ornith-1.5:9b \
	qwen3-embedding:0.6b \
	qwen3-embedding:4b \
	qwen3.5:9b

# Ollama defaults to the other card; the knob still works (flag beats env).
up-ollama: ## start the default stack plus Ollama (GPU 1, reference only)
	GPU_OLLAMA=$(or $(GPU),$(GPU_OLLAMA)) docker compose -f compose.yml -f ollama/compose.ollama.yml --profile ollama up -d --build

down-ollama: ## stop and remove Ollama only
	docker compose -f ollama/compose.ollama.yml --profile ollama down

ollama-deps: ## pull every model in ollama/ollama.mk (needs up-ollama)
	@for model in $(MODELS); do \
		echo "Pulling $$model..."; \
		docker compose -f compose.yml -f ollama/compose.ollama.yml --profile ollama exec -T ollama ollama pull $$model; \
	done
