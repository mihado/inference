# Ollama

Ollama runs GGUF models behind its own API, beside the router rather than behind it. One Ollama server holds many models with on-demand loading, while the router assumes one model per server with a static id — and the router buffers full responses, so token streaming would break passing through it. No label, no discovery, no streaming work: clients use port 11434 directly with the full `/api/*` surface. Reference only; rarely used these days.

```sh
make up-ollama                                # the default stack plus Ollama
make ollama-deps                              # pull every model in ollama/ollama.mk
docker compose -f compose.yml -f ollama/compose.ollama.yml --profile ollama exec -T ollama ollama run gemma4:e4b
```

Residency is bounded because the box is shared: `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_KEEP_ALIVE` (30m — covers a work session's gaps; any call extends it per-request), plus the `GPU_OLLAMA` pin (GPU 1 by default: GPU 0 collects the primaries plus every extras profile, and one card keeps each model whole instead of split across PCIe). Models live in Ollama's own blob store (`OLLAMA_HOME`); nothing dedupes with the HF cache. Upgrade by bumping `OLLAMA_IMAGE_TAG`.
