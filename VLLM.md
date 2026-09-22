# vLLM notes — serving LLMs and VLMs on current hardware

Generic engine notes, not box-specific. vLLM serves autoregressive models (LLMs, vision-language models, embedding models in pooling mode) behind an OpenAI-compatible server, a raw CLI, or a Python offline engine. The two ideas that matter are PagedAttention (KV cache in non-contiguous blocks, so memory stops fragmenting and long contexts stop OOMing) and continuous batching (requests join and leave the batch per decode step instead of waiting for the slowest). Everything else is configuration around those two. Docs at https://docs.vllm.ai, code at https://github.com/vllm-project/vllm, the PagedAttention paper at https://arxiv.org/abs/2309.06180.

## Server shapes

`vllm serve` exposes `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` — any OpenAI client works unmodified, which is why agents and harnesses can point at local models with one URL change. `--runner pooling` turns the same binary into an embedder (this is how voyage-4-nano serves here). The offline engine (`LLM` class) skips HTTP for batch jobs and eval harnesses. The native `/v1` surface never streams unless asked (`stream: true`); the router in this repo buffers full responses, so anything needing token streaming stays out of it.

## Knobs that matter

`--gpu-memory-utilization` (fraction of VRAM vLLM may claim; the complement stays for neighbors on a shared box), `--max-model-len` (context ceiling; KV cache scales with it, so a 32k window on 16 GB needs the utilization knob set with eyes open), `--dtype` (`auto` usually resolves bf16 on Ampere and later; force it when it guesses wrong), `--enforce-eager` (disables CUDA graphs: slower per token, but the only mode that works on some models and the honest fallback before blaming anything else), `--trust-remote-code` (required for models with custom code — a supply-chain decision, pin the revision), `--tensor-parallel-size` for multi-GPU (needs NCCL across cards that match; mismatched cards train and serve badly, keep pairs identical), `--max-num-seqs` and `--max-num-batched-tokens` (throughput vs latency: high values favor bulk evals, low values favor interactive agents). Quantized serving (AWQ, GPTQ, FP8, GGUF via llama.cpp instead) trades VRAM for small quality loss — measure on the eval, not the leaderboard.

## Structured output

vLLM enforces JSON schemas and grammars at decode time (xgrammar backend): schema-valid output by construction, which removes an entire class of parse-repair code from agent loops and decomposition pipelines. Prefer enforcement over prompt pleading wherever the schema is fixed — the decomposition work should lean on this before training format obedience into any model.

## Operations

Pin the image tag (this stack pins `v0.16.0`; upgrades move model support and kernel behavior). First token after a cold start is slow (weights load, graphs capture, LoRA adapters attach) — healthchecks must distinguish loading from broken, same rule as every other service here. The `/metrics` endpoint exposes queue depth, tokens per second, and cache pressure; watch queue depth first when agents complain about latency. CUDA-graph failures and dtype mismatches cause most of the confusing errors; `--enforce-eager` plus explicit `--dtype` resolves most of them before any deeper debugging earns its time.
