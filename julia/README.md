# Julia-1

- Julia-1 is a 144.3M-parameter decision model (Supersonic Labs): an mmBERT-small multilingual encoder with a trained decision head that scores supplied answer options for a typed question about a state, one forward pass, zero generated tokens.
- Question types: `choice` (2-20 options, winning id plus full softmax probabilities), `noul` (P(true)), and `score` (the expected zero-based index on an ordered rubric). Caller-defined ids are returned unchanged.
- Own Python service in `julia/`, in the `julia` profile, on GPU 0 — TEI and vLLM cannot host a custom head.
- Apache-2.0 code and weights: this service may serve. The runtime ships inside the model repository; the container downloads the repository (code plus the 550 MiB FP32 checkpoint) at start into the shared HF cache and imports it off the snapshot.

| Surface | Shape | How the router treats it |
| --- | --- | --- |
| `GET /info` | TEI's `{model_id}` | discovery; `503` until the model is loaded, so an unloaded model is never routed |
| `GET /health` | liveness | compose healthcheck only |
| `POST /v1/predict` | `{model, state, questions}` | forwarded verbatim; answers carry `type`, `probabilities`, and `max_probability` |

Start it alone, or with everything:

```sh
make up-julia                             # the default stack plus Julia-1
make up-julia GPU=1                       # same, on the other card
make up-julia CONCURRENCY=2               # both cards (ports 8047/8048)
```

Clients keep using the router port:

```sh
curl -s localhost:8100/v1/predict -H 'content-type: application/json' -d '{
  "model": "SupersonicLabs/Julia-1",
  "state": "I was charged twice for the same order.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "Billing and payment disputes",
                          "shipping": "Shipping and delivery",
                          "access": "Account access and login"}},
    "urgent": {"type": "noul", "instructions": "This needs same-day attention."}}}'
```

One answer per question: `choice` with `probabilities` and `max_probability`, `score` with its expected index and per-level probabilities, `noul` with its probability. Questions are independently scored in a batch.

### Knobs

| Variable | Default | Effect |
| --- | --- | --- |
| `JULIA_REPO` | `SupersonicLabs/Julia-1` | the Hub repository (code + weights), or a local directory |
| `JULIA_DEVICE` | auto (`cuda` when visible, else `cpu`) | `cpu` runs without a GPU — the runtime is CPU-capable; `JULIA_CPU_THREADS` (default 4) tunes it |
| `JULIA_MAX_LENGTH` | `8192` | combined state/question/option token budget |
| `JULIA_HEAD_LENGTH` | `512` | the question-and-options budget (each option gets 48 tokens) |
| `JULIA_MAX_QUESTIONS` | `32` | per-request question cap |

### Honest limits

- **Text only.** Unlike OmniJev there is no image input; `state` is a string.
- **2-20 options per question** at the native call. Larger choice sets exist upstream behind a hierarchical `Router` (group, rerank survivors) whose grouped probabilities are conditional, not global — not wired here.
- **Strict encoding is on**: marker injection and any truncation are refused, not silently truncated.
- It compares the answers you provide; it does not supply missing facts or reason multi-step. The model card's own pilots show a weak spot on long label lists (Banking77: 64/100).
- Fresh upstream (released 2026-09-24): eval-grade until measured on this box; the card's numbers are CPU/GPU-mixed, so re-measure with the repo's harness if they matter.
- GPU builds carry a C compiler (`gcc` + `libc6-dev`) because transformers compiles Triton kernels at the first request; a slim image fails every decision with `Failed to find C compiler` until it is present. CPU runs never take that path.
- The runtime is imported from the downloaded snapshot (there is no PyPI package); keep the checkpoint files unchanged while an engine is loaded.
