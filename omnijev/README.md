# OmniJev

- OmniJev is an omni-modal System One decision model (Beijing Zhongguancun Academy / Institute of Automation, CAS / Zevo): a Qwen3.5 backbone with LoRA adapters and a trained decision head that answers typed questions about one image in one forward pass, zero generated tokens, with calibrated probabilities.
- Question types: `choice` (a probability per option plus `abstain` = P(none of the above) and a `valid` flag), `noul` (P(true)), and `score` (ordered levels). An option may be an image region — `{"region": {"box": [x1, y1, x2, y2]}}` in 0–1000 coordinates.
- Own Python service in `omnijev/`, in the `omnijev` profile, on GPU 0 — TEI and vLLM cannot host a custom head.
- Apache-2.0 code and weights throughout (the Qwen3.5 base included): this is the one decision profile that may serve. The model code is the upstream repo at a pinned commit in the Dockerfile.

| Surface | Shape | How the router treats it |
| --- | --- | --- |
| `GET /info` | TEI's `{model_id}` | discovery; `503` until the model is loaded, so an unloaded model is never routed |
| `GET /health` | liveness | compose healthcheck only |
| `POST /v1/systemone` | `{model, state, questions}` | forwarded verbatim; each answer carries its own `latency_s` |

Start it alone, or with everything:

```sh
make up-omnijev                           # the default stack plus OmniJev (0.8B)
make up-omnijev GPU=1                     # same, on the other card
```

One code base, one API, three sizes:

| `OMNIJEV_SIZE` | Checkpoint | Base |
| --- | --- | --- |
| `0.8` (default) | `tinnel123/OmniJev-0.8B` | `Qwen/Qwen3.5-0.8B` |
| `2` | `tinnel123/OmniJev-2B` | `Qwen/Qwen3.5-2B` |
| `4` | `tinnel123/OmniJev` | `Qwen/Qwen3.5-4B` |

`OMNIJEV_CKPT` / `OMNIJEV_BASE` override the pair directly (a local directory works). The served model id is the checkpoint id, and clients send that name.

Clients keep using the router port:

```sh
curl -s localhost:8100/v1/systemone -H 'content-type: application/json' -d '{
  "model": "tinnel123/OmniJev-0.8B",
  "state": {"images": ["data:image/png;base64,..."]},
  "questions": {
    "color": {"type": "choice", "instructions": "What color is the square?",
              "criteria": {"red": null, "blue": null}},
    "red": {"type": "noul", "instructions": "The image is a red square."}}}'
```

One answer per question: `choice` with `probabilities`, `abstain`, `valid`, and `confidence`; `noul` with its probability; `score` with levels and confidence. `confidence` is Jev's `(K·p_max − 1)/(K − 1)`.

### Honest limits

- One image per request in this release (`state.images[0]`; more are refused, not ignored). Video states exist upstream — a 4×4 frame mosaic plus a `video` object — and need ffmpeg; not wired here yet.
- Images arrive as base64 `data:` URLs; remote URLs are refused, so the service never fetches a caller-supplied URL.
- `OMNIJEV_MAX_PIXELS` bounds the image budget (default: 768 vision tokens); `OMNIJEV_MAX_QUESTIONS` (32) bounds a request.
- Fresh upstream (released 2026-09-25, latency measured on an A800): eval-grade until measured on this box.
- `fla-core` (fast Triton kernels for the linear-attention layers) is deliberately not in the image — with it present, transformers dispatches to Triton unconditionally and CPU hosts fail at request time with `0 active drivers`. Add `fla-core==0.5.2` to the Dockerfile for a GPU-only image; `MSO_FLA=0` then disables the upstream wrapper.
