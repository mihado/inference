# Julia-1 behind the stack's router: TEI-shaped discovery plus the
# named-question decision surface.
#
# Julia-1 is a 144.3M-parameter decision model — an mmBERT-small encoder with a
# trained decision head — that scores supplied answer options for a typed
# question about a state in one forward pass with zero generated tokens:
# `choice` (2-20 options with full softmax probabilities), `noul` (P(true)),
# and `score` (the expected index on an ordered rubric). TEI and vLLM cannot
# host a custom head, so it runs this small service. One surface:
#
#   POST /v1/predict   {model, state, questions} -> {model, answers}. The router
#                      forwards it verbatim, like /v1/decisions and
#                      /v1/systemone.
#
# The upstream runtime lives in the model repository itself (SupersonicLabs/
# Julia-1): the code and the checkpoint download together at container start
# into the shared HF cache, and the package is imported off the snapshot path.
# There is no separate code repo to clone and no PyPI package to install.
#
# state is text only (Julia is not multimodal, unlike OmniJev); each question
# carries 2-20 options. Strict encoding is on: marker injection and any
# state/question/option truncation are refused, not silently truncated.
#
# Apache-2.0 code and weights: unlike the Jina profiles, this service may serve.

import os
import sys
import threading
from typing import Any, Dict, Optional

import stack_service

# The Hub repository doubles as the runtime: code and weights, one snapshot.
REPO = os.environ.get("JULIA_REPO") or "SupersonicLabs/Julia-1"
SERVED_ID = REPO
# Empty -> cuda when the container sees a GPU, else cpu. The runtime is
# CPU-capable; the GPU reservation is the stack's default shape.
DEVICE = os.environ.get("JULIA_DEVICE") or ""
MAX_LENGTH = int(os.environ.get("JULIA_MAX_LENGTH") or "8192")
HEAD_LENGTH = int(os.environ.get("JULIA_HEAD_LENGTH") or "512")
MAX_QUESTIONS = int(os.environ.get("JULIA_MAX_QUESTIONS") or "32")

model: Optional[Any] = None
# One model copy, one predict at a time: the engine keeps bounded encoding
# caches on the instance, so threads must not share a call. A second replica is
# a separate service on the other card (`make up-julia CONCURRENCY=2`), with its
# own copy and its own lock.
MODEL_LOCK = threading.Lock()


def _device() -> str:
    if DEVICE:
        return DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load() -> None:
    global model
    from huggingface_hub import snapshot_download

    repo_dir = REPO
    if not os.path.isdir(repo_dir):
        cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or None
        repo_dir = snapshot_download(REPO, cache_dir=cache)
    # The runtime ships inside the checkpoint repository; import it off the
    # snapshot rather than installing it (there is no code-only download).
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from julia import load_model

    device = _device()
    print("[julia] loading %s on %s ..." % (SERVED_ID, device), flush=True)
    model = load_model(
        repo_dir,
        device=device,
        strict_encoding=True,
        max_length=MAX_LENGTH,
        head_length=HEAD_LENGTH,
    )
    print("[julia] %s ready" % SERVED_ID, flush=True)


app = stack_service.make_app(_load, "julia-load")


@app.get("/info")
def info():
    return stack_service.info_response(
        loaded=model is not None,
        model_id=SERVED_ID,
        max_client_batch_size=MAX_QUESTIONS,
        extra=lambda: {
            "device": _device(),
            "paths": ["/v1/predict"],
        },
    )


@app.post("/v1/predict")
def predict(body: Dict[str, Any]):
    if model is None:
        return stack_service.error(503, "the model is still loading", "model_loading")
    err = stack_service.check_model_match(body, SERVED_ID)
    if err is not None:
        return err
    state = body.get("state")
    if not isinstance(state, str) or not state:
        return stack_service.error(
            400,
            "'state' (non-empty text) is required; Julia-1 is text-only.",
            "invalid_request_error",
        )
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        return stack_service.error(
            400,
            "'questions' (a non-empty object) is required.",
            "invalid_request_error",
        )
    if len(questions) > MAX_QUESTIONS:
        return stack_service.error(
            413,
            "At most %d questions per request, got %d." % (MAX_QUESTIONS, len(questions)),
            "invalid_request_error",
        )
    try:
        with MODEL_LOCK:
            answers = model.predict(state=state, questions=questions)["answers"]
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        return stack_service.error(400, str(exc), "invalid_request_error")
    return {"object": "predict", "model": SERVED_ID, "answers": answers}
