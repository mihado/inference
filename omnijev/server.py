# OmniJev behind the stack's router: TEI-shaped discovery plus the native
# system_one surface.
#
# OmniJev is an omni-modal System One decision model — a Qwen3.5 backbone with
# LoRA adapters and a trained decision head that answers typed questions about
# one image in a single forward pass with zero generated tokens: `choice` with
# per-option probabilities and an explicit `abstain`, `noul` booleans, and
# `score` on an ordered scale. TEI and vLLM cannot host a custom head, so it
# runs this small service. One surface:
#
#   POST /v1/systemone   {model, state, questions} -> {model, answers}. The
#                        router forwards it verbatim, like /v1/decisions and
#                        /api/evaluate.
#
# state.images entries are base64 `data:` URLs, materialized to temporary files
# because the upstream code reads images from disk. Remote URLs are refused —
# no outbound fetch of caller-supplied URLs, the same rule the router holds.
# This upstream release reads state.images[0] (an optional 4x4 video mosaic),
# so more than one image is refused rather than silently ignored.
#
# Apache-2.0 code and weights: unlike the Jina profiles, this service may serve.
#
# Readiness lives at /info, not /health: the router only registers a backend
# whose /info names a model, so a model that is still loading is simply not
# routable, while /health stays green for the compose healthcheck. A model that
# fails to load exits the process, the way the TEI and vLLM services do.
import base64
import binascii
import os
import shutil
import tempfile
import threading
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# One code base, one API, three sizes; 0.8B is the default because it fits any
# card. An explicit OMNIJEV_CKPT / OMNIJEV_BASE pair beats the size catalog —
# a local directory or a mirror both work.
SIZES = {
    "0.8": ("tinnel123/OmniJev-0.8B", "Qwen/Qwen3.5-0.8B"),
    "2": ("tinnel123/OmniJev-2B", "Qwen/Qwen3.5-2B"),
    "4": ("tinnel123/OmniJev", "Qwen/Qwen3.5-4B"),
}
SIZE = os.environ.get("OMNIJEV_SIZE", "0.8")
_pair = SIZES.get(SIZE)
CKPT = os.environ.get("OMNIJEV_CKPT") or (_pair[0] if _pair is not None else "")
BASE = os.environ.get("OMNIJEV_BASE") or (_pair[1] if _pair is not None else "")
if not CKPT or not BASE:
    raise SystemExit(
        "OMNIJEV_SIZE must be one of %s, or set OMNIJEV_CKPT and OMNIJEV_BASE."
        % ", ".join(SIZES)
    )
SERVED_ID = CKPT
# Upstream's default image budget: 768 vision tokens.
MAX_PIXELS = int(os.environ.get("OMNIJEV_MAX_PIXELS") or str(768 * 28 * 28))
MAX_QUESTIONS = int(os.environ.get("OMNIJEV_MAX_QUESTIONS", "32"))

model: Optional[Any] = None
# One model copy, one system_one at a time: the upstream wrapper keeps
# per-request state on the instance, so threads must not share a forward.
# A second replica is a separate service on the other card (`make up-omnijev
# CONCURRENCY=2`), with its own copy and its own lock.
MODEL_LOCK = threading.Lock()


def error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def _decode_image(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Bytes for a base64 image data URL, or a refusal message.

    Only `data:` URLs pass: fetching a caller-supplied URL would add the
    outbound-fetch surface the router deliberately does not have.
    """
    header, separator, payload = url.partition(",")
    if not url.startswith("data:") or separator == "" or ";base64" not in header:
        return None, "Only base64 'data:' URLs are accepted; no outbound fetch."
    try:
        return base64.b64decode(payload, validate=True), None
    except (binascii.Error, ValueError):
        return None, "The data URL payload is not valid base64."


def _load() -> None:
    global model
    from huggingface_hub import snapshot_download

    from mso.infer import MSO1

    ckpt_dir = CKPT
    if not os.path.isdir(ckpt_dir):
        cache = os.environ.get("HUGGINGFACE_HUB_CACHE") or None
        ckpt_dir = snapshot_download(CKPT, cache_dir=cache)
    print("[omnijev] loading %s (base %s) ..." % (CKPT, BASE), flush=True)
    model = MSO1(ckpt_dir, BASE, max_pixels=MAX_PIXELS)
    print(
        "[omnijev] %s ready on %s" % (SERVED_ID, model.dev),
        flush=True,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    def run() -> None:
        try:
            _load()
        except BaseException:
            traceback.print_exc()
            # A model that will not load is a stopped container: `docker ps`
            # shows the restart, `docker logs` shows why.
            os._exit(1)

    threading.Thread(target=run, daemon=True, name="omnijev-load").start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness only. Readiness is /info: it answers once the model is loaded."""
    return {"status": "ok"}


@app.get("/info")
def info():
    if model is None:
        return error(503, "the model is still loading", "model_loading")
    # TEI's /info shape. The router reads model_id from it and registers this
    # container under that one model, on the next 30s scan.
    return {
        "model_id": SERVED_ID,
        "model_dtype": str(model.dtype).replace("torch.", ""),
        "device": str(model.dev),
        "max_client_batch_size": MAX_QUESTIONS,
    }


@app.post("/v1/systemone")
def systemone(body: Dict[str, Any]):
    if model is None:
        return error(503, "the model is still loading", "model_loading")
    req_model = body.get("model")
    if req_model is not None and req_model != SERVED_ID:
        return error(404, "No backend serves model '%s'." % req_model, "model_not_found")
    state = body.get("state")
    if not isinstance(state, dict):
        return error(
            400,
            "'state' (an object with 'images') is required.",
            "invalid_request_error",
        )
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        return error(
            400,
            "'questions' (a non-empty object) is required.",
            "invalid_request_error",
        )
    if len(questions) > MAX_QUESTIONS:
        return error(
            413,
            "At most %d questions per request, got %d." % (MAX_QUESTIONS, len(questions)),
            "invalid_request_error",
        )
    images = state.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        return error(
            400,
            "'state.images' must carry exactly one base64 data: URL — this "
            "release reads images[0], and more are refused rather than "
            "silently ignored.",
            "invalid_request_error",
        )
    decoded, refusal = _decode_image(images[0])
    if decoded is None:
        return error(400, refusal or "The image is malformed.", "invalid_request_error")
    workdir = tempfile.mkdtemp(prefix="omnijev-")
    try:
        path = os.path.join(workdir, "state.png")
        with open(path, "wb") as handle:
            handle.write(decoded)
        call_state: Dict[str, Any] = {"images": [path]}
        if "video" in state:
            call_state["video"] = state["video"]
        with MODEL_LOCK:
            answers = model.system_one(call_state, questions)
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        return error(400, str(exc), "invalid_request_error")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {"object": "systemone", "model": SERVED_ID, "answers": answers}
