# Jina embeddings v5-omni-small behind the stack's router: TEI-shaped discovery
# plus the OpenAI embeddings shape the router already forwards verbatim.
#
# jina-embeddings-v5-omni-small is a multimodal embedder (text, image, and —
# with the omni modality — audio and video) built on a Qwen3 tower family with
# LoRA task adapters: custom modeling code, so TEI cannot host it and
# sentence-transformers is the blessed path. vLLM can host it, but the weights
# are non-commercial, so this stays a local dev and eval tool where the plain
# path wins. Two surfaces:
#
#   POST /v1/embeddings   OpenAI's shape ({model, input}) plus three extensions
#                         that the router forwards verbatim:
#                           input — strings (text) or image items in OpenAI's
#                           chat shape: {type: image_url, image_url: {url}}.
#                           Only base64 data: URLs are accepted; the service
#                           never fetches a caller-supplied URL, the same rule
#                           the router holds.
#                           input_type — "document" (default) or "query",
#                           selecting the retrieval side. The model wants the
#                           Query:/Document: prefixes; callers should not have
#                           to hand-build them.
#                           dimensions — Matryoshka truncation (up to 1024);
#                           the model re-normalizes the shortened vector.
#
# Non-commercial model (CC-BY-NC-4.0): local dev and eval only, never serving.
#
# Readiness lives at /info, not /health: the router only registers a backend whose
# /info names a model, so a model that is still loading is simply not routable,
# while /health stays green for the compose healthcheck. A model that fails to
# load exits the process, so the container shows an exit code the way the TEI
# and vLLM services do.
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

import stack_service

MODEL = os.environ.get("JINA_EMBED_MODEL", "jinaai/jina-embeddings-v5-omni-small")
DEVICE = os.environ.get("JINA_EMBED_DEVICE", "").strip()
# The task adapter to load. Retrieval is the one behind /v1/embeddings'
# query/document split; the other three have no query side.
TASK = os.environ.get("JINA_EMBED_TASK", "retrieval")
# Which towers load. "vision" keeps text + image and skips the audio tower;
# "omni" adds audio and video; "text" is the smallest.
MODALITY = os.environ.get("JINA_EMBED_MODALITY", "vision")
# One request carries at most this many texts; over-length texts truncate at
# the model limit (upstream default), never refuse.
MAX_TEXTS = int(os.environ.get("JINA_EMBED_MAX_TEXTS", "64"))
MAX_DIMENSIONS = 1024
SERVED_ID = MODEL

model: Optional[Any] = None
# One checkpoint, one encode at a time.
MODEL_LOCK = threading.Lock()


def _decode_image_item(item: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[str]]:
    """Bytes for an OpenAI-shaped image item, or a refusal message.

    Only base64 `data:` URLs pass: fetching a caller-supplied URL would add the
    outbound-fetch surface the router deliberately does not have.
    """
    image_url = item.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else None
    if not isinstance(url, str):
        return None, "'image_url' items need {image_url: {url: 'data:...'}}."
    return stack_service.decode_data_url(url)


def _load() -> None:
    global model
    from sentence_transformers import SentenceTransformer

    device = DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        "[jina-embed] loading %s (%s, %s) on %s ..." % (SERVED_ID, TASK, MODALITY, device),
        flush=True,
    )
    model = SentenceTransformer(
        MODEL,
        trust_remote_code=True,
        device=device,
        model_kwargs={"default_task": TASK, "modality": MODALITY},
    )
    print("[jina-embed] %s ready on %s" % (SERVED_ID, device), flush=True)


app = stack_service.make_app(_load, "jina-embed-load")


@app.get("/info")
def info():
    if model is None:
        return stack_service.error(503, "the model is still loading", "model_loading")
    # TEI's /info shape. The router reads model_id from it and registers this
    # container under that one model, on the next 30s scan.
    return {
        "model_id": SERVED_ID,
        "model_dtype": "bfloat16",
        "device": str(model.device),
        "max_client_batch_size": MAX_TEXTS,
        "paths": ["/v1/embeddings"],
    }


@app.post("/v1/embeddings")
def embeddings(body: Dict[str, Any]):
    if model is None:
        return stack_service.error(503, "the model is still loading", "model_loading")
    err = stack_service.check_model_match(body, SERVED_ID)
    if err is not None:
        return err
    items: Union[str, Dict[str, Any], List[Any], None] = body.get("input")
    if isinstance(items, (str, dict)):
        items = [items]
    if not isinstance(items, list) or len(items) == 0:
        return stack_service.error(
            400,
            "'input' (string or array of strings) is required.",
            "invalid_request_error",
        )
    inputs: List[Any] = []
    for item in items:
        if isinstance(item, str):
            inputs.append(item)
            continue
        if isinstance(item, dict) and item.get("type") == "image_url":
            decoded, refusal = _decode_image_item(item)
            if decoded is None:
                return stack_service.error(400, refusal or "The image item is malformed.", "invalid_request_error")
            inputs.append(decoded)
            continue
        return stack_service.error(
            400,
            "'input' items must be strings or {type: 'image_url', image_url: {url: 'data:...'}} objects.",
            "invalid_request_error",
        )
    if len(inputs) > MAX_TEXTS:
        return stack_service.error(
            413,
            "At most %d inputs per request, got %d." % (MAX_TEXTS, len(inputs)),
            "invalid_request_error",
        )
    input_type = body.get("input_type", "document")
    if input_type not in ("query", "document"):
        return stack_service.error(
            400,
            "'input_type' (if given) must be 'query' or 'document'.",
            "invalid_request_error",
        )
    dimensions = body.get("dimensions")
    if dimensions is not None and (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions < 1
        or dimensions > MAX_DIMENSIONS
    ):
        return stack_service.error(
            400,
            "'dimensions' (if given) must be an integer between 1 and %d." % MAX_DIMENSIONS,
            "invalid_request_error",
        )
    # Retrieval is a two-sided task: the query and the document sides carry
    # different prefixes, so the side is chosen here instead of by the caller.
    encode = model.encode_query if input_type == "query" else model.encode_document
    kwargs = {} if dimensions is None else {"truncate_dim": dimensions}
    try:
        with MODEL_LOCK:
            vectors = encode(inputs, **kwargs)
    except (ValueError, TypeError, RuntimeError) as exc:
        return stack_service.error(400, str(exc), "invalid_request_error")
    return {
        "object": "list",
        "model": SERVED_ID,
        "data": [
            {"object": "embedding", "index": i, "embedding": [float(x) for x in vec]}
            for i, vec in enumerate(vectors)
        ],
    }
