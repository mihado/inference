# Shared bootstrap for the stack's Python model services (laya, agentjev, jina,
# jina-embed, omnijev, julia).
#
# Every service is one checkpoint behind the same surfaces: TEI-shaped
# discovery at /info, liveness at /health, and its own POST endpoints. This
# module owns the identical parts — error envelope, app bootstrap, readiness
# shape, model-match and rerank validation, base64 data-URL decoding — so one
# change applies six times. Model logic stays in each server.py.
#
# Rules: nothing here imports torch or any model package, so it stays light.
# Behavior-preserving by construction: messages and shapes below are copied
# verbatim from the services they replace.
import base64
import binascii
import os
import threading
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse


def error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def make_app(loader, thread_name: str) -> FastAPI:
    """A service app: background load, liveness at /health.

    Readiness lives at /info, not /health: the router only registers a
    backend whose /info names a model, so a loading model is simply not
    routable, while /health stays green for the compose healthcheck. A model
    that fails to load exits the process, the way the TEI and vLLM services
    do: `docker ps` shows the restart, `docker logs` shows why.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        def run() -> None:
            try:
                loader()
            except BaseException:
                traceback.print_exc()
                os._exit(1)

        threading.Thread(target=run, daemon=True, name=thread_name).start()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> Dict[str, str]:
        """Liveness only. Readiness is /info: it answers once the model is loaded."""
        return {"status": "ok"}

    return app


def info_response(
    *,
    loaded: bool,
    model_id: str,
    max_client_batch_size: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """TEI's /info shape. The router reads model_id from it and registers the
    container under that one model, on the next 30s scan."""
    if not loaded:
        return error(503, "the model is still loading", "model_loading")
    return {"model_id": model_id, **(extra or {}), "max_client_batch_size": max_client_batch_size}


def check_model_match(body: Dict[str, Any], served_id: str, keys: Tuple[str, ...] = ("model", "model_id")) -> Any:
    """The 404 when the request names a different model, else None."""
    model = None
    for key in keys:
        if body.get(key) is not None:
            model = body.get(key)
            break
    if model is not None and model != served_id:
        return error(404, "No backend serves model '%s'." % model, "model_not_found")
    return None


def check_rerank_body(
    body: Dict[str, Any], max_texts: int
) -> Tuple[Optional[Tuple[str, List[str]]], Any]:
    """TEI's ({query, texts}) shape validated: ((query, texts), None), or
    (None, error response). Accepts Cohere's `documents` for `texts`."""
    query = body.get("query")
    texts = body.get("texts")
    if texts is None:
        texts = body.get("documents")
    if not isinstance(query, str) or not isinstance(texts, list):
        return None, error(
            400,
            "'query' (string) and 'texts' (array of strings) are required.",
            "invalid_request_error",
        )
    if not all(isinstance(text, str) for text in texts):
        return None, error(400, "'texts' must be an array of strings.", "invalid_request_error")
    if len(texts) > max_texts:
        return None, error(
            413,
            "At most %d texts per request, got %d." % (max_texts, len(texts)),
            "invalid_request_error",
        )
    return (query, texts), None


def decode_data_url(url: str, max_bytes: int = 25_000_000) -> Tuple[Optional[bytes], Optional[str]]:
    """Bytes for a base64 image data URL, or a refusal message.

    Only `data:` URLs pass: fetching a caller-supplied URL would add the
    outbound-fetch surface the router deliberately does not have. Payloads
    past max_bytes are refused before decoding, so one oversized image
    cannot balloon memory (a photo at the services' pixel budgets decodes
    to single-digit megabytes; 25MB is headroom, not a target)."""
    header, separator, payload = url.partition(",")
    if not url.startswith("data:") or separator == "" or ";base64" not in header:
        return None, "Only base64 'data:' URLs are accepted; no outbound fetch."
    # Compared before decoding, so an oversized image cannot balloon memory.
    # The payload is base64 text, so this bounds characters, not decoded bytes;
    # base64 is 4/3 the size it encodes, so the decoded cap is ~3/4 of max_bytes.
    if len(payload) > max_bytes:
        return None, "The image data exceeds %d base64 characters." % max_bytes
    try:
        return base64.b64decode(payload, validate=True), None
    except (binascii.Error, ValueError):
        return None, "The data URL payload is not valid base64."
