# Jina embeddings v3 behind the stack's router: TEI-shaped discovery plus the
# OpenAI embeddings shape the router already forwards verbatim.
#
# jina-embeddings-v3 is an XLM-R embedder with custom modeling code (flash
# attention implementation plus task adapters), so TEI cannot host it and it
# needs this small service. sentence-transformers is the blessed path for it.
# Two surfaces:
#
#   POST /v1/embeddings   OpenAI's shape ({model, input}). The router already
#                         forwards /v1/embeddings verbatim to whatever /info
#                         names — no router change.
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
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse

MODEL = os.environ.get("JINA_EMBED_MODEL", "jinaai/jina-embeddings-v3")
DEVICE = os.environ.get("JINA_EMBED_DEVICE", "").strip()
# One request carries at most this many texts; over-length texts truncate at
# the model limit (upstream default), never refuse.
MAX_TEXTS = int(os.environ.get("JINA_EMBED_MAX_TEXTS", "64"))
SERVED_ID = MODEL

model: Optional[Any] = None
# One checkpoint, one encode at a time.
MODEL_LOCK = threading.Lock()


def error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def _load() -> None:
    global model
    from sentence_transformers import SentenceTransformer

    device = DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[jina-embed] loading %s on %s ..." % (SERVED_ID, device), flush=True)
    model = SentenceTransformer(MODEL, trust_remote_code=True, device=device)
    print("[jina-embed] %s ready on %s" % (SERVED_ID, device), flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    def run() -> None:
        try:
            _load()
        except BaseException:
            traceback.print_exc()
            # A model that will not load is a stopped container, like TEI and vLLM:
            # `docker ps` shows the restart, `docker logs` shows why.
            os._exit(1)

    threading.Thread(target=run, daemon=True, name="jina-embed-load").start()
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
        "model_dtype": "bfloat16",
        "device": str(model.device),
        "max_client_batch_size": MAX_TEXTS,
    }


@app.post("/v1/embeddings")
def embeddings(body: Dict[str, Any]):
    if model is None:
        return error(503, "the model is still loading", "model_loading")
    req_model = body.get("model")
    if req_model is not None and req_model != SERVED_ID:
        return error(404, "No backend serves model '%s'." % req_model, "model_not_found")
    texts: Union[str, List[str], None] = body.get("input")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        return error(
            400,
            "'input' (string or array of strings) is required.",
            "invalid_request_error",
        )
    if len(texts) == 0:
        return error(400, "'input' must not be empty.", "invalid_request_error")
    if len(texts) > MAX_TEXTS:
        return error(
            413,
            "At most %d texts per request, got %d." % (MAX_TEXTS, len(texts)),
            "invalid_request_error",
        )
    try:
        with MODEL_LOCK:
            vectors = model.encode(texts)
    except (ValueError, TypeError, RuntimeError) as exc:
        return error(400, str(exc), "invalid_request_error")
    return {
        "object": "list",
        "model": SERVED_ID,
        "data": [
            {"object": "embedding", "index": i, "embedding": [float(x) for x in vec]}
            for i, vec in enumerate(vectors)
        ],
    }
