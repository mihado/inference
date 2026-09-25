# Jina v3.5 behind the stack's router: a TEI-shaped /rerank over its native API.
#
# jina-reranker-v3.5 is a listwise reranker — Qwen3-0.6B plus custom modeling
# code that ranks many documents jointly in one forward pass. It needs
# trust_remote_code and its own `rerank` method, so TEI cannot host it and it
# needs this small service. One surface:
#
#   POST /rerank         TEI's shape ({query, texts}). Passed straight into the
#                        native `model.rerank(query, documents)`, which already
#                        returns relevance-ordered results. The router already
#                        proxies /rerank to whatever /info names — no router
#                        change.
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
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get("JINA_MODEL", "jinaai/jina-reranker-v3.5")
DEVICE = os.environ.get("JINA_DEVICE", "").strip()
# One request carries at most this many documents in one listwise pass. The
# rerankers in this stack use the same bound (--max-client-batch-size 64).
MAX_TEXTS = int(os.environ.get("JINA_MAX_TEXTS", "64"))
SERVED_ID = MODEL

model: Optional[Any] = None
tokenizer: Optional[Any] = None
# One checkpoint, one forward at a time.
MODEL_LOCK = threading.Lock()


def error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def _load() -> None:
    global model, tokenizer
    device = DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[jina] loading %s on %s ..." % (SERVED_ID, device), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    print("[jina] %s ready on %s" % (SERVED_ID, device), flush=True)


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

    threading.Thread(target=run, daemon=True, name="jina-load").start()
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
        "device": str(next(model.parameters()).device),
        "max_client_batch_size": MAX_TEXTS,
    }


@app.post("/rerank")
def rerank(body: Dict[str, Any]):
    if model is None:
        return error(503, "the model is still loading", "model_loading")
    req_model = body.get("model") or body.get("model_id")
    if req_model is not None and req_model != SERVED_ID:
        return error(404, "No backend serves model '%s'." % req_model, "model_not_found")
    query = body.get("query")
    texts = body.get("texts")
    if texts is None:
        texts = body.get("documents")
    if not isinstance(query, str) or not isinstance(texts, list):
        return error(
            400,
            "'query' (string) and 'texts' (array of strings) are required.",
            "invalid_request_error",
        )
    if not all(isinstance(text, str) for text in texts):
        return error(400, "'texts' must be an array of strings.", "invalid_request_error")
    if len(texts) == 0:
        return {"results": []}
    if len(texts) > MAX_TEXTS:
        return error(
            413,
            "At most %d texts per request, got %d." % (MAX_TEXTS, len(texts)),
            "invalid_request_error",
        )
    try:
        with MODEL_LOCK:
            out = model.rerank(query, texts)
    except (ValueError, TypeError, RuntimeError) as exc:
        return error(400, str(exc), "invalid_request_error")
    results: List[Dict[str, Any]] = [
        {"index": int(item["index"]), "relevance_score": round(float(item["relevance_score"]), 4)}
        for item in out
    ]
    return {"results": results}
