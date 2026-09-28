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
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

import stack_service

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


app = stack_service.make_app(_load, "jina-load")


@app.get("/info")
def info():
    if model is None:
        return stack_service.error(503, "the model is still loading", "model_loading")
    # TEI's /info shape. The router reads model_id from it and registers this
    # container under that one model, on the next 30s scan.
    return {
        "model_id": SERVED_ID,
        "model_dtype": "bfloat16",
        "device": str(next(model.parameters()).device),
        "max_client_batch_size": MAX_TEXTS,
        "paths": ["/rerank"],
    }


@app.post("/rerank")
def rerank(body: Dict[str, Any]):
    if model is None:
        return stack_service.error(503, "the model is still loading", "model_loading")
    err = stack_service.check_model_match(body, SERVED_ID)
    if err is not None:
        return err
    parsed, err = stack_service.check_rerank_body(body, MAX_TEXTS)
    if err is not None:
        return err
    query, texts = parsed
    if len(texts) == 0:
        return {"results": []}
    try:
        with MODEL_LOCK:
            out = model.rerank(query, texts)
    except (ValueError, TypeError, RuntimeError) as exc:
        return stack_service.error(400, str(exc), "invalid_request_error")
    results: List[Dict[str, Any]] = [
        {"index": int(item["index"]), "relevance_score": round(float(item["relevance_score"]), 4)}
        for item in out
    ]
    return {"results": results}
