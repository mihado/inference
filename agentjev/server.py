# AgentJev behind the stack's router: a TEI-shaped /rerank plus the native API.
#
# AgentJev is a decision model — Qwen3-0.6B with the LM head removed and a
# trained candidate head. It answers boolean/choice/score questions with full
# distributions and decodes zero tokens, so like Laya, TEI and vLLM cannot host
# it and it needs this small service. Two surfaces (mirroring laya/):
#
#   POST /rerank         TEI's shape ({query, texts}). One boolean question per
#                        document ("does it help answer the query?"), all sent
#                        as one native batch call (32 states max — the upstream
#                        batch limit, so no chunking). The router already proxies
#                        /rerank to whatever /info names.
#   POST /api/evaluate   the native API ({state, questions} or batched
#                        {requests}). The router forwards it verbatim.
#
# Upstream code (malevrigns/agent-jev, pinned commit in the Dockerfile) owns the
# model and the contract: DecisionEngine.evaluate takes the raw payload and
# returns the exact upstream response shape. This file owns only the bootstrap
# (weights, skeleton, wrapped checkpoint — all idempotent, all into the shared
# HF cache) and the stack surfaces. No outer lock: the engine locks internally.
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
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse

REPO = os.environ.get("AGENTJEV_REPO", "aimeigaoshou/agent-jev")
BASE = os.environ.get("AGENTJEV_BASE", "Qwen/Qwen3-0.6B")
DEVICE = os.environ.get("AGENTJEV_DEVICE", "").strip()
# One /rerank call carries at most this many documents: the native batch limit
# is 32 states per call (contract.py: prepare), so the bound needs no chunking.
MAX_TEXTS = int(os.environ.get("AGENTJEV_MAX_TEXTS", "32"))
SERVED_ID = REPO

RERANK_QUESTION = "Does the document help answer the query?"
RERANK_CRITERIA = {
    "true": "the document helps answer the query",
    "false": "the document does not help answer the query",
}
RERANK_STATE = "Query: %s\n\nDocument: %s"

engine: Optional[Any] = None


def error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": code, "code": code}},
    )


def _bootstrap() -> tuple:
    """Weights, skeleton, temperatures, wrapped checkpoint. The tokenizer loads
    offline-only (upstream engine.py), so the base snapshot is pre-seeded here
    instead of left to first use; the wrapped torch checkpoint is reused across
    restarts. Every step skips when its artifact already exists."""
    from huggingface_hub import constants as hub_constants
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file

    cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip() or hub_constants.HF_HUB_CACHE
    print("[agentjev] seeding %s ..." % BASE, flush=True)
    snapshot_download(BASE, cache_dir=cache)
    print("[agentjev] downloading weights ...", flush=True)
    src = hf_hub_download(REPO, "model.safetensors", cache_dir=cache)
    temps = hf_hub_download(REPO, "temperatures.json", cache_dir=cache)
    ckpt = os.path.join(cache, "agentjev_v1.pt")
    if not os.path.exists(ckpt):
        print("[agentjev] wrapping torch checkpoint ...", flush=True)
        torch.save({"state_dict": load_file(src)}, ckpt)
    return ckpt, temps


def _load() -> None:
    global engine
    from jev_service.engine import DecisionEngine

    device = DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt, temps = _bootstrap()
    engine = DecisionEngine(ckpt, BASE, device, 2048, temperatures=temps)
    print(
        "[agentjev] %s ready on %s (sha %s)"
        % (SERVED_ID, device, engine.checkpoint_sha256[:12]),
        flush=True,
    )


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

    threading.Thread(target=run, daemon=True, name="agentjev-load").start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness only. Readiness is /info: it answers once the model is loaded."""
    return {"status": "ok"}


@app.get("/info")
def info():
    if engine is None:
        return error(503, "the model is still loading", "model_loading")
    # TEI's /info shape. The router reads model_id from it and registers this
    # container under that one model, on the next 30s scan.
    return {
        "model_id": SERVED_ID,
        "model_dtype": "bfloat16",
        "device": engine.device,
        "max_client_batch_size": MAX_TEXTS,
    }


@app.post("/rerank")
def rerank(body: Dict[str, Any]):
    if engine is None:
        return error(503, "the model is still loading", "model_loading")
    model = body.get("model") or body.get("model_id")
    if model is not None and model != SERVED_ID:
        return error(404, "No backend serves model '%s'." % model, "model_not_found")
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
    question = {
        "id": "relevant",
        "type": "boolean",
        "question": RERANK_QUESTION,
        "criteria": RERANK_CRITERIA,
    }
    payload = {
        "requests": [
            {"state": RERANK_STATE % (query, text), "questions": [question]} for text in texts
        ]
    }
    try:
        out = engine.evaluate(payload)
    except (ValueError, TypeError, KeyError) as exc:
        # ValueError is also the over-length refusal (>2048 tokens): an error,
        # never a silent truncation.
        return error(400, str(exc), "invalid_request_error")
    scores = [result["answers"][0]["probability"] for result in out["results"]]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return {
        "results": [
            {"index": int(i), "relevance_score": round(float(scores[i]), 4)} for i in order
        ]
    }


@app.post("/api/evaluate")
def api_evaluate(body: Dict[str, Any]):
    if engine is None:
        return error(503, "the model is still loading", "model_loading")
    model = body.get("model") or body.get("model_id")
    if model is not None and model != SERVED_ID:
        return error(404, "No backend serves model '%s'." % model, "model_not_found")
    try:
        # The exact upstream response shape (api_version, results, usage):
        # prepare() ignores the router's `model` key like any unknown key.
        return engine.evaluate(body)
    except (ValueError, TypeError, KeyError) as exc:
        return error(400, str(exc), "invalid_request_error")
