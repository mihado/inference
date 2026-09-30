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

import os
from typing import Any, Dict, Optional

import torch

import stack_service

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


app = stack_service.make_app(_load, "agentjev-load")


@app.get("/info")
def info():
    return stack_service.info_response(
        loaded=engine is not None,
        model_id=SERVED_ID,
        max_client_batch_size=MAX_TEXTS,
        extra=lambda: {
            "model_dtype": "bfloat16",
            "device": engine.device,
            "paths": ["/rerank", "/api/evaluate"],
        },
    )


@app.post("/rerank")
def rerank(body: Dict[str, Any]):
    if engine is None:
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
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        # ValueError is also the over-length refusal (>2048 tokens): an error,
        # never a silent truncation.
        return stack_service.error(400, str(exc), "invalid_request_error")
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
        return stack_service.error(503, "the model is still loading", "model_loading")
    err = stack_service.check_model_match(body, SERVED_ID)
    if err is not None:
        return err
    try:
        # The exact upstream response shape (api_version, results, usage):
        # prepare() ignores the router's `model` key like any unknown key.
        return engine.evaluate(body)
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        return stack_service.error(400, str(exc), "invalid_request_error")
