# Laya behind the stack's router: a TEI-shaped /rerank plus the native typed API.
#
# Laya is neither an embedder nor a cross-encoder: it answers typed questions about
# one state in a single forward pass, so TEI and vLLM cannot host it and it needs
# this small service. Two surfaces:
#
#   POST /rerank         TEI's shape ({query, texts}). The router already proxies
#                        /rerank to whatever /info names, so a Laya slot is
#                        discovered exactly like a TEI slot — no router change.
#   POST /v1/decisions   the native API ({state, questions}) for guardrails,
#                        triage, moderation. The router forwards it verbatim.

import os
import threading
from typing import Any, Dict, Optional

import numpy as np
import torch

import laya
import stack_service
from laya.common import QTYPES, build_sequence, collate_items, temp_bucket
from laya.presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)

MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
SUBFOLDER = os.environ.get("LAYA_SUBFOLDER", "").strip()
DEVICE = os.environ.get("LAYA_DEVICE", "").strip()
# One request carries at most this many documents through one forward pass. The
# rerankers in this stack use the same bound (--max-client-batch-size 64).
MAX_TEXTS = int(os.environ.get("LAYA_MAX_TEXTS", "64"))
SERVED_ID = "%s/%s" % (MODEL, SUBFOLDER) if SUBFOLDER else MODEL

PRESETS = {
    "triage": triage_questions,
    "email": email_questions,
    "guard": guard_questions,
    "moderation": moderation_questions,
    "router": router_questions,
}

# One noul question per document, never one choice question over all documents.
# Choice options share one head_max_len budget (common.py: build_sequence), so at
# 30 candidates each document keeps ~5 tokens and stops being distinguishable —
# the collapse the model card reports for Banking77. The fixed two-option noul
# head leaves the whole state budget for query + document. See README, "Laya".
RERANK_QUESTION = {
    "t": "noul",
    "ins": "Is the document relevant to the query: does it directly help answer it?",
    "crit": {
        "false": "the document does not help answer the query",
        "true": "the document helps answer the query",
    },
}
RERANK_STATE = "Query: %s\n\nDocument: %s"

agent: Optional[Any] = None
# One checkpoint, one forward at a time: /rerank and /v1/decisions share the model.
MODEL_LOCK = threading.Lock()


def _load() -> None:
    global agent
    agent = laya.load(MODEL, device=DEVICE or None, subfolder=SUBFOLDER or None)
    print(
        "[laya] %s ready on %s (%s)"
        % (SERVED_ID, agent.device, str(agent.dtype).replace("torch.", "")),
        flush=True,
    )


app = stack_service.make_app(_load, "laya-load")


@app.get("/info")
def info():
    return stack_service.info_response(
        loaded=agent is not None,
        model_id=SERVED_ID,
        max_client_batch_size=MAX_TEXTS,
        extra=lambda: {
            "model_dtype": str(agent.dtype).replace("torch.", ""),
            "device": str(agent.device),
            "paths": ["/rerank", "/v1/decisions"],
        },
    )


def _scores(agent, query: str, texts: list) -> np.ndarray:
    """P(relevant) per document, every document in ONE forward pass.

    Mirrors Agent.system_one: build a sequence per question, collate them into a
    batch, one model call. Here the "questions" are the per-document noul items,
    so 30 candidates cost one pass, not 30.
    """
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items = []
    for text in texts:
        seq, markers = build_sequence(
            agent.tok, RERANK_STATE % (query, text), RERANK_QUESTION, max_len, head_max_len
        )
        if len(markers) != len(RERANK_QUESTION["crit"]):
            raise ValueError("the question's options exceed head_max_len=%d" % head_max_len)
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES["noul"]})

    b = collate_items([items], agent.tok.pad_token_id)
    use_amp = agent.device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=agent.device.type, dtype=agent.dtype, enabled=use_amp
    ):
        logits, _act = agent.model(
            b["input_ids"].to(agent.device),
            b["attention_mask"].to(agent.device),
            b["marker_pos"].to(agent.device),
            b["marker_mask"].to(agent.device),
            b["qtype"].to(agent.device),
        )

    # Same calibration the SDK applies to its published answers (agent.py): one
    # temperature per (qtype, option-count) bucket. It cannot change the order —
    # every document shares it — but a threshold on the score means nothing
    # without it.
    t_scale = agent.temperature_by_options.get(
        temp_bucket(QTYPES["noul"], 2), agent.temperature[QTYPES["noul"]]
    )
    z = logits.float().cpu().numpy() / t_scale
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=1, keepdims=True)
    return p[:, 1]  # marker 1 is "true": render_options puts false first, true second


@app.post("/rerank")
def rerank(body: Dict[str, Any]):
    if agent is None:
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
            scores = _scores(agent, query, texts)
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        return stack_service.error(400, str(exc), "invalid_request_error")
    order = np.argsort(-scores, kind="stable")
    return {
        "results": [
            {"index": int(i), "relevance_score": round(float(scores[i]), 4)} for i in order
        ]
    }


@app.post("/v1/decisions")
def decisions(body: Dict[str, Any]):
    if agent is None:
        return stack_service.error(503, "the model is still loading", "model_loading")
    err = stack_service.check_model_match(body, SERVED_ID)
    if err is not None:
        return err
    state = body.get("state")
    if not isinstance(state, (str, dict, list)):
        return stack_service.error(
            400,
            "'state' is required: a string, an object, or an array of turns.",
            "invalid_request_error",
        )
    questions = body.get("questions")
    if questions is None:
        preset = body.get("preset")
        preset_fn = PRESETS.get(preset) if isinstance(preset, str) else None
        if preset_fn is None:
            return stack_service.error(
                400,
                "Provide 'questions', or a 'preset': %s." % ", ".join(sorted(PRESETS)),
                "invalid_request_error",
            )
        questions = preset_fn()
    if not isinstance(questions, dict) or not questions:
        return stack_service.error(400, "'questions' must be a non-empty object.", "invalid_request_error")
    try:
        with MODEL_LOCK:
            return agent.predict(state, questions)
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
        return stack_service.error(400, str(exc), "invalid_request_error")
