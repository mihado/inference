# AgentJev decision model: Qwen3-0.6B backbone, custom candidate head — the
# stack's fourth runtime (Python, not TEI/vLLM), sibling to laya/.
#
# Pins mirror laya/: torch cu126 (driver 575; README, "Image tag"),
# transformers 5.17 (the combination the CPU smoke test verified against the
# upstream engine). Upstream has no PyPI package: its tree is cloned at a pinned
# commit and imported off PYTHONPATH. Weights, the Qwen3 skeleton, and the
# fitted temperatures download at container start into the shared HF cache
# (server.py bootstraps them; the tokenizer loads offline-only, so the skeleton
# is pre-seeded there rather than left to first use).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.53.0 \
    transformers==5.17.0 \
    huggingface_hub \
    safetensors \
    numpy
ARG AGENTJEV_SHA=b66a8cef7abdf66e6795eb229b55b387709f6ff7
RUN git clone https://github.com/malevrigns/agent-jev.git /opt/agent-jev \
    && git -C /opt/agent-jev checkout $AGENTJEV_SHA
ENV PYTHONPATH=/opt/agent-jev
COPY server.py .
EXPOSE 80
# One worker: each worker process would load its own copy of the checkpoint.
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
