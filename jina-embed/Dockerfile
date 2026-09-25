# Jina embeddings v3: XLM-R embedder with custom modeling code (flash attention
# implementation plus task adapters) — served by sentence-transformers, the
# blessed path for this checkpoint. Sibling to jina/ (the reranker).
#
# Pins: torch cu126 (driver 575; README, "Image tag"). transformers stays on
# the 4.x line (here 4.57.6): the custom modeling code targets the 4.x module
# API and breaks on 5.x (`all_tied_weights_keys` was renamed). sentence-
# transformers 6.1.0 and einops (imported by the custom code) verified by the
# CPU smoke test. Remote code loads from the hub checkpoint itself
# (trust_remote_code); pin the model revision for supply-chain stability
# before any use past local dev.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.53.0 \
    transformers==4.57.6 \
    sentence-transformers==5.7.0 \
    einops \
    huggingface_hub \
    safetensors \
    numpy
# sentence-transformers stays below 6: 6.x requires transformers>=5, and the
# custom modeling code above needs the 4.x module API. Verified pair.
COPY server.py .
EXPOSE 80
# One worker: each worker process would load its own copy of the checkpoint.
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
