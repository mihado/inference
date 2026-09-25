# Jina v3.5 listwise reranker: Qwen3-0.6B plus custom modeling code — the
# stack's fifth runtime (Python, not TEI/vLLM), sibling to laya/ and agentjev/.
#
# Pins mirror laya/: torch cu126 (driver 575; README, "Image tag"),
# transformers 5.17. The custom modeling code loads from the hub checkpoint
# itself (trust_remote_code); if a future checkpoint needs newer transformers
# than the pin, the import fails loudly at container start, not silently.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.53.0 \
    transformers==5.17.0 \
    huggingface_hub \
    safetensors \
    numpy
COPY server.py .
EXPOSE 80
# One worker: each worker process would load its own copy of the checkpoint.
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
