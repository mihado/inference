# Jina embeddings v5-omni-small: a Qwen3-family multimodal embedder (text,
# image, and — with the omni modality — audio and video) with custom modeling
# code and LoRA task adapters, served by sentence-transformers, the blessed
# path for this checkpoint. Sibling to jina/ (the reranker).
#
# Pins: torch cu126 (driver 575; README, "Image tag"). transformers 5.x and
# sentence-transformers 6.x are the line this checkpoint targets (its config
# says transformers >= 5.1; the v3-era 4.x line does not apply here). peft and
# torchvision are required by the custom code and the processor — peft by the
# task adapters, torchvision because the processor builds Qwen3-VL's video
# sub-processor even for image-only use; both found by the CPU smoke test.
# Remote code loads from the hub checkpoint itself (trust_remote_code); pin the
# model revision for supply-chain stability before any use past local dev.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.0 torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.53.0 \
    transformers==5.17.0 \
    sentence-transformers==6.1.0 \
    peft \
    pillow \
    huggingface_hub \
    safetensors \
    numpy
COPY server.py .
EXPOSE 80
# One worker: each worker process would load its own copy of the checkpoint.
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
