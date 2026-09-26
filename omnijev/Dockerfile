# OmniJev: a Qwen3.5 backbone with LoRA adapters and a trained decision head,
# served by the upstream inference code (mso) cloned at a pinned commit — the
# custom head is why TEI and vLLM cannot host it. Apache-2.0 code and weights.
#
# Pins: torch cu126 (driver 575; README, "Image tag"). transformers 5.x is the
# Qwen3.5 line (5.17.0 verified by the CPU smoke), peft and accelerate come
# with the LoRA adapters, pillow serves the image inputs, and torchvision is
# required by the processor (Qwen3-VL builds its video sub-processor even for
# image-only use — found by the container smoke).
#
# fla-core (flash-linear-attention) is deliberately NOT installed: with it
# present, transformers' Qwen3.5 layers dispatch to its Triton kernels
# unconditionally, and on a CPU host that fails at request time with "0 active
# drivers" — the MSO_FLA=0 switch only guards the upstream wrapper, not
# transformers' own resolver. On a GPU host, add `fla-core==0.5.2` to the pip
# list for the fast kernels; the image then targets GPU only.
#
# Video states need ffmpeg and ffprobe on PATH; the first cut serves image
# states, so they are intentionally not installed.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.14.0 torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.53.0 \
    transformers==5.17.0 \
    peft==0.21.0 \
    accelerate==1.15.0 \
    pillow \
    huggingface_hub \
    safetensors \
    numpy
RUN git clone https://github.com/tinnel123666888/OmniJev /opt/omnijev \
    && git -C /opt/omnijev checkout 0676e20980a6b87063ff8205966b44be0e7fb526
ENV PYTHONPATH=/opt/omnijev
COPY server.py .
EXPOSE 80
# One worker: each worker process would load its own copy of the checkpoint.
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
