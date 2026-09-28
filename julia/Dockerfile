# Julia-1: a 144.3M-parameter decision model (mmBERT-small encoder plus a
# trained decision head) served by its own Python runtime — the named-question
# interface (state + typed questions -> choice/score/noul answers) is why TEI
# and vLLM cannot host it. Apache-2.0 code and weights.
#
# There is no code repo to clone and no PyPI package: the runtime ships inside
# the model repository, so the container downloads the repository (code plus the
# 550 MiB FP32 checkpoint) at start into the shared HF cache and imports the
# package off the snapshot (server.py). The image carries only the Python
# dependencies.
#
# Pins: python 3.12 (the runtime needs >=3.11), torch cu126 (driver 575;
# README, "Image tag"), transformers 5.0.0 (the runtime's own `>=5.0,<5.1`
# bound and the combination its CPU reproduction verified), plus the HTTP stack.
#
# gcc + libc6-dev: the Triton kernel that compiles at the first CUDA request
# needs a C compiler (docs/operations.md, "Gotchas").
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.53.0 \
    transformers==5.0.0 \
    huggingface_hub \
    safetensors \
    numpy
COPY julia/server.py ./server.py
COPY py-common/stack_service.py ./stack_service.py
EXPOSE 80
# One worker per instance. A second replica is a separate service on the other
# card (`make up-julia CONCURRENCY=2`), each process loading its own copy.
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
