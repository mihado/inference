# Laya decision model: the stack's third runtime (Python, not TEI/vLLM).
#
# The pins are the combination the smoke test verified. torch comes from the
# cu126 index, not PyPI: PyPI's torch 2.14 wheel ships CUDA 13, which needs
# driver >= 580, while this box runs 575+ (README, "Image tag"). The +cu126
# build runs on the box's driver and on any newer one, and falls back to CPU on
# a CPU-only host — so the same image serves the box and a dev smoke test.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    laya==0.3.5 \
    fastapi==0.141.1 \
    uvicorn==0.53.0 \
    transformers==5.17.0
COPY server.py .
EXPOSE 80
# One worker: each worker process would load its own copy of the checkpoint.
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
