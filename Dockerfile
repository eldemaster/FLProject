# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Minimal system libs (runtime only) + build tools for any wheel that needs it
RUN apt-get update && apt-get install -y --no-install-recommends \
      libopenblas0 libgomp1 build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2

WORKDIR /app

# --- install PyTorch first from the CPU wheel index (ARM64 wheels available) ---
# Pin versions that are known to exist for cp311 manylinux aarch64
RUN python -m pip install --upgrade pip wheel && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.7.0 torchvision==0.22.0

# --- then the rest of your deps from PyPI ---
COPY requirements.txt /app/requirements.txt
# Ensure requirements.txt does NOT list torch/torchvision again
RUN sed -i '/^torch\|^torchvision/d' /app/requirements.txt && \
    pip install -r /app/requirements.txt

# --- your code ---
COPY client_main.py models.py dataset.py fl_client.py utils.py /app/

ENTRYPOINT ["python", "/app/client_main.py"]