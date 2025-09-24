# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System libs for NumPy/SciPy/Torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Global pip config: Torch from PyTorch CPU index, everything else from PyPI
ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

WORKDIR /app

# Install deps first (better cache)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip wheel && \
    pip install -r /app/requirements.txt

# Copy code
COPY client_main.py models.py dataset.py fl_client.py utils.py /app/

ENTRYPOINT ["python", "/app/client_main.py"]