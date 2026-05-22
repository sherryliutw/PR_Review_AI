# ---------------------------------------------------------------------------
# PR Review AI — Dockerfile
# Uses NVIDIA CUDA base for bitsandbytes 4-bit quantization (GPU inference)
# ---------------------------------------------------------------------------

# ---------- Stage 1: Install Python dependencies ----------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create virtualenv so we can copy it cleanly to the runtime stage
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---------- Stage 2: Runtime image ----------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application code
COPY training/ ./training/
COPY api/ ./api/

# LoRA adapter weights (~20-30 MB) — mount or bake in at build time
# Default path matches api/config.py default; override via ADAPTER_PATH env var.
# To bake in:  COPY code-review-lora-adapter/ ./code-review-lora-adapter/
# To mount:    -v /host/path/adapter:/app/code-review-lora-adapter

EXPOSE 8000

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
