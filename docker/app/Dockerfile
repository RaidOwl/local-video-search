FROM python:3.11-slim

WORKDIR /app

# Install FFmpeg + hardware acceleration libraries.
# mesa-va-drivers enables VAAPI for AMD/Intel GPUs; libvulkan1 enables Vulkan decode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    mesa-va-drivers \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies. The reranker runs on CPU, so torch
# comes from the CPU-only index (amd64 + arm64) instead of the CUDA build.
ARG TORCH_VERSION=2.12.1
COPY requirements.txt requirements-reranker.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && pip install --no-cache-dir -r requirements.txt -r requirements-reranker.txt

# Copy application code
COPY src/ ./src/

# Create data directory
RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]