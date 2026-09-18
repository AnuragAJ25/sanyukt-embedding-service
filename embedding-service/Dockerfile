# Production Dockerfile for Standalone Multilingual Embedding API
# Base Python image with minimal runtime overhead
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install minimal system dependencies required for torch/tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache model during container build (Strategy A)
# Ensures zero cold-start delay and no runtime internet dependency for weights
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Copy application source code
COPY app /app/app

# Create non-root system user for security
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose microservice HTTP port
EXPOSE 8000

# Health check using the public /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI via Uvicorn with single worker (safeguards CPU memory usage)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
