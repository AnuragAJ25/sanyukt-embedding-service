# Production Dockerfile for Standalone Multilingual Embedding API
# Base Python image with minimal runtime overhead
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/cache/huggingface \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    EMBEDDING_MODEL_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install minimal system dependencies required for torch/tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install exact frozen dependencies from requirements.lock
COPY requirements.lock /app/
RUN pip install --no-cache-dir -r requirements.lock

# Pre-download and bake pinned model revision into container image during build (Strategy A)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small', revision='614241f622f53c4eeff9890bdc4f31cfecc418b3')"

# Copy application source code
COPY app /app/app

# Enforce strict offline operation at runtime: zero outbound internet calls to HuggingFace
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

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
