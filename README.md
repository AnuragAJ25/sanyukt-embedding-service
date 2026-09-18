# Standalone Multilingual Embedding API Microservice

This repository hosts the **Standalone Multilingual Embedding API Microservice**, an isolated, high-performance service designed to generate 384-dimensional normalized vector embeddings using `intfloat/multilingual-e5-small`.

The service is completely self-contained in the [`embedding-service/`](embedding-service/) directory.

## Quick Links
- **Project Folder:** [`embedding-service/`](embedding-service/)
- **Application Code:** [`embedding-service/app/`](embedding-service/app/)
- **Test Suite:** [`embedding-service/tests/`](embedding-service/tests/)
- **Dockerfile:** [`embedding-service/Dockerfile`](embedding-service/Dockerfile)
- **Comprehensive Documentation:** [embedding-service/README.md](embedding-service/README.md)

## Quick Start
```bash
cd embedding-service
uv venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```
