# Standalone Multilingual Embedding API Microservice

This repository hosts the **Standalone Multilingual Embedding API Microservice**, an isolated, high-performance service designed to generate 384-dimensional normalized vector embeddings using `intfloat/multilingual-e5-small`.

The service is completely self-contained in the [`embedding-service`](file:///c:/AntiGravity_WorkSpaces/Standalone%20Embedding%20API%20Microservices/embedding-service) directory.

## Quick Links
- **Project Folder:** [`embedding-service/`](file:///c:/AntiGravity_WorkSpaces/Standalone%20Embedding%20API%20Microservices/embedding-service)
- **Application Code:** [`embedding-service/app/`](file:///c:/AntiGravity_WorkSpaces/Standalone%20Embedding%20API%20Microservices/embedding-service/app)
- **Test Suite:** [`embedding-service/tests/`](file:///c:/AntiGravity_WorkSpaces/Standalone%20Embedding%20API%20Microservices/embedding-service/tests)
- **Dockerfile:** [`embedding-service/Dockerfile`](file:///c:/AntiGravity_WorkSpaces/Standalone%20Embedding%20API%20Microservices/embedding-service/Dockerfile)
- **Comprehensive Documentation:** [embedding-service/README.md](file:///c:/AntiGravity_WorkSpaces/Standalone%20Embedding%20API%20Microservices/embedding-service/README.md)

## Quick Start
```bash
cd embedding-service
uv venv .venv
.venv\Scripts\activate      # On Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```
