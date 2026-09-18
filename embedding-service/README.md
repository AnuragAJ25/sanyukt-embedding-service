# Standalone Multilingual Embedding API Microservice

A production-ready, lightweight, standalone FastAPI microservice that generates **384-dimensional unit-normalized vector embeddings** using HuggingFace's [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) and `sentence-transformers`.

---

## 1. Purpose & Responsibilities
This microservice serves a single, focused responsibility: **Transforming raw text into normalized multilingual vector embeddings**.

- **Primary Responsibility:**
  $$\text{Text} \longrightarrow \text{multilingual-e5-small} \longrightarrow \text{384-dimensional Normalized Embedding}$$
- **Zero Extraneous Complexity:**
  - No database (no PostgreSQL, Supabase, Redis, or SQLite).
  - No vector database or vector search.
  - No RAG pipelines or retrieval logic.
  - No chatbots, LLMs, or business logic.
  - Completely independent and decoupled from any consumer application.

---

## 2. Architecture & Design Principles

```
                              ┌────────────────────────────────────────┐
                              │           FastAPI Microservice         │
                              │                                        │
  HTTP Request                │  ┌──────────────────────────────────┐  │
 ──────────────> [Bearer Auth] │  │  FastAPI Lifespan Startup Hook   │  │
 (POST /v1/..)   [Validation] │  │  (Loads Model Once into Memory)  │  │
                              │  └──────────────────┬───────────────┘  │
                              │                     │                  │
                              │  ┌──────────────────▼───────────────┐  │
                              │  │        Embedding Engine          │  │
                              │  │  • Automatic Prefix Handler      │  │
                              │  │    (query: / passage:)           │  │
                              │  │  • L2 Normalization              │  │
                              │  │  • CPU Optimized (1 Worker)      │  │
                              │  └──────────────────┬───────────────┘  │
                              │                     │                  │
  HTTP Response <─────────────┴─────────────────────┘                  │
 (384-d floats)               └────────────────────────────────────────┘
```

- **Singleton Model Lifecycle:** The model is loaded exactly **once** at application startup inside the FastAPI lifespan context manager. It is never reloaded per request.
- **Automatic E5 Prefixing:** Callers supply `input_type: "query"` or `input_type: "passage"`. The service automatically prefixes `"query: <text>"` or `"passage: <text>"` as required by `multilingual-e5-small`.
- **Unit Normalization:** Embeddings are generated with $L_2$ unit normalization ($\|\mathbf{v}\|_2 = 1.0$). Dot products between normalized embeddings are mathematically identical to cosine similarity.
- **CPU First:** Operates reliably on CPU hardware without requiring CUDA. Detects and supports CUDA automatically when available.
- **Single Process / Safe Concurrency:** Configured with a single Uvicorn worker by default to avoid redundant multi-gigabyte memory copies of PyTorch models across worker processes.

---

## 3. Multilingual Support
Supported languages out of the box:
- **English** (e.g., `"tractor subsidy"`)
- **Hindi** (e.g., `"किसानों के लिए कृषि मशीन सहायता"`)
- **Marathi** (e.g., `"शेतकऱ्यांसाठी कृषी यंत्र मदत"`)
- **Hinglish / Mixed Dialects** (e.g., `"kisan ko tractor ke liye help"`)

---

## 4. Requirements & Prerequisites
- **Python:** 3.10, 3.11, 3.12, or 3.13
- **Memory:** Minimum 1.5 GB RAM (2 GB recommended for CPU inference)
- **Disk:** ~500 MB for PyTorch and cached model weights

---

## 5. Local Installation & Setup

### Clone or Navigate to Project
```bash
cd embedding-service
```

### Create Virtual Environment
Using `uv` (recommended for ultra-fast setup):
```bash
uv venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

Or using standard `venv`:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
# Or with uv:
uv pip install -r requirements.txt
```

### Configure Environment Variables
Copy the example environment configuration:
```bash
cp .env.example .env
```

Generate a cryptographically secure token:
```bash
openssl rand -hex 32
```
Set `EMBEDDING_API_TOKEN` in your `.env` file to this token.

---

## 6. Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | HuggingFace model identifier |
| `EMBEDDING_MODEL_REVISION` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | Pinned HuggingFace model commit revision hash for reproducibility |
| `EMBEDDING_API_TOKEN` | *(None / Required)* | Secret Bearer token required for `/v1/embeddings*` |
| `MAX_BATCH_SIZE` | `64` | Maximum allowable text items in a single batch request |
| `MAX_CONCURRENT_INFERENCE` | `1` | Maximum concurrent inference operations (CPU default: 1) |
| `PORT` | `8000` | Port for the Uvicorn web server |
| `HOST` | `0.0.0.0` | Host interface to bind |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DEVICE` | `cpu` *(auto)* | `cpu` or `cuda`. If unspecified, auto-detects CUDA |
| `ALLOWED_ORIGINS` | `[]` | CORS origins list. Kept empty for backend microservices |

---

## 7. Running Locally

### Development / Direct Execution
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Or execute directly with Python:
```bash
python -m app.main
```

The interactive OpenAPI / Swagger documentation will be live at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 8. Docker Build & Run

### Model Caching Strategy (Strategy A: Pre-Baked Cache)
The provided `Dockerfile` follows **Strategy A**: It downloads and bakes the model weights into `/app/cache/huggingface` during the Docker image build stage.
- **Benefit:** Zero cold-start latency when the container launches.
- **Zero Runtime Internet Dependency:** The container can start in air-gapped or restricted VPC environments without needing access to HuggingFace at runtime.

### Build Container
```bash
docker build -t multilingual-e5-api .
```

### Run Container
```bash
docker run -d \
  --name multilingual-e5-service \
  -e EMBEDDING_API_TOKEN="your_secure_generated_token" \
  -p 8000:8000 \
  --restart unless-stopped \
  multilingual-e5-api
```

### Verify Container Health
```bash
curl http://localhost:8000/health
```

---

## 9. API Reference & Endpoints

### 1. Health & Readiness Check
- **Endpoint:** `GET /health`
- **Authentication:** None (Public)
- **Status Codes:**
  - `200 OK`: Model is loaded and ready for inference.
  - `503 Service Unavailable`: Model failed to load or is not ready.

#### Example Response:
```json
{
  "status": "ok",
  "model": "intfloat/multilingual-e5-small",
  "dimensions": 384,
  "ready": true,
  "device": "cpu"
}
```

---

### 2. Single Text Embedding
- **Endpoint:** `POST /v1/embeddings`
- **Authentication:** `Authorization: Bearer <EMBEDDING_API_TOKEN>`
- **Headers:** `Content-Type: application/json`

#### Request Payload:
```json
{
  "text": "tractor subsidy",
  "input_type": "query"
}
```

#### Response (200 OK):
```json
{
  "model": "intfloat/multilingual-e5-small",
  "dimensions": 384,
  "embedding": [
    0.01524,
    -0.03819,
    0.07211,
    ...
  ]
}
```

---

### 3. Batch Text Embeddings
- **Endpoint:** `POST /v1/embeddings/batch`
- **Authentication:** `Authorization: Bearer <EMBEDDING_API_TOKEN>`
- **Headers:** `Content-Type: application/json`

#### Request Payload:
```json
{
  "texts": [
    "Sub-Mission on Farm Mechanization",
    "Age Nationality and Domicile Certificate",
    "Skill Development Training"
  ],
  "input_type": "passage"
}
```

#### Response (200 OK):
```json
{
  "model": "intfloat/multilingual-e5-small",
  "dimensions": 384,
  "count": 3,
  "embeddings": [
    [0.0124, -0.0481, ...],
    [0.0319, -0.0128, ...],
    [-0.0092, 0.0614, ...]
  ]
}
```

---

## 10. Working cURL Examples

Set your environment variables:
```bash
export EMBEDDING_SERVICE_URL="http://localhost:8000"
export EMBEDDING_SERVICE_TOKEN="your_configured_token"
```

### Single Query Embedding
```bash
curl -s -X POST "$EMBEDDING_SERVICE_URL/v1/embeddings" \
  -H "Authorization: Bearer $EMBEDDING_SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "tractor subsidy",
    "input_type": "query"
  }'
```

### Batch Passage Embeddings
```bash
curl -s -X POST "$EMBEDDING_SERVICE_URL/v1/embeddings/batch" \
  -H "Authorization: Bearer $EMBEDDING_SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Sub-Mission on Farm Mechanization",
      "Age Nationality and Domicile Certificate",
      "Skill Development Training"
    ],
    "input_type": "passage"
  }'
```

---

## 11. Security & Privacy

### Security
- **Constant-Time Verification:** Uses Python's `secrets.compare_digest` to validate tokens, preventing timing side-channel attacks.
- **Never Committed / Never Logged:** Secret tokens are never hardcoded, never logged, and never included in error responses.
- **Least-Privilege Docker:** Runs as a non-root system user (`appuser` UID 1000).

### Privacy & Citizen Data Protection
- **Zero Text Logging:** The microservice **never logs raw input text** or queries.
- **Safe Operational Metadata Only:** Request logs only record safe operational metrics:
  ```
  method=POST path=/v1/embeddings status=200 duration_ms=28.4 client=127.0.0.1
  ```
- **Stateless Operation:** No database, no local caching of input content, and no external tracking.

---

## 12. Deployment Guidance

### Recommended Production Platforms
1. **Container Platforms (Recommended):**
   - AWS ECS / Fargate (1 vCPU, 2 GB RAM)
   - Google Cloud Run (1 CPU, 2 GB RAM, minimum instances = 1 for instant response)
   - DigitalOcean App Platform (Basic/Pro container with 2 GB RAM)
   - Kubernetes cluster / Nomad
2. **Reverse Proxy & TLS Termination:**
   - Terminate SSL/TLS with Nginx, Traefik, AWS ALB, or Cloudflare.
   - Always run the microservice behind an HTTPS domain in production (`https://api.yourdomain.com/v1/embeddings`).

> [!WARNING]
> **Google Colab:** Do **not** use Google Colab as a production service. Colab is suitable only for experimentation, temporary benchmarking, or model validation, as it lacks stable ingress URLs, persistent SLA guarantees, and enterprise security.

---

## 13. Integration Contract for Consumer Backends
When calling this microservice from your application backend:
1. **Input Type Rule:**
   - Use `"input_type": "query"` for search input, user prompts, or retrieval queries.
   - Use `"input_type": "passage"` for document chunks, scheme descriptions, or catalog items to be indexed into a vector store.
2. **Do Not Prepend Prefix:** Never prepend `"query: "` or `"passage: "` manually; the microservice handles this automatically.
3. **Dimensions:** Expect an array of exactly `384` floats.
4. **Vector Store Compatibility:** Since all embeddings are unit-normalized ($L_2$ norm = 1.0), you can configure your vector database (e.g. pgvector, Pinecone, Qdrant) with either **Cosine Similarity** or **Inner Product (dot product)** for maximum retrieval performance.
