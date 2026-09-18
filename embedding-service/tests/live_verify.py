import os
import sys
import math
import threading
import time
import httpx
import uvicorn

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output for multilingual text on Windows (cp1252 cannot encode Hindi/Marathi)
sys.stdout.reconfigure(encoding="utf-8")

# Set test token before importing application settings
os.environ["EMBEDDING_API_TOKEN"] = "test-live-token-9876543210fedcba"

from app.main import app

EXPECTED_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

def is_unit_normalized(vec: list, tolerance: float = 1e-3) -> bool:
    norm = math.sqrt(sum(x * x for x in vec))
    return abs(norm - 1.0) < tolerance

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

base_url = "http://127.0.0.1:8008"
token = "test-live-token-9876543210fedcba"
headers = {"Authorization": f"Bearer {token}"}

# Poll /health until server responds and ready is True
max_retries = 30
server_ready = False

with httpx.Client(base_url=base_url, timeout=30.0) as client:
    for attempt in range(max_retries):
        try:
            h = client.get("/health")
            if h.status_code == 200 and h.json().get("ready") is True:
                print(f"Server ready after {attempt + 1} polls! Health response: {h.json()}")
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(1)

    if not server_ready:
        print("ERROR: Server did not become ready in time.")
        sys.exit(1)

    # 1. Health check verification
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["ready"] is True
    assert h.json()["revision"] == EXPECTED_REVISION
    print("[PASS] GET /health")

    # 2. Unauthorized check
    u = client.post("/v1/embeddings", json={"text": "tractor", "input_type": "query"})
    assert u.status_code == 401
    assert u.json() == {"detail": "Unauthorized"}
    print("[PASS] Unauthorized request rejected with 401")

    # 3. Single Embedding query
    s = client.post(
        "/v1/embeddings",
        json={"text": "tractor subsidy", "input_type": "query"},
        headers=headers,
    )
    assert s.status_code == 200
    s_data = s.json()
    assert s_data["model"] == "intfloat/multilingual-e5-small"
    assert s_data["revision"] == EXPECTED_REVISION
    assert s_data["dimensions"] == 384
    assert len(s_data["embedding"]) == 384
    assert is_unit_normalized(s_data["embedding"])
    print(f"[PASS] POST /v1/embeddings (Single query, dim: {len(s_data['embedding'])}, normalized: True)")

    # 4. Batch Embedding passage
    passages = [
        "Sub-Mission on Farm Mechanization",
        "Age, Nationality and Domicile Certificate",
        "Skill Development Training",
    ]
    b = client.post(
        "/v1/embeddings/batch",
        json={"texts": passages, "input_type": "passage"},
        headers=headers,
    )
    assert b.status_code == 200
    b_data = b.json()
    assert b_data["count"] == 3
    assert b_data.get("revision") == EXPECTED_REVISION
    assert len(b_data["embeddings"]) == 3
    assert all(len(v) == 384 and is_unit_normalized(v) for v in b_data["embeddings"])
    print(f"[PASS] POST /v1/embeddings/batch (Count: {b_data['count']}, Dims: 384, normalized: True)")

    # 5. Multilingual verification across 4 query types
    multilingual = [
        ("tractor subsidy", "English"),
        ("kisan ko tractor ke liye help", "Hinglish"),
        ("किसानों के लिए कृषि मशीन सहायता", "Hindi"),
        ("शेतकऱ्यांसाठी कृषी यंत्र मदत", "Marathi"),
    ]
    for text, lang in multilingual:
        res = client.post(
            "/v1/embeddings",
            json={"text": text, "input_type": "query"},
            headers=headers,
        )
        assert res.status_code == 200
        emb = res.json()["embedding"]
        assert len(emb) == 384
        assert is_unit_normalized(emb)
        print(f"[PASS] Multilingual [{lang}]: text='{text}' -> 384 dims, normalized=True")

print("\n=======================================================")
print("ALL LIVE HTTP MICROSERVICE CHECKS PASSED SUCCESSFULLY!")
print("=======================================================")
