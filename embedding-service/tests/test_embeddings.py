import math
import pytest
from starlette.testclient import TestClient
from app.config import settings


def is_unit_normalized(vec: list[float], tolerance: float = 1e-3) -> bool:
    """Check if vector L2 norm is approximately 1.0."""
    norm = math.sqrt(sum(x * x for x in vec))
    return abs(norm - 1.0) < tolerance


def test_single_query_embedding_success(client: TestClient, auth_headers: dict):
    """Test generating single embedding for a query."""
    payload = {
        "text": "tractor subsidy",
        "input_type": "query",
    }
    response = client.post("/v1/embeddings", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    assert data["model"] == settings.EMBEDDING_MODEL
    assert data["dimensions"] == 384
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) == 384
    assert all(isinstance(val, float) for val in data["embedding"])
    assert is_unit_normalized(data["embedding"])


def test_single_passage_embedding_success(client: TestClient, auth_headers: dict):
    """Test generating single embedding for a passage."""
    payload = {
        "text": "Sub-Mission on Farm Mechanization provides financial assistance for procurement of tractors.",
        "input_type": "passage",
    }
    response = client.post("/v1/embeddings", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    assert data["model"] == settings.EMBEDDING_MODEL
    assert data["dimensions"] == 384
    assert len(data["embedding"]) == 384
    assert is_unit_normalized(data["embedding"])


def test_batch_embedding_success(client: TestClient, auth_headers: dict):
    """Test generating embeddings for a batch of texts."""
    texts = [
        "Age Nationality and Domicile Certificate",
        "Sub-Mission on Farm Mechanization",
        "Skill Development Training",
    ]
    payload = {
        "texts": texts,
        "input_type": "passage",
    }
    response = client.post("/v1/embeddings/batch", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    assert data["model"] == settings.EMBEDDING_MODEL
    assert data["dimensions"] == 384
    assert data["count"] == 3
    assert len(data["embeddings"]) == 3

    for emb in data["embeddings"]:
        assert len(emb) == 384
        assert is_unit_normalized(emb)


def test_reject_blank_or_whitespace_text(client: TestClient, auth_headers: dict):
    """Verify that empty string or whitespace-only texts are rejected with 422."""
    for empty_val in ["", "   ", "\t\n  "]:
        resp_single = client.post(
            "/v1/embeddings",
            json={"text": empty_val, "input_type": "query"},
            headers=auth_headers,
        )
        assert resp_single.status_code == 422

        resp_batch = client.post(
            "/v1/embeddings/batch",
            json={"texts": ["Valid text", empty_val], "input_type": "passage"},
            headers=auth_headers,
        )
        assert resp_batch.status_code == 422


def test_reject_invalid_input_type(client: TestClient, auth_headers: dict):
    """Verify that invalid input_type is rejected with 422."""
    payload = {
        "text": "Valid text here",
        "input_type": "invalid_type",
    }
    response = client.post("/v1/embeddings", json=payload, headers=auth_headers)
    assert response.status_code == 422


def test_reject_oversized_batch(client: TestClient, auth_headers: dict):
    """Verify that batch requests exceeding MAX_BATCH_SIZE are rejected."""
    limit = settings.MAX_BATCH_SIZE
    oversized_texts = [f"Sample passage number {i}" for i in range(limit + 5)]
    payload = {
        "texts": oversized_texts,
        "input_type": "passage",
    }
    response = client.post("/v1/embeddings/batch", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"].lower()


def test_multilingual_inputs_produce_valid_embeddings(client: TestClient, auth_headers: dict):
    """Verify English, Hindi, Marathi, and Hinglish all produce 384-dim normalized embeddings."""
    test_cases = [
        ("tractor subsidy", "English"),
        ("kisan ko tractor ke liye help", "Hinglish"),
        ("किसानों के लिए कृषि मशीन सहायता", "Hindi"),
        ("शेतकऱ्यांसाठी कृषी यंत्र मदत", "Marathi"),
    ]
    for text, lang in test_cases:
        response = client.post(
            "/v1/embeddings",
            json={"text": text, "input_type": "query"},
            headers=auth_headers,
        )
        assert response.status_code == 200, f"Failed for {lang}"
        data = response.json()
        assert len(data["embedding"]) == 384
        assert is_unit_normalized(data["embedding"])
