from starlette.testclient import TestClient
from app.model import engine


def test_health_endpoint_public_and_healthy(client: TestClient):
    """Verify GET /health is publicly accessible without auth and reports ready."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "intfloat/multilingual-e5-small"
    assert data["dimensions"] == 384
    assert data["ready"] is True
    assert "device" in data


def test_health_endpoint_degraded_when_unready(client: TestClient):
    """Verify GET /health returns 503 if model is unready."""
    original_state = engine.is_ready
    try:
        engine.is_ready = False
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert data["status"] == "degraded"
    finally:
        engine.is_ready = original_state
