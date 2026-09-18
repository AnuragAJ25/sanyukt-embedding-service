from starlette.testclient import TestClient
from app.config import settings
from app.model import engine

EXPECTED_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def test_health_endpoint_public_and_healthy(client: TestClient):
    """Verify GET /health is publicly accessible without auth and reports ready when token and model are valid."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "intfloat/multilingual-e5-small"
    assert data["dimensions"] == 384
    assert data["ready"] is True
    assert data["revision"] == EXPECTED_REVISION
    assert "device" in data


def test_health_endpoint_degraded_when_model_unready(client: TestClient):
    """Verify GET /health returns 503 if model is unready."""
    original_state = engine.is_ready
    try:
        engine.is_ready = False
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert "degraded" in data["status"]
    finally:
        engine.is_ready = original_state


def test_health_endpoint_fails_closed_when_token_missing_or_blank(client: TestClient):
    """
    Verify GET /health returns 503 ready=false if EMBEDDING_API_TOKEN is missing or blank.
    Never reports ready=true when protected endpoints cannot be used.
    """
    original_token = settings.EMBEDDING_API_TOKEN
    try:
        settings.EMBEDDING_API_TOKEN = ""
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert "token_not_configured" in data["status"]

        # Also test whitespace-only token
        settings.EMBEDDING_API_TOKEN = "    "
        response_ws = client.get("/health")
        assert response_ws.status_code == 503
        data_ws = response_ws.json()
        assert data_ws["ready"] is False
    finally:
        settings.EMBEDDING_API_TOKEN = original_token
