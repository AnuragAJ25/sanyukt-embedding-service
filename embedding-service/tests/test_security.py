from starlette.testclient import TestClient
from app.config import settings


def test_missing_auth_header_returns_401(client: TestClient):
    """Calling protected endpoints without Authorization header returns 401."""
    response = client.post("/v1/embeddings", json={"text": "test", "input_type": "query"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}

    batch_response = client.post(
        "/v1/embeddings/batch",
        json={"texts": ["test text"], "input_type": "passage"},
    )
    assert batch_response.status_code == 401
    assert batch_response.json() == {"detail": "Unauthorized"}


def test_invalid_bearer_token_returns_401(client: TestClient):
    """Calling protected endpoints with incorrect Bearer token returns 401."""
    headers = {"Authorization": "Bearer wrong-token-value"}
    response = client.post(
        "/v1/embeddings",
        json={"text": "test", "input_type": "query"},
        headers=headers,
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_malformed_auth_header_returns_401(client: TestClient):
    """Calling protected endpoints with non-Bearer scheme returns 401."""
    headers = {"Authorization": "Basic dXNlcjpwYXNz"}
    response = client.post(
        "/v1/embeddings",
        json={"text": "test", "input_type": "query"},
        headers=headers,
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_token_never_leaks_in_response(client: TestClient, auth_headers: dict):
    """Verify that secret API token is never reflected in response body or headers."""
    response = client.post(
        "/v1/embeddings",
        json={"text": "tractor subsidy", "input_type": "query"},
        headers=auth_headers,
    )
    assert response.status_code == 200

    raw_response_text = response.text
    assert settings.EMBEDDING_API_TOKEN not in raw_response_text

    for header_name, header_value in response.headers.items():
        assert settings.EMBEDDING_API_TOKEN not in header_value
