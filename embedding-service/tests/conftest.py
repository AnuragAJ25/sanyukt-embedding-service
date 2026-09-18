import os
import pytest
from starlette.testclient import TestClient

# Ensure test token is set before importing settings or app
os.environ["EMBEDDING_API_TOKEN"] = "test-secret-token-abcdef123456"

from app.config import settings
from app.main import app

settings.EMBEDDING_API_TOKEN = "test-secret-token-abcdef123456"
settings.MAX_BATCH_SIZE = 64


@pytest.fixture(scope="session")
def auth_headers():
    """Valid Bearer authorization header for test requests."""
    return {"Authorization": f"Bearer {settings.EMBEDDING_API_TOKEN}"}


@pytest.fixture(scope="session")
def client():
    """
    Session-scoped TestClient that runs app lifespan once,
    loading the model only once across the entire test run.
    """
    with TestClient(app) as test_client:
        yield test_client
