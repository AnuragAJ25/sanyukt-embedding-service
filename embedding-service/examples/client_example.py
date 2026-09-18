"""
Client Integration Example for Standalone Multilingual Embedding API.

This snippet demonstrates how an external consumer application can interface
with the embedding service to query and index embeddings.
"""

import os
import requests

# Set your microservice endpoint and token
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8000")
EMBEDDING_API_TOKEN = os.getenv("EMBEDDING_API_TOKEN", "your_api_token_here")

HEADERS = {
    "Authorization": f"Bearer {EMBEDDING_API_TOKEN}",
    "Content-Type": "application/json",
}


def get_query_embedding(query_text: str) -> list[float]:
    """
    Generate embedding for search queries or user questions.
    The service will automatically prepend 'query: '.
    """
    url = f"{EMBEDDING_SERVICE_URL}/v1/embeddings"
    payload = {
        "text": query_text,
        "input_type": "query",
    }
    response = requests.post(url, json=payload, headers=HEADERS, timeout=10.0)
    response.raise_for_status()
    data = response.json()
    return data["embedding"]


def get_passage_embeddings(passages: list[str]) -> list[list[float]]:
    """
    Generate embeddings for documents, scheme descriptions, or catalog items to index.
    The service will automatically prepend 'passage: '.
    """
    url = f"{EMBEDDING_SERVICE_URL}/v1/embeddings/batch"
    payload = {
        "texts": passages,
        "input_type": "passage",
    }
    response = requests.post(url, json=payload, headers=HEADERS, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    return data["embeddings"]


if __name__ == "__main__":
    print("Example client demonstration:")
    # Note: Make sure the embedding service is running with EMBEDDING_API_TOKEN configured.
    print("- To embed a search query: call get_query_embedding('tractor subsidy')")
    print("- To embed passages to index: call get_passage_embeddings(['Scheme text 1', 'Scheme text 2'])")
