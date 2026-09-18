import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials

from app.config import settings
from app.model import engine
from app.schemas import (
    SingleEmbeddingRequest,
    SingleEmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
)
from app.security import security_scheme, verify_api_token

logger = logging.getLogger("embedding-service.embeddings")

router = APIRouter(prefix="/v1/embeddings", tags=["Embeddings"])

_inference_semaphore: Optional[asyncio.Semaphore] = None


def get_inference_semaphore() -> asyncio.Semaphore:
    """Bounded concurrency guard to prevent CPU thrashing or memory overflow."""
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INFERENCE)
    return _inference_semaphore


@router.post(
    "",
    response_model=SingleEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Single Embedding",
    description=(
        "Generates a 384-dimensional unit-normalized embedding for a single text. "
        "The model automatically prepends 'query: ' or 'passage: ' based on the provided input_type."
    ),
    responses={
        200: {
            "description": "Normalized vector embedding generated successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "model": "intfloat/multilingual-e5-small",
                        "dimensions": 384,
                        "embedding": [0.0123, -0.0456, 0.0789],
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized. Missing or invalid Bearer token.",
            "content": {"application/json": {"example": {"detail": "Unauthorized"}}},
        },
        422: {"description": "Validation error (e.g. empty text, invalid input_type)."},
        500: {
            "description": "Internal embedding generation error.",
            "content": {"application/json": {"example": {"detail": "Embedding generation failed"}}},
        },
    },
)
async def create_single_embedding(
    request: SingleEmbeddingRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
):
    """
    Generate an embedding vector for a single query or passage.
    Requires Bearer token authentication.
    """
    verify_api_token(credentials)

    try:
        semaphore = get_inference_semaphore()
        async with semaphore:
            embeddings = await asyncio.to_thread(
                engine.embed_texts, [request.text], request.input_type
            )
        if not embeddings:
            raise RuntimeError("Engine returned no embeddings.")
        embedding = embeddings[0]
    except Exception as exc:
        logger.error(f"Failed to generate embedding: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed",
        )

    return SingleEmbeddingResponse(
        model=engine.model_name,
        revision=engine.model_revision,
        dimensions=len(embedding),
        embedding=embedding,
    )


@router.post(
    "/batch",
    response_model=BatchEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Batch Embeddings",
    description=(
        "Generates 384-dimensional unit-normalized embeddings for a batch of texts. "
        "Automatically prefixes each item with 'query: ' or 'passage: '. "
        "The maximum batch size is configurable (default 64)."
    ),
    responses={
        200: {
            "description": "Batch of normalized vector embeddings generated successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "model": "intfloat/multilingual-e5-small",
                        "dimensions": 384,
                        "count": 3,
                        "embeddings": [
                            [0.01, -0.02, 0.03],
                            [0.04, -0.05, 0.06],
                            [0.07, -0.08, 0.09],
                        ],
                    }
                }
            },
        },
        400: {
            "description": "Oversized batch or malformed batch payload.",
            "content": {"application/json": {"example": {"detail": "Batch size exceeds maximum limit"}}},
        },
        401: {
            "description": "Unauthorized. Missing or invalid Bearer token.",
            "content": {"application/json": {"example": {"detail": "Unauthorized"}}},
        },
        422: {"description": "Validation error in batch items or input_type."},
        500: {
            "description": "Internal embedding generation error.",
            "content": {"application/json": {"example": {"detail": "Embedding generation failed"}}},
        },
    },
)
async def create_batch_embeddings(
    request: BatchEmbeddingRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
):
    """
    Generate embedding vectors for a list of queries or passages.
    Enforces maximum batch size. Requires Bearer token authentication.
    """
    verify_api_token(credentials)

    if len(request.texts) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size {len(request.texts)} exceeds configured maximum allowed "
                f"batch size of {settings.MAX_BATCH_SIZE}"
            ),
        )

    try:
        semaphore = get_inference_semaphore()
        async with semaphore:
            embeddings = await asyncio.to_thread(
                engine.embed_texts, request.texts, request.input_type
            )
    except Exception as exc:
        logger.error(f"Failed to generate batch embeddings: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed",
        )

    return BatchEmbeddingResponse(
        model=engine.model_name,
        revision=engine.model_revision,
        dimensions=engine.dimensions,
        count=len(embeddings),
        embeddings=embeddings,
    )
