from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.model import engine
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health and Model Readiness",
    description="Public endpoint to verify service health, loaded model metadata, and inference readiness.",
    responses={
        200: {"description": "Service is healthy, model is loaded, and API token is configured."},
        503: {"description": "Service is unavailable or unready (model not loaded or token missing)."},
    },
)
async def get_health():
    """
    Returns health status of the service.
    If the model failed to load OR if EMBEDDING_API_TOKEN is missing/blank,
    returns HTTP 503 Service Unavailable with ready=False.
    Never reports ready=True when protected endpoints cannot be used.
    """
    token_ready = settings.is_token_configured()
    model_ready = engine.is_ready and engine.model is not None

    if not model_ready or not token_ready:
        reason = "model_not_ready" if not model_ready else "token_not_configured"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": f"degraded ({reason})",
                "model": engine.model_name,
                "revision": engine.model_revision,
                "dimensions": engine.dimensions,
                "ready": False,
                "device": engine.device,
            },
        )

    return HealthResponse(
        status="ok",
        model=engine.model_name,
        revision=engine.model_revision,
        dimensions=engine.dimensions,
        ready=True,
        device=engine.device,
    )
