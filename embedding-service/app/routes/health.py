from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.model import engine
from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health and Model Readiness",
    description="Public endpoint to verify service health, loaded model metadata, and inference readiness.",
)
async def get_health():
    """
    Returns health status of the service.
    If the model failed to load, returns HTTP 503 Service Unavailable with ready=False.
    """
    if not engine.is_ready or engine.model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "model": engine.model_name,
                "dimensions": engine.dimensions,
                "ready": False,
                "device": engine.device,
            },
        )

    return HealthResponse(
        status="ok",
        model=engine.model_name,
        dimensions=engine.dimensions,
        ready=True,
        device=engine.device,
    )
