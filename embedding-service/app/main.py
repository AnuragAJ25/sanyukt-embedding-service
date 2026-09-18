import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.config import settings
from app.model import engine
from app.routes.health import router as health_router
from app.routes.embeddings import router as embeddings_router
from app.security import SafeAccessLoggingMiddleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("embedding-service.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads the multilingual-e5-small model ONCE at startup and verifies readiness.
    Cleans up resources upon shutdown.
    """
    logger.info("Starting up Standalone Multilingual Embedding API...")
    logger.info(f"Configuration: model={settings.EMBEDDING_MODEL}, max_batch={settings.MAX_BATCH_SIZE}")

    try:
        engine.load()
        if not settings.is_token_configured():
            logger.warning(
                "CRITICAL SECURITY NOTICE: EMBEDDING_API_TOKEN is missing or blank! "
                "The service will fail readiness checks (/health -> 503 ready=false) "
                "until a valid non-empty EMBEDDING_API_TOKEN is provided."
            )
        else:
            logger.info("Model initialization complete. Microservice is ready to accept requests.")
    except Exception as exc:
        logger.critical(f"Fatal error during model initialization: {exc}")
        # Allow startup to complete so health check can accurately report unready status or exit cleanly
        engine.is_ready = False

    yield

    logger.info("Shutting down Standalone Multilingual Embedding API...")


app = FastAPI(
    title="Standalone Multilingual Embedding API",
    description=(
        "Production-ready standalone embedding microservice powered by "
        "`intfloat/multilingual-e5-small`. Pre-trained for multilingual semantic retrieval "
        "(English, Hindi, Marathi, Hinglish) with automatic query/passage prefix handling "
        "and unit-normalized 384-dimensional output embeddings."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach privacy-compliant access logging middleware
app.add_middleware(SafeAccessLoggingMiddleware)

# Attach CORS only if specific non-wildcard origins are configured
if settings.ALLOWED_ORIGINS:
    # Protect against accidental wildcard configuration
    safe_origins = [o for o in settings.ALLOWED_ORIGINS if o != "*"]
    if safe_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=safe_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

# Register route controllers
app.include_router(health_router)
app.include_router(embeddings_router)


def custom_openapi():
    """Enhance OpenAPI schema with BearerAuth security definitions."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": "Provide secret API token in 'Authorization: Bearer <EMBEDDING_API_TOKEN>' header.",
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,  # Safe CPU worker count to prevent multiple heavy model duplicates in memory
        log_level=settings.LOG_LEVEL.lower(),
    )
