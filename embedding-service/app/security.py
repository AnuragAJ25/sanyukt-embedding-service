import secrets
import logging
import time
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.config import settings

logger = logging.getLogger("embedding-service.access")

security_scheme = HTTPBearer(
    auto_error=False,
    description="Bearer token authentication for embedding endpoints",
)


def verify_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = None,
) -> bool:
    """
    Validates the bearer token against the configured EMBEDDING_API_TOKEN.
    Uses constant-time comparison to prevent timing side-channel attacks.
    """
    # If no token is configured in environment, reject for safety
    configured_token = settings.EMBEDDING_API_TOKEN.strip()
    if not configured_token:
        logger.warning("EMBEDDING_API_TOKEN is not configured in settings. Rejecting request.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_token = credentials.credentials.strip()

    # Constant-time comparison
    is_valid = secrets.compare_digest(provided_token, configured_token)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


class SafeAccessLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs safe operational metadata without logging raw input text or tokens.
    Logs: method, path, status_code, duration_ms, client_host.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        response: Optional[Response] = None
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            client_host = request.client.host if request.client else "unknown"

            # Privacy rule: NEVER log request bodies, query text, or Authorization headers
            logger.info(
                f"method={request.method} path={request.url.path} status={status_code} "
                f"duration_ms={duration_ms} client={client_host}"
            )
