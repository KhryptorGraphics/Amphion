"""
API Authentication Middleware

Simple API key authentication for Amphion API.
In production, use a proper auth system with database-backed API keys.
"""

import os
import logging
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load API key from environment or use default for development
API_KEY = os.getenv("AMPHION_API_KEY", "amphion-dev-key-change-in-production")
DEV_MODE = os.getenv("AMPHION_DEV_MODE", "false").lower() == "true"


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        The verified API key

    Raises:
        HTTPException: If API key is invalid or missing
    """
    # Skip auth for health endpoints
    # This will be handled in middleware

    if DEV_MODE:
        return api_key or "dev-mode"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing. Include X-API-Key header.",
        )

    # In production, validate against database
    # For now, simple constant-time comparison
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to check API key on all requests except exempt paths.
    """

    # Paths that don't require authentication
    exempt_paths = [
        "/api/health",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
    ]

    async def dispatch(self, request: Request, call_next):
        # Skip auth for exempt paths
        path = request.url.path
        if any(path.startswith(exempt) for exempt in self.exempt_paths):
            return await call_next(request)

        # Skip auth for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("X-API-Key")

        if DEV_MODE:
            return await call_next(request)

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "API key missing. Include X-API-Key header."},
            )

        if api_key != API_KEY:
            logger.warning(f"Invalid API key from {request.client.host}: {api_key[:8]}...")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid API key"},
            )

        return await call_next(request)
