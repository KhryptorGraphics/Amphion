"""
Rate Limiting Middleware

Simple in-memory rate limiting for Amphion API.
Uses token bucket algorithm per client IP.
"""

import time
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from fastapi import HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""
    tokens: float
    last_update: float
    requests: int  # Track request count for burst detection


class RateLimiter:
    """
    In-memory rate limiter using token bucket algorithm.

    For production, use Redis-backed rate limiting.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        cleanup_interval: int = 300  # 5 minutes
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.cleanup_interval = cleanup_interval
        self.buckets: Dict[str, RateLimitBucket] = {}
        self.last_cleanup = time.time()

    def _get_bucket(self, key: str) -> RateLimitBucket:
        """Get or create bucket for a client."""
        now = time.time()

        if key not in self.buckets:
            return RateLimitBucket(
                tokens=self.burst_size,
                last_update=now,
                requests=0
            )

        bucket = self.buckets[key]

        # Add tokens based on time elapsed
        elapsed = now - bucket.last_update
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        bucket.tokens = min(self.burst_size, bucket.tokens + tokens_to_add)
        bucket.last_update = now

        return bucket

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if request is allowed.

        Returns:
            (allowed, retry_after_seconds)
        """
        now = time.time()

        # Periodic cleanup of old entries
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_buckets()

        bucket = self._get_bucket(key)

        if bucket.tokens >= 1:
            bucket.tokens -= 1
            bucket.requests += 1
            self.buckets[key] = bucket
            return True, 0
        else:
            # Calculate retry after
            retry_after = int((1 - bucket.tokens) * 60 / self.requests_per_minute) + 1
            return False, retry_after

    def _cleanup_old_buckets(self):
        """Remove old bucket entries."""
        now = time.time()
        old_keys = [
            key for key, bucket in self.buckets.items()
            if now - bucket.last_update > self.cleanup_interval
        ]
        for key in old_keys:
            del self.buckets[key]
        self.last_cleanup = now
        if old_keys:
            logger.debug(f"Cleaned up {len(old_keys)} rate limit buckets")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to apply rate limiting.

    Default: 60 requests per minute with burst of 10.
    """

    # Paths with different rate limits
    # Inference endpoints are more expensive
    strict_paths = ["/api/tts/", "/api/vc/", "/api/svc/"]
    strict_limiter = RateLimiter(requests_per_minute=10, burst_size=3)

    # Standard endpoints
    standard_limiter = RateLimiter(requests_per_minute=60, burst_size=10)

    # Exempt paths (health, docs)
    exempt_paths = ["/api/health", "/api/docs", "/api/redoc", "/api/openapi.json"]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip rate limiting for exempt paths
        if any(path.startswith(exempt) for exempt in self.exempt_paths):
            return await call_next(request)

        # Skip rate limiting for OPTIONS requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Get client identifier (IP + user agent hash for simplicity)
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        # Choose limiter based on path
        if any(path.startswith(p) for p in self.strict_paths):
            limiter = self.strict_limiter
        else:
            limiter = self.standard_limiter

        # Check rate limit
        allowed, retry_after = limiter.is_allowed(client_ip)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_ip} on {path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": f"Rate limit exceeded. Retry after {retry_after} seconds."},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)


def get_rate_limit_headers(requests: int, window: int) -> dict:
    """Generate rate limit headers for responses."""
    return {
        "X-RateLimit-Limit": str(requests),
        "X-RateLimit-Window": str(window),
    }
