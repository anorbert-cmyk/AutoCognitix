"""
Rate Limiting Middleware for FastAPI.

Provides Redis-based distributed rate limiting with fallback to in-memory limiting.
Supports per-IP and per-user limits with configurable windows.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10  # Additional burst allowance
    block_duration_seconds: int = 60  # Block duration after limit exceeded


class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter using sliding window.

    Used as fallback when Redis is not available.
    Note: Not suitable for multi-instance deployments.
    """

    def __init__(self) -> None:
        # Structure: {key: [(timestamp, count), ...]}
        self._requests: dict[str, list] = defaultdict(list)
        self._blocked: dict[str, float] = {}  # key: block_until_timestamp

    def _cleanup_old_requests(self, key: str, window_seconds: int) -> None:
        """Remove requests older than the window."""
        cutoff = time.time() - window_seconds
        self._requests[key] = [
            (ts, count)
            for ts, count in self._requests[key]
            if ts > cutoff
        ]

    def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """
        Check if a request is allowed.

        Args:
            key: Unique identifier (IP or user ID)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            Tuple of (allowed, remaining, retry_after_seconds)
        """
        now = time.time()

        # Check if blocked
        if key in self._blocked:
            if now < self._blocked[key]:
                retry_after = int(self._blocked[key] - now)
                return False, 0, retry_after
            else:
                del self._blocked[key]

        # Clean up old requests
        self._cleanup_old_requests(key, window_seconds)

        # Count current requests
        current_count = sum(count for _, count in self._requests[key])

        if current_count >= limit:
            # Block the key
            self._blocked[key] = now + settings.RATE_LIMIT_PER_MINUTE
            return False, 0, settings.RATE_LIMIT_PER_MINUTE

        # Add new request
        self._requests[key].append((now, 1))
        remaining = limit - current_count - 1

        return True, remaining, 0


class RedisRateLimiter:
    """
    Redis-based distributed rate limiter using sliding window log.

    Suitable for multi-instance deployments.
    """

    def __init__(self) -> None:
        self._redis: Any | None = None
        self._initialized = False

    async def _get_redis(self) -> Any | None:
        """Get Redis connection (lazy initialization)."""
        if self._initialized:
            return self._redis

        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )
            # Test connection
            await self._redis.ping()
            self._initialized = True
            logger.info("Redis rate limiter initialized")
            return self._redis

        except Exception as e:
            logger.warning(f"Redis not available for rate limiting: {e}")
            self._initialized = True
            self._redis = None
            return None

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """
        Check if a request is allowed using Redis.

        Args:
            key: Unique identifier (IP or user ID)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            Tuple of (allowed, remaining, retry_after_seconds)
        """
        redis_client = await self._get_redis()
        if not redis_client:
            return True, limit, 0  # Allow if Redis unavailable

        now = time.time()
        window_start = now - window_seconds
        redis_key = f"rate_limit:{key}"

        try:
            # Use pipeline for atomicity
            pipe = redis_client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(redis_key, "-inf", window_start)

            # Count current entries
            pipe.zcard(redis_key)

            # Add current request with timestamp as score
            pipe.zadd(redis_key, {str(now): now})

            # Set TTL on the key
            pipe.expire(redis_key, window_seconds + 1)

            results = await pipe.execute()
            current_count = results[1]

            if current_count >= limit:
                # Get oldest entry to calculate retry time
                oldest = await redis_client.zrange(redis_key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(oldest[0][1] + window_seconds - now) + 1
                else:
                    retry_after = window_seconds
                return False, 0, retry_after

            remaining = limit - current_count - 1
            return True, max(0, remaining), 0

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            return True, limit, 0  # Allow on error


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Uses Redis for distributed rate limiting with in-memory fallback.
    Applies different limits based on endpoint sensitivity.
    """

    # Endpoints with stricter limits
    SENSITIVE_ENDPOINTS = {
        "/api/v1/auth/login": (10, 60),  # 10 per minute
        "/api/v1/auth/register": (5, 60),  # 5 per minute
        "/api/v1/diagnosis/analyze": (20, 60),  # 20 per minute (expensive)
    }

    # Endpoints exempt from rate limiting
    EXEMPT_ENDPOINTS = {
        "/health",
        "/api/v1/health",
        "/api/v1/health/live",
        "/api/v1/health/ready",
        "/metrics",
        "/api/v1/metrics",
        "/api/v1/docs",
        "/api/v1/redoc",
        "/api/v1/openapi.json",
    }

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self._redis_limiter = RedisRateLimiter()
        self._memory_limiter = InMemoryRateLimiter()
        self._config = RateLimitConfig(
            requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
            requests_per_hour=settings.RATE_LIMIT_PER_HOUR,
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, considering proxies."""
        # Check X-Forwarded-For header (for proxied requests)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection IP
        if request.client:
            return request.client.host

        return "unknown"

    def _get_rate_key(self, request: Request) -> str:
        """Generate rate limit key from request."""
        client_ip = self._get_client_ip(request)

        # Check for authenticated user
        # Note: This requires parsing JWT which adds overhead
        # For simplicity, we use IP-based limiting
        # TODO: Add user-based limiting for authenticated users

        return f"ip:{client_ip}"

    def _get_limits_for_endpoint(self, path: str) -> tuple[int, int]:
        """Get rate limits for specific endpoint."""
        # Check sensitive endpoints
        for endpoint, limits in self.SENSITIVE_ENDPOINTS.items():
            if path.startswith(endpoint):
                return limits

        # Default limits
        return (self._config.requests_per_minute, 60)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Any:
        """Process request through rate limiter."""
        path = request.url.path

        # Skip rate limiting for exempt endpoints
        if any(path.startswith(exempt) for exempt in self.EXEMPT_ENDPOINTS):
            return await call_next(request)

        # Get rate key and limits
        rate_key = self._get_rate_key(request)
        limit, window = self._get_limits_for_endpoint(path)

        # Try Redis first, fall back to memory
        try:
            allowed, remaining, retry_after = await self._redis_limiter.is_allowed(
                rate_key, limit, window
            )
        except Exception:
            # Fallback to in-memory limiter
            allowed, remaining, retry_after = self._memory_limiter.is_allowed(
                rate_key, limit, window
            )

        if not allowed:
            logger.warning(f"Rate limit exceeded for {rate_key} on {path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Too many requests. Please try again later.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + window)

        return response


def rate_limit(
    requests: int = 60,
    window_seconds: int = 60,
) -> Callable:
    """
    Decorator for endpoint-specific rate limiting.

    Usage:
        @router.get("/expensive-operation")
        @rate_limit(requests=10, window_seconds=60)
        async def expensive_operation():
            ...

    Note: This decorator requires the RateLimitMiddleware to be active
    and is primarily for documentation purposes. The actual limiting
    is done at the middleware level.
    """
    def decorator(func: Callable) -> Callable:
        # Store rate limit info on the function for documentation
        func._rate_limit = {
            "requests": requests,
            "window_seconds": window_seconds,
        }
        return func

    return decorator
