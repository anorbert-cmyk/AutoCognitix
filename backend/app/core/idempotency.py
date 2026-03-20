"""
Idempotency-Key middleware for preventing duplicate POST requests.

Stores request results keyed by Idempotency-Key header in Redis.
If the same key is seen again, returns the cached response instead of re-processing.
"""

import hashlib
import logging
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

IDEMPOTENCY_TTL = 86400  # 24 hours


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Middleware that handles Idempotency-Key header for POST requests."""

    @staticmethod
    async def _get_cache() -> Optional["RedisCacheService"]:  # noqa: F821
        """Get Redis cache service, returning None if unavailable."""
        try:
            from app.db.redis_cache import get_cache_service

            cache = await get_cache_service()
            if cache._circuit_open:
                return None
            return cache
        except Exception:
            return None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Only apply to POST/PUT methods with an Idempotency-Key header
        idempotency_key: Optional[str] = request.headers.get("Idempotency-Key")
        if request.method not in ("POST", "PUT") or not idempotency_key:
            return await call_next(request)

        # Get Redis cache (graceful degradation: skip if unavailable)
        cache = await self._get_cache()
        if cache is None:
            return await call_next(request)

        # Create cache key from idempotency key + endpoint path
        cache_key = (
            f"idempotency:"
            f"{hashlib.sha256(f'{idempotency_key}:{request.url.path}'.encode()).hexdigest()}"
        )

        # Check for cached response
        try:
            cached = await cache.get(cache_key)
            if cached is not None:
                return Response(
                    content=cached["body"],
                    status_code=cached["status_code"],
                    headers={
                        "X-Idempotent-Replayed": "true",
                        "Content-Type": "application/json",
                    },
                )
        except Exception:
            pass  # Proceed normally if cache read fails

        # Process request
        response = await call_next(request)

        # Cache successful responses (2xx)
        if 200 <= response.status_code < 300:
            try:
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk if isinstance(chunk, bytes) else chunk.encode()

                await cache.set(
                    cache_key,
                    {"body": body.decode(), "status_code": response.status_code},
                    ttl=IDEMPOTENCY_TTL,
                )

                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            except Exception as e:
                logger.warning("Failed to cache idempotent response: %s", e)

        return response
