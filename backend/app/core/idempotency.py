"""
Idempotency-Key middleware for preventing duplicate POST requests.

Stores request results keyed by Idempotency-Key header in Redis.
If the same key is seen again, returns the cached response instead of re-processing.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

if TYPE_CHECKING:
    from app.db.redis_cache import RedisCacheService

logger = logging.getLogger(__name__)

IDEMPOTENCY_TTL = 86400  # 24 hours
MAX_IDEMPOTENCY_KEY_LENGTH = 256  # Prevent DoS via huge headers
MAX_CACHEABLE_BODY_SIZE = 1024 * 1024  # 1MB - skip caching larger responses


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Middleware that handles Idempotency-Key header for POST requests."""

    @staticmethod
    async def _get_cache() -> Optional[RedisCacheService]:
        """Get Redis cache service, returning None if unavailable."""
        try:
            from app.db.redis_cache import get_cache_service

            cache = await get_cache_service()
            if cache.is_circuit_open():
                return None
            return cache
        except Exception:
            return None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Only apply to POST/PUT methods with a valid Idempotency-Key header
        idempotency_key: Optional[str] = request.headers.get("Idempotency-Key")
        skip = (
            request.method not in ("POST", "PUT")
            or not idempotency_key
            or len(idempotency_key or "") > MAX_IDEMPOTENCY_KEY_LENGTH
        )
        if skip:
            return await call_next(request)

        # Get Redis cache (graceful degradation: skip if unavailable)
        cache = await self._get_cache()
        if cache is None:
            return await call_next(request)

        # Include request body hash in cache key to prevent collisions
        # (same key + different payload should not return cached response)
        try:
            body_bytes = await request.body()
            body_hash = hashlib.sha256(body_bytes).hexdigest()[:16]

            async def receive():
                return {"type": "http.request", "body": body_bytes}

            request._receive = receive
        except Exception:
            body_hash = "nobody"

        # Bind the cache key to the caller's credentials so one user's cached
        # response can never be replayed to a different user (or an
        # unauthenticated caller) that reuses the same key/path/body.
        # NOTE (follow-up): this middleware currently runs before auth, so it
        # cannot see the resolved user id. Keying on the raw Authorization
        # header is a defensive mitigation; a full fix requires reordering the
        # middleware to run after authentication.
        auth_identity = request.headers.get("Authorization", "")
        cache_key = (
            f"idempotency:"
            f"{hashlib.sha256(f'{auth_identity}:{idempotency_key}:{request.url.path}:{body_hash}'.encode()).hexdigest()}"
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

        # Only consider non-streaming 2xx responses for caching
        content_type = response.headers.get("content-type", "")
        content_encoding = response.headers.get("content-encoding", "")
        is_streaming = "text/event-stream" in content_type
        if 200 <= response.status_code < 300 and not is_streaming:
            # Fully drain the response body first. We must always rebuild the
            # response from the buffered bytes, because reading body_iterator
            # exhausts it -- returning the original `response` afterward would
            # send an empty body to the client.
            try:
                body = b""
                async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                    body += chunk if isinstance(chunk, bytes) else chunk.encode()
            except Exception as e:
                logger.warning("Failed to read idempotent response body: %s", e)
                return response

            # Rebuild the intact response (full body, original headers) so
            # oversized/compressed responses are streamed through unchanged
            # instead of being truncated.
            buffered_response = Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

            # Only cache small, uncompressed, UTF-8-decodable responses.
            # Skip gzip/deflate-encoded bodies (body.decode() would raise on
            # compressed bytes) and anything above the size cap.
            if not content_encoding and len(body) <= MAX_CACHEABLE_BODY_SIZE:
                try:
                    await cache.set(
                        cache_key,
                        {"body": body.decode(), "status_code": response.status_code},
                        ttl=IDEMPOTENCY_TTL,
                    )
                except Exception as e:
                    logger.warning("Failed to cache idempotent response: %s", e)

            return buffered_response

        return response
