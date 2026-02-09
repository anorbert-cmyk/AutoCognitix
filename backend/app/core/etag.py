"""
ETag middleware for HTTP caching.

Provides efficient caching by computing ETags for responses and handling
conditional requests (If-None-Match).

Benefits:
- Reduces bandwidth by returning 304 Not Modified for unchanged content
- Improves perceived performance on client side
- Works with browser caching and CDNs
"""

import hashlib
import logging
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ETagMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds ETag headers and handles conditional requests.

    Features:
    - Generates weak ETags based on response content hash
    - Returns 304 Not Modified when ETag matches
    - Configurable paths to include/exclude
    - Only applies to GET/HEAD requests with 200 responses
    """

    def __init__(
        self,
        app,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        weak_etag: bool = True,
    ) -> None:
        """
        Initialize ETag middleware.

        Args:
            app: ASGI application
            include_paths: Paths to include (None = all paths)
            exclude_paths: Paths to exclude from ETag generation
            weak_etag: Use weak ETags (W/ prefix) - recommended for semantic equality
        """
        super().__init__(app)
        self.include_paths = include_paths
        self.exclude_paths: set[str] = set(
            exclude_paths
            or [
                "/api/v1/metrics",
                "/health",
                "/api/v1/auth",
            ]
        )
        self.weak_etag = weak_etag

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add ETag headers to response."""

        # Only process GET/HEAD requests
        if request.method not in ("GET", "HEAD"):
            response: Response = await call_next(request)
            return response

        # Check if path should be processed
        path = request.url.path
        if not self._should_process_path(path):
            response = await call_next(request)
            return response

        # Get the response
        response = await call_next(request)

        # Only process successful responses
        if response.status_code != 200:
            return response

        # Skip streaming responses
        if not hasattr(response, "body"):
            # Need to read the response body for non-streaming responses
            body = b""
            async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                body += chunk

            # Generate ETag from body content
            etag = self._generate_etag(body)

            # Check If-None-Match header
            if_none_match = request.headers.get("if-none-match")
            if if_none_match and self._etag_matches(etag, if_none_match):
                return Response(
                    status_code=304,
                    headers={
                        "ETag": etag,
                        "Cache-Control": response.headers.get(
                            "Cache-Control", "public, max-age=300"
                        ),
                    },
                )

            # Return response with ETag header
            return Response(
                content=body,
                status_code=response.status_code,
                headers={
                    **dict(response.headers),
                    "ETag": etag,
                },
                media_type=response.media_type,
            )

        return response

    def _should_process_path(self, path: str) -> bool:
        """Check if path should have ETag processing."""
        # Check exclusions first
        for excluded in self.exclude_paths:
            if path.startswith(excluded):
                return False

        # Check inclusions if specified
        if self.include_paths:
            return any(path.startswith(included) for included in self.include_paths)

        return True

    def _generate_etag(self, content: bytes) -> str:
        """Generate ETag from content bytes."""
        hash_value = hashlib.md5(content).hexdigest()

        if self.weak_etag:
            return f'W/"{hash_value}"'
        return f'"{hash_value}"'

    def _etag_matches(self, etag: str, if_none_match: str) -> bool:
        """
        Check if ETag matches If-None-Match header.

        Handles multiple ETags separated by commas and weak comparison.
        """

        # Normalize the ETag (remove W/ prefix for comparison if needed)
        def normalize(tag: str) -> str:
            tag = tag.strip()
            if tag.startswith("W/"):
                tag = tag[2:]
            return tag.strip('"')

        etag_normalized = normalize(etag)

        # Parse If-None-Match (can be comma-separated list)
        for tag in if_none_match.split(","):
            tag = tag.strip()
            if tag == "*" or normalize(tag) == etag_normalized:
                return True

        return False


class CacheControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds Cache-Control headers based on path patterns.

    Provides different caching strategies for different content types.
    """

    # Default cache settings by path pattern
    DEFAULT_CACHE_RULES = {
        "/api/v1/dtc/categories": "public, max-age=86400",  # 24 hours (static)
        "/api/v1/vehicles/makes": "public, max-age=86400",  # 24 hours (static)
        "/api/v1/vehicles/years": "public, max-age=86400",  # 24 hours (static)
        "/api/v1/dtc/search": "public, max-age=300",  # 5 minutes
        "/api/v1/dtc/": "public, max-age=3600",  # 1 hour (DTC details)
        "/api/v1/vehicles/": "public, max-age=3600",  # 1 hour
        "/api/v1/diagnosis/": "private, no-cache",  # Always fresh (user data)
        "/api/v1/auth/": "private, no-store",  # Never cache auth
    }

    def __init__(
        self,
        app,
        cache_rules: dict | None = None,
    ) -> None:
        """
        Initialize Cache-Control middleware.

        Args:
            app: ASGI application
            cache_rules: Path prefix -> Cache-Control value mapping
        """
        super().__init__(app)
        self.cache_rules = cache_rules or self.DEFAULT_CACHE_RULES

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add Cache-Control headers based on path."""
        response: Response = await call_next(request)

        # Only add cache headers to successful GET/HEAD responses
        if (
            request.method in ("GET", "HEAD")
            and response.status_code == 200
            and "Cache-Control" not in response.headers
        ):
            path = request.url.path

            # Find matching cache rule
            for prefix, cache_control in self.cache_rules.items():
                if path.startswith(prefix):
                    response.headers["Cache-Control"] = cache_control
                    break
            else:
                # Default: short cache for API responses
                response.headers["Cache-Control"] = "public, max-age=60"

        return response
