"""
CSRF Protection Middleware for FastAPI

Implements header-only, HMAC-signed-token CSRF protection.
"""

from typing import List, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.logging import get_logger
from app.core.security import verify_csrf_token

logger = get_logger(__name__)


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CSRF protection middleware using header-only, HMAC-signed tokens.

    For state-changing requests (POST, PUT, PATCH, DELETE):
    - Client must send a valid HMAC-signed token in the X-CSRF-Token header
    - Returns 403 if the token is missing or the signature is invalid

    For safe methods (GET, HEAD, OPTIONS, TRACE) and excluded paths:
    - Request passes through without validation

    No cookie is used, so protection works across cross-site deployments
    where samesite cookies are never sent.
    """

    SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}
    CSRF_HEADER = "X-CSRF-Token"

    def __init__(self, app, exclude_paths: Optional[List[str]] = None):
        """
        Initialize CSRF middleware.

        Args:
            app: The ASGI application
            exclude_paths: List of path prefixes to exclude from CSRF validation
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or []

    async def dispatch(self, request: Request, call_next):
        """Process request with CSRF validation."""
        # Skip CSRF for excluded paths (e.g., health checks, metrics, auth bootstrap)
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Skip CSRF for safe methods
        if request.method in self.SAFE_METHODS:
            return await call_next(request)

        # Validate the signed CSRF token from the header for state-changing methods
        if not verify_csrf_token(request.headers.get(self.CSRF_HEADER)):
            logger.warning(
                f"CSRF token invalid - Method: {request.method}, Path: {request.url.path}"
            )
            return JSONResponse(
                {"detail": "CSRF token invalid", "code": "csrf_token_invalid"},
                status_code=403,
            )

        # Token valid, proceed with request
        return await call_next(request)
