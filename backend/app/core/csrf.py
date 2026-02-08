"""
CSRF Protection Middleware for FastAPI

Implements double-submit cookie pattern for CSRF protection.
"""

import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.logging import get_logger

logger = get_logger(__name__)


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CSRF protection middleware using double-submit cookie pattern.

    For state-changing requests (POST, PUT, PATCH, DELETE):
    - Client must send X-CSRF-Token header matching csrf_token cookie
    - Returns 403 if token is missing or mismatched

    For safe methods (GET, HEAD, OPTIONS, TRACE):
    - Automatically sets csrf_token cookie if not present
    """

    SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}
    CSRF_HEADER = "X-CSRF-Token"
    CSRF_COOKIE = "csrf_token"

    def __init__(self, app, exclude_paths: list[str] | None = None):
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
        # Skip CSRF for excluded paths (e.g., health checks, metrics)
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Skip CSRF for safe methods - just ensure cookie is set
        if request.method in self.SAFE_METHODS:
            response = await call_next(request)
            return self._ensure_csrf_cookie(request, response)

        # Validate CSRF for state-changing methods
        token_header = request.headers.get(self.CSRF_HEADER)
        token_cookie = request.cookies.get(self.CSRF_COOKIE)

        if not token_header or not token_cookie:
            logger.warning(
                f"CSRF token missing - Method: {request.method}, "
                f"Path: {request.url.path}, "
                f"Header: {bool(token_header)}, Cookie: {bool(token_cookie)}"
            )
            return JSONResponse(
                {"detail": "CSRF token missing", "code": "csrf_token_missing"},
                status_code=403,
            )

        if token_header != token_cookie:
            logger.warning(
                f"CSRF token mismatch - Method: {request.method}, Path: {request.url.path}"
            )
            return JSONResponse(
                {"detail": "CSRF token invalid", "code": "csrf_token_invalid"},
                status_code=403,
            )

        # Token valid, proceed with request
        response = await call_next(request)
        return response

    def _ensure_csrf_cookie(self, request: Request, response: Response) -> Response:
        """
        Ensure CSRF cookie is set on response.

        If cookie doesn't exist, generate new token and set cookie.
        """
        if not request.cookies.get(self.CSRF_COOKIE):
            csrf_token = secrets.token_urlsafe(32)
            response.set_cookie(
                key=self.CSRF_COOKIE,
                value=csrf_token,
                httponly=False,  # Must be readable by JavaScript
                samesite="strict",  # Strict same-site policy
                secure=True,  # HTTPS only in production
                max_age=3600,  # 1 hour
            )
            logger.debug(f"CSRF cookie set for request: {request.url.path}")

        return response
