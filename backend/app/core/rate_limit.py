"""
Rate limiting middleware for authentication endpoints.

Uses in-memory storage (for single instance) or Redis (for distributed).
Implements sliding window rate limiting.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int
    requests_per_hour: int
    lockout_threshold: int = 10  # Number of requests before temporary ban
    lockout_duration_seconds: int = 300  # 5 minutes


# Default configuration from settings
DEFAULT_CONFIG = RateLimitConfig(
    requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
    requests_per_hour=settings.RATE_LIMIT_PER_HOUR,
)

# Auth-specific stricter limits
AUTH_CONFIG = RateLimitConfig(
    requests_per_minute=10,  # 10 auth requests per minute
    requests_per_hour=100,  # 100 auth requests per hour
    lockout_threshold=20,  # Lock after 20 failed attempts
    lockout_duration_seconds=600,  # 10 minute lockout
)


class InMemoryRateLimiter:
    """
    In-memory rate limiter using sliding window algorithm.

    For production with multiple instances, use Redis-based rate limiter.
    """

    def __init__(self):
        # Structure: {client_key: [(timestamp, count), ...]}
        self._minute_windows: dict[str, list] = defaultdict(list)
        self._hour_windows: dict[str, list] = defaultdict(list)
        self._lockouts: dict[str, float] = {}  # {client_key: lockout_expires_at}

    def _clean_old_entries(self, entries: list, window_seconds: int, now: float) -> list:
        """Remove entries outside the time window."""
        cutoff = now - window_seconds
        return [e for e in entries if e[0] > cutoff]

    def _get_request_count(self, entries: list) -> int:
        """Get total request count from entries."""
        return sum(e[1] for e in entries)

    def is_locked_out(self, client_key: str) -> bool:
        """Check if client is currently locked out."""
        if client_key in self._lockouts:
            if time.time() < self._lockouts[client_key]:
                return True
            # Lockout expired, remove it
            del self._lockouts[client_key]
        return False

    def check_rate_limit(
        self,
        client_key: str,
        config: RateLimitConfig = DEFAULT_CONFIG,
    ) -> tuple[bool, int | None]:
        """
        Check if request is within rate limits.

        Args:
            client_key: Unique identifier for the client (IP, user ID, etc.)
            config: Rate limit configuration

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = time.time()

        # Check lockout first
        if self.is_locked_out(client_key):
            retry_after = int(self._lockouts[client_key] - now)
            return False, retry_after

        # Clean old entries
        self._minute_windows[client_key] = self._clean_old_entries(
            self._minute_windows[client_key], 60, now
        )
        self._hour_windows[client_key] = self._clean_old_entries(
            self._hour_windows[client_key], 3600, now
        )

        # Check minute limit
        minute_count = self._get_request_count(self._minute_windows[client_key])
        if minute_count >= config.requests_per_minute:
            # Find when oldest entry expires
            if self._minute_windows[client_key]:
                oldest = self._minute_windows[client_key][0][0]
                retry_after = int(oldest + 60 - now) + 1
            else:
                retry_after = 60
            return False, retry_after

        # Check hour limit
        hour_count = self._get_request_count(self._hour_windows[client_key])
        if hour_count >= config.requests_per_hour:
            if self._hour_windows[client_key]:
                oldest = self._hour_windows[client_key][0][0]
                retry_after = int(oldest + 3600 - now) + 1
            else:
                retry_after = 3600
            return False, retry_after

        # Check if approaching lockout threshold
        if minute_count >= config.lockout_threshold:
            self._lockouts[client_key] = now + config.lockout_duration_seconds
            logger.warning(f"Rate limit lockout triggered for: {client_key}")
            return False, config.lockout_duration_seconds

        return True, None

    def record_request(self, client_key: str) -> None:
        """Record a request for the client."""
        now = time.time()
        self._minute_windows[client_key].append((now, 1))
        self._hour_windows[client_key].append((now, 1))

    def reset(self, client_key: str) -> None:
        """Reset rate limit counters for a client."""
        self._minute_windows.pop(client_key, None)
        self._hour_windows.pop(client_key, None)
        self._lockouts.pop(client_key, None)


# Global rate limiter instance
_rate_limiter = InMemoryRateLimiter()


def get_client_key(request: Request) -> str:
    """
    Get unique client identifier from request.

    Uses IP address, with X-Forwarded-For header support for proxies.
    """
    # Check for X-Forwarded-For header (common with reverse proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain (original client)
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        # Fall back to direct client IP
        client_ip = request.client.host if request.client else "unknown"

    return client_ip


async def check_rate_limit(
    request: Request,
    config: RateLimitConfig = DEFAULT_CONFIG,
) -> None:
    """
    FastAPI dependency for rate limiting.

    Raises HTTPException if rate limit exceeded.

    Usage:
        @router.post("/endpoint")
        async def endpoint(
            _: None = Depends(check_rate_limit),
        ):
            ...
    """
    client_key = get_client_key(request)
    allowed, retry_after = _rate_limiter.check_rate_limit(client_key, config)

    if not allowed:
        logger.warning(f"Rate limit exceeded for: {client_key}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Túl sok kérés. Kérjük, próbálja újra később.",
            headers={"Retry-After": str(retry_after)} if retry_after else None,
        )

    # Record the request
    _rate_limiter.record_request(client_key)


async def check_auth_rate_limit(request: Request) -> None:
    """
    Rate limiting specifically for auth endpoints (stricter limits).

    Usage:
        @router.post("/login")
        async def login(
            _: None = Depends(check_auth_rate_limit),
        ):
            ...
    """
    await check_rate_limit(request, AUTH_CONFIG)


def reset_rate_limit(client_key: str) -> None:
    """
    Reset rate limits for a specific client.

    Useful after successful authentication to give user a fresh start.
    """
    _rate_limiter.reset(client_key)


# =============================================================================
# Rate Limit Middleware (Alternative approach)
# =============================================================================


class RateLimitMiddleware:
    """
    ASGI middleware for rate limiting.

    Alternative to dependency-based rate limiting.
    Applies to all requests or specific paths.
    """

    def __init__(
        self,
        app,
        config: RateLimitConfig = DEFAULT_CONFIG,
        paths: list[str] | None = None,
    ):
        self.app = app
        self.config = config
        self.paths = paths  # If None, apply to all paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check if path should be rate limited
        path = scope.get("path", "")
        if self.paths and not any(path.startswith(p) for p in self.paths):
            await self.app(scope, receive, send)
            return

        # Get client key
        client_host = None
        for header in scope.get("headers", []):
            if header[0] == b"x-forwarded-for":
                client_host = header[1].decode().split(",")[0].strip()
                break

        if not client_host:
            client = scope.get("client")
            client_host = client[0] if client else "unknown"

        # Check rate limit
        allowed, retry_after = _rate_limiter.check_rate_limit(client_host, self.config)

        if not allowed:
            # Send 429 response
            response_body = b'{"detail":"Tul sok keres. Probalja ujra kesobb."}'
            headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(response_body)).encode()),
            ]
            if retry_after:
                headers.append((b"retry-after", str(retry_after).encode()))

            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": headers,
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": response_body,
                }
            )
            return

        # Record request and continue
        _rate_limiter.record_request(client_host)
        await self.app(scope, receive, send)
