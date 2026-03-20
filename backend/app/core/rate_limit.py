"""
Rate limiting middleware for authentication endpoints.

Uses in-memory storage (for single instance) or Redis (for distributed).
Implements sliding window rate limiting.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import HTTPException, Request, Response, status

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
# lockout_threshold must be LOWER than requests_per_minute to trigger lockout
AUTH_CONFIG = RateLimitConfig(
    requests_per_minute=5,
    requests_per_hour=30,
    lockout_threshold=3,
    lockout_duration_seconds=900,
)

# Diagnosis endpoints - expensive AI/RAG operations
DIAGNOSIS_CONFIG = RateLimitConfig(
    requests_per_minute=3,
    requests_per_hour=20,
    lockout_threshold=10,
    lockout_duration_seconds=300,
)

# Search endpoints - more permissive
SEARCH_CONFIG = RateLimitConfig(
    requests_per_minute=30,
    requests_per_hour=500,
    lockout_threshold=50,
    lockout_duration_seconds=60,
)


@dataclass
class RateLimitInfo:
    """Rate limit status returned after a check."""

    allowed: bool
    retry_after: Optional[int]
    limit: int
    remaining: int
    reset_seconds: int


class InMemoryRateLimiter:
    """
    In-memory rate limiter using sliding window algorithm.

    For production with multiple instances, use Redis-based rate limiter.
    """

    MAX_TRACKED_CLIENTS = 10000

    def __init__(self):
        # Structure: {client_key: [(timestamp, count), ...]}
        # Use regular dict (not defaultdict) to avoid phantom entries on read access
        self._minute_windows: Dict[str, list] = {}
        self._hour_windows: Dict[str, list] = {}
        self._lockouts: Dict[str, float] = {}  # {client_key: lockout_expires_at}

    def _clean_old_entries(self, entries: list, window_seconds: int, now: float) -> list:
        """Remove entries outside the time window."""
        cutoff = now - window_seconds
        return [e for e in entries if e[0] > cutoff]

    def _get_request_count(self, entries: list) -> int:
        """Get total request count from entries."""
        return sum(e[1] for e in entries)

    def _compute_reset_seconds(self, entries: list, window_seconds: int, now: float) -> int:
        """Compute seconds until the oldest entry in the window expires."""
        if entries:
            oldest = entries[0][0]
            return max(1, int(oldest + window_seconds - now) + 1)
        return window_seconds

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
    ) -> RateLimitInfo:
        """
        Check if request is within rate limits.

        Args:
            client_key: Unique identifier for the client (IP, user ID, etc.)
            config: Rate limit configuration

        Returns:
            RateLimitInfo with allowed status and header values.
        """
        now = time.time()

        # Evict oldest entries if memory cap exceeded
        if len(self._minute_windows) > self.MAX_TRACKED_CLIENTS:
            oldest_keys = sorted(
                self._minute_windows.keys(),
                key=lambda k: self._minute_windows[k][-1][0] if self._minute_windows[k] else 0,
            )[: self.MAX_TRACKED_CLIENTS // 10]
            for key in oldest_keys:
                self._minute_windows.pop(key, None)
                self._hour_windows.pop(key, None)
                self._lockouts.pop(key, None)

        # Check lockout first
        if self.is_locked_out(client_key):
            retry_after = int(self._lockouts[client_key] - now)
            return RateLimitInfo(
                allowed=False,
                retry_after=retry_after,
                limit=config.requests_per_minute,
                remaining=0,
                reset_seconds=retry_after,
            )

        # Clean old entries
        minute_entries = self._minute_windows.get(client_key, [])
        hour_entries = self._hour_windows.get(client_key, [])
        minute_entries = self._clean_old_entries(minute_entries, 60, now)
        hour_entries = self._clean_old_entries(hour_entries, 3600, now)
        if minute_entries:
            self._minute_windows[client_key] = minute_entries
        else:
            self._minute_windows.pop(client_key, None)
        if hour_entries:
            self._hour_windows[client_key] = hour_entries
        else:
            self._hour_windows.pop(client_key, None)

        # Check minute limit
        minute_count = self._get_request_count(minute_entries)
        if minute_count >= config.requests_per_minute:
            retry_after = self._compute_reset_seconds(minute_entries, 60, now)
            return RateLimitInfo(
                allowed=False,
                retry_after=retry_after,
                limit=config.requests_per_minute,
                remaining=0,
                reset_seconds=retry_after,
            )

        # Check hour limit
        hour_count = self._get_request_count(hour_entries)
        if hour_count >= config.requests_per_hour:
            retry_after = self._compute_reset_seconds(hour_entries, 3600, now)
            return RateLimitInfo(
                allowed=False,
                retry_after=retry_after,
                limit=config.requests_per_hour,
                remaining=0,
                reset_seconds=retry_after,
            )

        # Check if approaching lockout threshold
        if minute_count >= config.lockout_threshold:
            self._lockouts[client_key] = now + config.lockout_duration_seconds
            logger.warning(f"Rate limit lockout triggered for: {client_key}")
            return RateLimitInfo(
                allowed=False,
                retry_after=config.lockout_duration_seconds,
                limit=config.requests_per_minute,
                remaining=0,
                reset_seconds=config.lockout_duration_seconds,
            )

        # Allowed - compute remaining and reset
        remaining = max(0, config.requests_per_minute - minute_count)
        reset_seconds = self._compute_reset_seconds(minute_entries, 60, now)

        return RateLimitInfo(
            allowed=True,
            retry_after=None,
            limit=config.requests_per_minute,
            remaining=remaining,
            reset_seconds=reset_seconds,
        )

    def record_request(self, client_key: str) -> None:
        """Record a request for the client."""
        now = time.time()
        self._minute_windows.setdefault(client_key, []).append((now, 1))
        self._hour_windows.setdefault(client_key, []).append((now, 1))

    def reset(self, client_key: str) -> None:
        """Reset rate limit counters for a client."""
        self._minute_windows.pop(client_key, None)
        self._hour_windows.pop(client_key, None)
        self._lockouts.pop(client_key, None)


# Global rate limiter instance
_rate_limiter = InMemoryRateLimiter()


def _set_rate_limit_headers(response: Response, info: RateLimitInfo) -> None:
    """Set standard rate limit headers on a response."""
    response.headers["X-RateLimit-Limit"] = str(info.limit)
    response.headers["X-RateLimit-Remaining"] = str(info.remaining)
    response.headers["X-RateLimit-Reset"] = str(info.reset_seconds)


def get_client_key(request: Request) -> str:
    """
    Get unique client identifier from request.

    Uses IP address, with X-Forwarded-For header support for proxies.
    """
    # Check for X-Forwarded-For header (common with reverse proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the rightmost (most recently added by proxy) IP
        # The first IP can be spoofed by the client; the last is added by the trusted proxy
        ips = [ip.strip() for ip in forwarded_for.split(",")]
        client_ip = ips[-1]
    else:
        # Fall back to direct client IP
        client_ip = str(request.client.host) if request.client else "unknown"

    return str(client_ip)


async def check_rate_limit(
    request: Request,
    response: Response,
    config: RateLimitConfig = DEFAULT_CONFIG,
) -> None:
    """
    FastAPI dependency for rate limiting.

    Raises HTTPException if rate limit exceeded.
    Sets X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset headers
    on both successful and rejected responses.

    Usage:
        @router.post("/endpoint")
        async def endpoint(
            _: None = Depends(check_rate_limit),
        ):
            ...
    """
    client_key = get_client_key(request)
    info = _rate_limiter.check_rate_limit(client_key, config)

    if not info.allowed:
        logger.warning(f"Rate limit exceeded for: {client_key}")
        headers = {
            "X-RateLimit-Limit": str(info.limit),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(info.reset_seconds),
        }
        if info.retry_after is not None:
            headers["Retry-After"] = str(info.retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Túl sok kérés. Kérjük, próbálja újra később.",
            headers=headers,
        )

    # Record the request (decrements remaining by 1)
    _rate_limiter.record_request(client_key)

    # Set rate limit headers directly on the response object
    # The remaining count accounts for this request being recorded
    updated_info = RateLimitInfo(
        allowed=True,
        retry_after=None,
        limit=info.limit,
        remaining=max(0, info.remaining - 1),
        reset_seconds=info.reset_seconds,
    )
    _set_rate_limit_headers(response, updated_info)

    # Also store on request state for downstream access if needed
    request.state.rate_limit_info = updated_info


async def check_auth_rate_limit(request: Request, response: Response) -> None:
    """
    Rate limiting specifically for auth endpoints (stricter limits).

    Usage:
        @router.post("/login")
        async def login(
            _: None = Depends(check_auth_rate_limit),
        ):
            ...
    """
    await check_rate_limit(request, response, AUTH_CONFIG)


async def check_diagnosis_rate_limit(request: Request, response: Response) -> None:
    """
    Rate limiting for diagnosis endpoints (expensive AI/RAG operations).

    Usage:
        @router.post("/analyze")
        async def analyze(
            _: None = Depends(check_diagnosis_rate_limit),
        ):
            ...
    """
    await check_rate_limit(request, response, DIAGNOSIS_CONFIG)


async def check_search_rate_limit(request: Request, response: Response) -> None:
    """
    Rate limiting for search endpoints (more permissive).

    Usage:
        @router.get("/search")
        async def search(
            _: None = Depends(check_search_rate_limit),
        ):
            ...
    """
    await check_rate_limit(request, response, SEARCH_CONFIG)


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
    Sets X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset headers.
    """

    def __init__(
        self,
        app,
        config: RateLimitConfig = DEFAULT_CONFIG,
        paths: Optional[List[str]] = None,
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
                ips = [ip.strip() for ip in header[1].decode().split(",")]
                client_host = ips[-1]  # Last IP is from the trusted proxy
                break

        if not client_host:
            client = scope.get("client")
            client_host = client[0] if client else "unknown"

        # Check rate limit
        info = _rate_limiter.check_rate_limit(client_host, self.config)

        if not info.allowed:
            # Send 429 response with rate limit headers
            response_body = b'{"detail":"Tul sok keres. Probalja ujra kesobb."}'
            headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(response_body)).encode()),
                (b"x-ratelimit-limit", str(info.limit).encode()),
                (b"x-ratelimit-remaining", b"0"),
                (b"x-ratelimit-reset", str(info.reset_seconds).encode()),
            ]
            if info.retry_after is not None:
                headers.append((b"retry-after", str(info.retry_after).encode()))

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

        # Record request and continue, injecting rate limit headers
        _rate_limiter.record_request(client_host)
        remaining = max(0, info.remaining - 1)
        reset_seconds = info.reset_seconds

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-ratelimit-limit", str(info.limit).encode()))
                headers.append((b"x-ratelimit-remaining", str(remaining).encode()))
                headers.append((b"x-ratelimit-reset", str(reset_seconds).encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)
