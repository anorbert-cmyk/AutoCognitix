"""Unit tests for app.core.rate_limiter and app.core.rate_limit modules."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# rate_limiter.py (middleware-based)
# ---------------------------------------------------------------------------
from app.core.rate_limiter import (
    InMemoryRateLimiter as MiddlewareInMemoryRateLimiter,
    RateLimitConfig as MiddlewareRateLimitConfig,
    RateLimitMiddleware,
    RedisRateLimiter,
    rate_limit,
)

# ---------------------------------------------------------------------------
# rate_limit.py (dependency-based)
# ---------------------------------------------------------------------------
from app.core.rate_limit import (
    AUTH_CONFIG,
    DEFAULT_CONFIG,
    DIAGNOSIS_CONFIG,
    SEARCH_CONFIG,
    InMemoryRateLimiter as DepInMemoryRateLimiter,
    RateLimitConfig as DepRateLimitConfig,
    RateLimitInfo,
    RateLimitMiddleware as DepRateLimitMiddleware,
    _set_rate_limit_headers,
    check_auth_rate_limit,
    check_diagnosis_rate_limit,
    check_rate_limit,
    check_search_rate_limit,
    get_client_key,
    reset_rate_limit,
)
from typing import Optional


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mem_limiter():
    """Fresh middleware InMemoryRateLimiter for each test."""
    return MiddlewareInMemoryRateLimiter()


@pytest.fixture
def dep_limiter():
    """Fresh dependency InMemoryRateLimiter for each test."""
    return DepInMemoryRateLimiter()


@pytest.fixture
def redis_limiter():
    """Fresh RedisRateLimiter with uninitialised state."""
    rl = RedisRateLimiter()
    rl._initialized = False
    rl._redis = None
    return rl


def _make_request(
    ip: str = "1.2.3.4",
    path: str = "/api/v1/test",
    headers: Optional[dict] = None,
):
    """Build a minimal mock Request."""
    req = MagicMock()
    req.url.path = path
    req.client.host = ip
    req.headers = headers or {}
    req.state = MagicMock()
    return req


def _make_response():
    """Build a minimal mock Response."""
    resp = MagicMock()
    resp.headers = {}
    return resp


# ============================================================================
# MiddlewareRateLimitConfig tests
# ============================================================================


class TestMiddlewareRateLimitConfig:
    def test_defaults(self):
        cfg = MiddlewareRateLimitConfig()
        assert cfg.requests_per_minute == 60
        assert cfg.requests_per_hour == 1000
        assert cfg.burst_limit == 10
        assert cfg.block_duration_seconds == 60

    def test_custom_values(self):
        cfg = MiddlewareRateLimitConfig(requests_per_minute=5, block_duration_seconds=300)
        assert cfg.requests_per_minute == 5
        assert cfg.block_duration_seconds == 300


# ============================================================================
# InMemoryRateLimiter (rate_limiter.py) tests
# ============================================================================


class TestMiddlewareInMemoryRateLimiter:
    def test_first_request_allowed(self, mem_limiter):
        allowed, remaining, retry = mem_limiter.is_allowed("k1", 10, 60)
        assert allowed is True
        assert remaining == 9
        assert retry == 0

    def test_requests_up_to_limit(self, mem_limiter):
        # is_allowed checks THEN adds, so with limit=5, calls 1-5 are allowed
        # (counts 0..4 before adding), and call 6 is blocked (count=5 >= limit)
        for _ in range(5):
            allowed, _, _ = mem_limiter.is_allowed("k1", 5, 60)
            assert allowed is True
        # 6th request should be blocked
        allowed, remaining, _ = mem_limiter.is_allowed("k1", 5, 60)
        assert allowed is False
        assert remaining == 0

    def test_blocked_after_limit(self, mem_limiter):
        for _ in range(5):
            mem_limiter.is_allowed("k1", 5, 60, block_duration_seconds=120)
        allowed, _remaining, retry = mem_limiter.is_allowed("k1", 5, 60, block_duration_seconds=120)
        assert allowed is False
        assert retry > 0

    def test_block_expires(self, mem_limiter):
        # Simulate a block that has already expired
        mem_limiter._blocked["k1"] = time.time() - 1
        # Clear any requests so the count is 0
        mem_limiter._requests["k1"] = []
        allowed, _, _ = mem_limiter.is_allowed("k1", 3, 60)
        assert allowed is True
        assert "k1" not in mem_limiter._blocked

    def test_old_requests_cleaned(self, mem_limiter):
        old_ts = time.time() - 120
        mem_limiter._requests["k1"].append((old_ts, 1))
        allowed, remaining, _ = mem_limiter.is_allowed("k1", 5, 60)
        assert allowed is True
        assert remaining == 4  # old request was cleaned

    def test_separate_keys(self, mem_limiter):
        for _ in range(5):
            mem_limiter.is_allowed("a", 5, 60)
        # Key "b" should still be allowed
        allowed, _, _ = mem_limiter.is_allowed("b", 5, 60)
        assert allowed is True

    def test_max_tracked_clients_eviction(self, mem_limiter):
        mem_limiter.MAX_TRACKED_CLIENTS = 5
        # Add 6 clients
        for i in range(6):
            mem_limiter.is_allowed(f"client_{i}", 100, 60)
        # 7th call should trigger eviction, but still work
        allowed, _, _ = mem_limiter.is_allowed("client_new", 100, 60)
        assert allowed is True
        assert len(mem_limiter._requests) <= 6  # at most MAX+1 temporarily


# ============================================================================
# RedisRateLimiter tests
# ============================================================================


class TestRedisRateLimiter:
    @pytest.mark.asyncio
    async def test_redis_unavailable_fail_closed(self, redis_limiter):
        """When Redis cannot connect, requests should be denied (fail-closed)."""
        with patch("app.core.rate_limiter.settings") as mock_settings:
            mock_settings.REDIS_URL = "redis://nonexistent:6379"
            with patch("redis.asyncio.from_url", side_effect=ConnectionError("no redis")):
                allowed, _remaining, retry = await redis_limiter.is_allowed("k1", 10, 60)
        assert allowed is False
        assert retry == 60

    @pytest.mark.asyncio
    async def test_redis_already_initialized_none(self, redis_limiter):
        """When Redis was already tried and failed, returns fail-closed."""
        redis_limiter._initialized = True
        redis_limiter._redis = None
        allowed, _remaining, retry = await redis_limiter.is_allowed("k1", 10, 60)
        assert allowed is False
        assert retry == 60

    @pytest.mark.asyncio
    async def test_redis_successful_under_limit(self, redis_limiter):
        """When Redis works and under limit, request is allowed."""
        mock_redis = MagicMock()  # pipeline() is sync
        mock_pipe = MagicMock()
        # pipeline methods (zremrangebyscore, zcard, zadd, expire) are sync on the pipeline object
        mock_pipe.execute = AsyncMock(return_value=[None, 3, None, None])  # zcard=3
        mock_redis.pipeline.return_value = mock_pipe

        redis_limiter._initialized = True
        redis_limiter._redis = mock_redis

        allowed, remaining, retry = await redis_limiter.is_allowed("k1", 10, 60)
        assert allowed is True
        assert remaining == 6  # 10 - 3 - 1
        assert retry == 0

    @pytest.mark.asyncio
    async def test_redis_over_limit(self, redis_limiter):
        """When at or over limit, request is denied."""
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[None, 10, None, None])  # zcard=10, limit=10
        mock_redis.pipeline.return_value = mock_pipe
        # oldest entry score
        mock_redis.zrange = AsyncMock(return_value=[("ts", time.time() - 30)])

        redis_limiter._initialized = True
        redis_limiter._redis = mock_redis

        allowed, remaining, retry = await redis_limiter.is_allowed("k1", 10, 60)
        assert allowed is False
        assert remaining == 0
        assert retry > 0

    @pytest.mark.asyncio
    async def test_redis_over_limit_no_oldest(self, redis_limiter):
        """When over limit and no oldest entry, retry_after = window."""
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[None, 10, None, None])
        mock_redis.pipeline.return_value = mock_pipe
        mock_redis.zrange = AsyncMock(return_value=[])

        redis_limiter._initialized = True
        redis_limiter._redis = mock_redis

        allowed, _, retry = await redis_limiter.is_allowed("k1", 10, 60)
        assert allowed is False
        assert retry == 60

    @pytest.mark.asyncio
    async def test_redis_pipeline_error_fail_closed(self, redis_limiter):
        """Pipeline errors should fail-closed."""
        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = RuntimeError("pipe broken")

        redis_limiter._initialized = True
        redis_limiter._redis = mock_redis

        allowed, _, retry = await redis_limiter.is_allowed("k1", 10, 60)
        assert allowed is False
        assert retry == 60

    @pytest.mark.asyncio
    async def test_redis_init_success(self, redis_limiter):
        """Successful Redis initialization stores client."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            client = await redis_limiter._get_redis()
        assert client is mock_redis
        assert redis_limiter._initialized is True


# ============================================================================
# RateLimitMiddleware (rate_limiter.py) tests
# ============================================================================


class TestRateLimitMiddleware:
    def test_get_client_ip_direct(self):
        """Direct client IP when no proxy headers."""
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        req = _make_request(ip="10.0.0.1")
        assert mw._get_client_ip(req) == "10.0.0.1"

    def test_get_client_ip_forwarded(self):
        """X-Forwarded-For header, takes last IP."""
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        req = _make_request(headers={"X-Forwarded-For": "1.1.1.1, 2.2.2.2, 3.3.3.3"})
        assert mw._get_client_ip(req) == "3.3.3.3"

    def test_get_client_ip_real_ip(self):
        """X-Real-IP header."""
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        req = _make_request(headers={"X-Real-IP": "5.5.5.5"})
        assert mw._get_client_ip(req) == "5.5.5.5"

    def test_get_client_ip_no_client(self):
        """No client info at all."""
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        req = _make_request()
        req.client = None
        req.headers = {}
        assert mw._get_client_ip(req) == "unknown"

    def test_get_rate_key(self):
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        req = _make_request(ip="9.9.9.9")
        assert mw._get_rate_key(req) == "ip:9.9.9.9"

    def test_sensitive_endpoint_limits(self):
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        limit, window = mw._get_limits_for_endpoint("/api/v1/auth/login")
        assert limit == 10
        assert window == 60

    def test_default_endpoint_limits(self):
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        _limit, window = mw._get_limits_for_endpoint("/api/v1/vehicles")
        assert window == 60

    def test_exempt_endpoints(self):
        """Exempt endpoints are listed."""
        assert "/health" in RateLimitMiddleware.EXEMPT_ENDPOINTS
        assert "/api/v1/health" in RateLimitMiddleware.EXEMPT_ENDPOINTS

    @pytest.mark.asyncio
    async def test_dispatch_exempt_endpoint(self):
        """Exempt endpoints skip rate limiting."""
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        req = _make_request(path="/health")
        call_next = AsyncMock(return_value=MagicMock())
        await mw.dispatch(req, call_next)
        call_next.assert_awaited_once_with(req)

    @pytest.mark.asyncio
    async def test_dispatch_allowed(self):
        """Normal request under limit gets rate limit headers."""
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        mock_response = MagicMock()
        mock_response.headers = {}
        call_next = AsyncMock(return_value=mock_response)

        # Ensure Redis returns allowed
        mw._redis_limiter = AsyncMock()
        mw._redis_limiter.is_allowed = AsyncMock(return_value=(True, 59, 0))

        req = _make_request(path="/api/v1/test")
        resp = await mw.dispatch(req, call_next)
        assert resp.headers["X-RateLimit-Remaining"] == "59"

    @pytest.mark.asyncio
    async def test_dispatch_denied(self):
        """Request over limit returns 429."""
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        mw._redis_limiter = AsyncMock()
        mw._redis_limiter.is_allowed = AsyncMock(return_value=(False, 0, 30))

        req = _make_request(path="/api/v1/test")
        call_next = AsyncMock()
        resp = await mw.dispatch(req, call_next)
        assert resp.status_code == 429
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_redis_exception_falls_back(self):
        """When Redis raises, falls back to in-memory limiter."""
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        mw._redis_limiter = AsyncMock()
        mw._redis_limiter.is_allowed = AsyncMock(side_effect=RuntimeError("boom"))

        mock_response = MagicMock()
        mock_response.headers = {}
        call_next = AsyncMock(return_value=mock_response)

        req = _make_request(path="/api/v1/test")
        await mw.dispatch(req, call_next)
        # Should succeed (in-memory limiter allows first request)
        call_next.assert_awaited_once()


# ============================================================================
# rate_limit decorator (rate_limiter.py)
# ============================================================================


class TestRateLimitDecorator:
    def test_stores_metadata(self):
        @rate_limit(requests=5, window_seconds=120)
        async def dummy():
            pass

        assert dummy._rate_limit == {"requests": 5, "window_seconds": 120}

    def test_default_values(self):
        @rate_limit()
        async def dummy():
            pass

        assert dummy._rate_limit == {"requests": 60, "window_seconds": 60}


# ============================================================================
# InMemoryRateLimiter (rate_limit.py - dependency-based) tests
# ============================================================================


class TestDepInMemoryRateLimiter:
    def test_check_first_request_allowed(self, dep_limiter):
        cfg = DepRateLimitConfig(requests_per_minute=10, requests_per_hour=100)
        info = dep_limiter.check_rate_limit("c1", cfg)
        assert info.allowed is True
        assert info.remaining == 10  # no requests recorded yet

    def test_record_request(self, dep_limiter):
        dep_limiter.record_request("c1")
        assert len(dep_limiter._minute_windows.get("c1", [])) == 1
        assert len(dep_limiter._hour_windows.get("c1", [])) == 1

    def test_minute_limit_exceeded(self, dep_limiter):
        cfg = DepRateLimitConfig(
            requests_per_minute=3,
            requests_per_hour=100,
            lockout_threshold=100,  # high to avoid lockout
        )
        for _ in range(3):
            dep_limiter.record_request("c1")
        info = dep_limiter.check_rate_limit("c1", cfg)
        assert info.allowed is False
        assert info.remaining == 0
        assert info.retry_after is not None

    def test_hour_limit_exceeded(self, dep_limiter):
        cfg = DepRateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=3,
            lockout_threshold=1000,  # high to avoid lockout
        )
        # Record 3 requests but only add to hour window
        for _ in range(3):
            dep_limiter.record_request("c1")
        # Clear minute window so minute check passes
        dep_limiter._minute_windows.pop("c1", None)
        info = dep_limiter.check_rate_limit("c1", cfg)
        assert info.allowed is False

    def test_lockout_triggered(self, dep_limiter):
        cfg = DepRateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            lockout_threshold=3,
            lockout_duration_seconds=600,
        )
        for _ in range(3):
            dep_limiter.record_request("c1")
        info = dep_limiter.check_rate_limit("c1", cfg)
        assert info.allowed is False
        assert info.retry_after == 600

    def test_lockout_blocks_subsequent(self, dep_limiter):
        cfg = DepRateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            lockout_threshold=2,
            lockout_duration_seconds=300,
        )
        for _ in range(2):
            dep_limiter.record_request("c1")
        dep_limiter.check_rate_limit("c1", cfg)  # triggers lockout
        info = dep_limiter.check_rate_limit("c1", cfg)  # should be locked
        assert info.allowed is False
        assert dep_limiter.is_locked_out("c1") is True

    def test_lockout_expires(self, dep_limiter):
        dep_limiter._lockouts["c1"] = time.time() - 1
        assert dep_limiter.is_locked_out("c1") is False

    def test_reset(self, dep_limiter):
        dep_limiter.record_request("c1")
        dep_limiter._lockouts["c1"] = time.time() + 999
        dep_limiter.reset("c1")
        assert "c1" not in dep_limiter._minute_windows
        assert "c1" not in dep_limiter._hour_windows
        assert "c1" not in dep_limiter._lockouts

    def test_old_entries_cleaned(self, dep_limiter):
        old_ts = time.time() - 120
        dep_limiter._minute_windows["c1"] = [(old_ts, 1)]
        dep_limiter._hour_windows["c1"] = [(old_ts, 1)]
        cfg = DepRateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            lockout_threshold=100,
        )
        info = dep_limiter.check_rate_limit("c1", cfg)
        assert info.allowed is True
        # minute window entry should be cleaned
        assert "c1" not in dep_limiter._minute_windows

    def test_max_tracked_clients_eviction(self, dep_limiter):
        dep_limiter.MAX_TRACKED_CLIENTS = 5
        for i in range(7):
            dep_limiter.record_request(f"c_{i}")
        cfg = DepRateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=1000,
            lockout_threshold=100,
        )
        # Trigger eviction
        info = dep_limiter.check_rate_limit("c_new", cfg)
        assert info.allowed is True

    def test_compute_reset_seconds_empty(self, dep_limiter):
        result = dep_limiter._compute_reset_seconds([], 60, time.time())
        assert result == 60

    def test_compute_reset_seconds_with_entries(self, dep_limiter):
        now = time.time()
        entries = [(now - 10, 1)]
        result = dep_limiter._compute_reset_seconds(entries, 60, now)
        assert result == 51  # (now-10) + 60 - now + 1 = 51


# ============================================================================
# RateLimitInfo tests
# ============================================================================


class TestRateLimitInfo:
    def test_dataclass_fields(self):
        info = RateLimitInfo(
            allowed=True,
            retry_after=None,
            limit=60,
            remaining=59,
            reset_seconds=60,
        )
        assert info.allowed is True
        assert info.retry_after is None
        assert info.limit == 60


# ============================================================================
# Config presets (rate_limit.py)
# ============================================================================


class TestConfigPresets:
    def test_auth_config_stricter(self):
        assert AUTH_CONFIG.requests_per_minute < DEFAULT_CONFIG.requests_per_minute
        assert AUTH_CONFIG.lockout_threshold == 3
        assert AUTH_CONFIG.lockout_duration_seconds == 900

    def test_diagnosis_config(self):
        assert DIAGNOSIS_CONFIG.requests_per_minute == 3
        assert DIAGNOSIS_CONFIG.requests_per_hour == 20

    def test_search_config_more_permissive(self):
        assert SEARCH_CONFIG.requests_per_minute > AUTH_CONFIG.requests_per_minute


# ============================================================================
# get_client_key tests
# ============================================================================


class TestGetClientKey:
    def test_direct_ip(self):
        req = _make_request(ip="10.0.0.1")
        assert get_client_key(req) == "10.0.0.1"

    def test_forwarded_for_takes_last(self):
        req = _make_request(headers={"X-Forwarded-For": "a, b, c"})
        assert get_client_key(req) == "c"

    def test_no_client(self):
        req = _make_request()
        req.client = None
        req.headers = {}
        assert get_client_key(req) == "unknown"


# ============================================================================
# _set_rate_limit_headers tests
# ============================================================================


class TestSetRateLimitHeaders:
    def test_headers_set(self):
        resp = _make_response()
        info = RateLimitInfo(
            allowed=True, retry_after=None, limit=60, remaining=50, reset_seconds=45
        )
        _set_rate_limit_headers(resp, info)
        assert resp.headers["X-RateLimit-Limit"] == "60"
        assert resp.headers["X-RateLimit-Remaining"] == "50"
        assert resp.headers["X-RateLimit-Reset"] == "45"


# ============================================================================
# check_rate_limit dependency tests
# ============================================================================


class TestCheckRateLimitDependency:
    @pytest.mark.asyncio
    async def test_allowed_request(self):
        req = _make_request(ip="7.7.7.7")
        resp = _make_response()
        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            mock_rl.check_rate_limit.return_value = RateLimitInfo(
                allowed=True,
                retry_after=None,
                limit=60,
                remaining=59,
                reset_seconds=60,
            )
            mock_rl.record_request = MagicMock()
            await check_rate_limit(req, resp)
        mock_rl.record_request.assert_called_once()
        assert resp.headers["X-RateLimit-Remaining"] == "58"

    @pytest.mark.asyncio
    async def test_denied_request_raises_429(self):
        from fastapi import HTTPException

        req = _make_request(ip="7.7.7.7")
        resp = _make_response()
        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            mock_rl.check_rate_limit.return_value = RateLimitInfo(
                allowed=False,
                retry_after=30,
                limit=60,
                remaining=0,
                reset_seconds=30,
            )
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit(req, resp)
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_denied_without_retry_after(self):
        from fastapi import HTTPException

        req = _make_request(ip="8.8.8.8")
        resp = _make_response()
        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            mock_rl.check_rate_limit.return_value = RateLimitInfo(
                allowed=False,
                retry_after=None,
                limit=60,
                remaining=0,
                reset_seconds=30,
            )
            with pytest.raises(HTTPException) as exc_info:
                await check_rate_limit(req, resp)
            # Retry-After header should NOT be present when retry_after is None
            headers = exc_info.value.headers or {}
            assert "Retry-After" not in headers

    @pytest.mark.asyncio
    async def test_stores_info_on_request_state(self):
        req = _make_request(ip="1.1.1.1")
        resp = _make_response()
        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            mock_rl.check_rate_limit.return_value = RateLimitInfo(
                allowed=True,
                retry_after=None,
                limit=60,
                remaining=59,
                reset_seconds=60,
            )
            mock_rl.record_request = MagicMock()
            await check_rate_limit(req, resp)
        assert req.state.rate_limit_info.remaining == 58


# ============================================================================
# Convenience dependency wrappers
# ============================================================================


class TestConvenienceDependencies:
    @pytest.mark.asyncio
    async def test_auth_rate_limit(self):
        req = _make_request(ip="1.1.1.1")
        resp = _make_response()
        with patch("app.core.rate_limit.check_rate_limit", new_callable=AsyncMock) as mock_check:
            await check_auth_rate_limit(req, resp)
            mock_check.assert_awaited_once_with(req, resp, AUTH_CONFIG)

    @pytest.mark.asyncio
    async def test_diagnosis_rate_limit(self):
        req = _make_request(ip="1.1.1.1")
        resp = _make_response()
        with patch("app.core.rate_limit.check_rate_limit", new_callable=AsyncMock) as mock_check:
            await check_diagnosis_rate_limit(req, resp)
            mock_check.assert_awaited_once_with(req, resp, DIAGNOSIS_CONFIG)

    @pytest.mark.asyncio
    async def test_search_rate_limit(self):
        req = _make_request(ip="1.1.1.1")
        resp = _make_response()
        with patch("app.core.rate_limit.check_rate_limit", new_callable=AsyncMock) as mock_check:
            await check_search_rate_limit(req, resp)
            mock_check.assert_awaited_once_with(req, resp, SEARCH_CONFIG)


# ============================================================================
# reset_rate_limit tests
# ============================================================================


class TestResetRateLimit:
    def test_reset_delegates_to_limiter(self):
        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            reset_rate_limit("some_key")
            mock_rl.reset.assert_called_once_with("some_key")


# ============================================================================
# RateLimitMiddleware (rate_limit.py ASGI middleware) tests
# ============================================================================


class TestDepRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_non_http_scope_passthrough(self):
        app = AsyncMock()
        mw = DepRateLimitMiddleware(app)
        scope = {"type": "websocket"}
        await mw(scope, AsyncMock(), AsyncMock())
        app.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_matching_path_passthrough(self):
        app = AsyncMock()
        mw = DepRateLimitMiddleware(app, paths=["/api/v1/auth"])
        scope = {
            "type": "http",
            "path": "/api/v1/vehicles",
            "headers": [],
            "client": ("1.2.3.4", 0),
        }
        await mw(scope, AsyncMock(), AsyncMock())
        app.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_matching_path_allowed(self):
        app = AsyncMock()
        cfg = DepRateLimitConfig(
            requests_per_minute=100, requests_per_hour=1000, lockout_threshold=200
        )
        mw = DepRateLimitMiddleware(app, config=cfg, paths=["/api"])
        scope = {
            "type": "http",
            "path": "/api/test",
            "headers": [],
            "client": ("1.2.3.4", 0),
        }

        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            mock_rl.check_rate_limit.return_value = RateLimitInfo(
                allowed=True,
                retry_after=None,
                limit=100,
                remaining=99,
                reset_seconds=60,
            )
            mock_rl.record_request = MagicMock()
            await mw(scope, AsyncMock(), AsyncMock())
        mock_rl.record_request.assert_called_once()
        app.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_matching_path_denied(self):
        app = AsyncMock()
        cfg = DepRateLimitConfig(requests_per_minute=1, requests_per_hour=10, lockout_threshold=100)
        mw = DepRateLimitMiddleware(app, config=cfg, paths=["/api"])
        scope = {
            "type": "http",
            "path": "/api/test",
            "headers": [],
            "client": ("1.2.3.4", 0),
        }

        send = AsyncMock()
        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            mock_rl.check_rate_limit.return_value = RateLimitInfo(
                allowed=False,
                retry_after=30,
                limit=1,
                remaining=0,
                reset_seconds=30,
            )
            await mw(scope, AsyncMock(), send)

        # Should send 429 response
        calls = send.call_args_list
        assert calls[0][0][0]["status"] == 429
        app.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_x_forwarded_for_header(self):
        app = AsyncMock()
        cfg = DepRateLimitConfig(
            requests_per_minute=100, requests_per_hour=1000, lockout_threshold=200
        )
        mw = DepRateLimitMiddleware(app, config=cfg)
        scope = {
            "type": "http",
            "path": "/test",
            "headers": [(b"x-forwarded-for", b"10.0.0.1, 20.0.0.1")],
            "client": ("1.2.3.4", 0),
        }

        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            mock_rl.check_rate_limit.return_value = RateLimitInfo(
                allowed=True,
                retry_after=None,
                limit=100,
                remaining=99,
                reset_seconds=60,
            )
            mock_rl.record_request = MagicMock()
            await mw(scope, AsyncMock(), AsyncMock())
        # Should have extracted "20.0.0.1" (last IP)
        mock_rl.check_rate_limit.assert_called_once()
        call_key = mock_rl.check_rate_limit.call_args[0][0]
        assert call_key == "20.0.0.1"

    @pytest.mark.asyncio
    async def test_no_paths_applies_to_all(self):
        """When paths=None, middleware applies to all HTTP requests."""
        app = AsyncMock()
        cfg = DepRateLimitConfig(
            requests_per_minute=100, requests_per_hour=1000, lockout_threshold=200
        )
        mw = DepRateLimitMiddleware(app, config=cfg, paths=None)
        scope = {
            "type": "http",
            "path": "/anything",
            "headers": [],
            "client": ("1.2.3.4", 0),
        }

        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            mock_rl.check_rate_limit.return_value = RateLimitInfo(
                allowed=True,
                retry_after=None,
                limit=100,
                remaining=99,
                reset_seconds=60,
            )
            mock_rl.record_request = MagicMock()
            await mw(scope, AsyncMock(), AsyncMock())
        mock_rl.record_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_denied_without_retry_after_no_header(self):
        """429 response omits Retry-After header when retry_after is None."""
        app = AsyncMock()
        mw = DepRateLimitMiddleware(app)
        scope = {
            "type": "http",
            "path": "/test",
            "headers": [],
            "client": ("1.2.3.4", 0),
        }
        send = AsyncMock()
        with patch("app.core.rate_limit._rate_limiter") as mock_rl:
            mock_rl.check_rate_limit.return_value = RateLimitInfo(
                allowed=False,
                retry_after=None,
                limit=1,
                remaining=0,
                reset_seconds=30,
            )
            await mw(scope, AsyncMock(), send)

        response_start = send.call_args_list[0][0][0]
        header_names = [h[0] for h in response_start["headers"]]
        assert b"retry-after" not in header_names
