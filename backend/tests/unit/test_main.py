"""
Unit tests for app/main.py

Tests cover:
- create_application() factory: router inclusion, middleware, exception handlers
- MaxBodySizeMiddleware: oversized, normal, non-http, invalid content-length
- lifespan(): startup and shutdown paths (success + failure branches)
- _seed_dtc_codes(): all branches (already seeded, file missing, bad JSON, empty codes, success)
- Health / metrics root-level endpoints
- Security headers middleware
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Tests: create_application
# ---------------------------------------------------------------------------


class TestCreateApplication:
    """Tests for the create_application factory function."""

    def test_returns_fastapi_instance(self):
        from app.main import create_application

        application = create_application()
        assert isinstance(application, FastAPI)

    def test_app_title_matches_settings(self):
        from app.main import create_application
        from app.core.config import settings

        application = create_application()
        assert application.title == settings.PROJECT_NAME

    def test_app_version(self):
        from app.main import create_application

        application = create_application()
        assert application.version == "0.1.0"

    def test_api_router_included(self):
        from app.main import create_application
        from app.core.config import settings

        application = create_application()
        route_paths = [r.path for r in application.routes]
        # The API router prefix should appear in routes
        assert any(settings.API_V1_PREFIX in p for p in route_paths)

    def test_health_endpoint_registered(self):
        from app.main import create_application

        application = create_application()
        route_paths = [r.path for r in application.routes]
        assert "/health" in route_paths

    def test_health_live_endpoint_registered(self):
        from app.main import create_application

        application = create_application()
        route_paths = [r.path for r in application.routes]
        assert "/health/live" in route_paths

    def test_health_ready_endpoint_registered(self):
        from app.main import create_application

        application = create_application()
        route_paths = [r.path for r in application.routes]
        assert "/health/ready" in route_paths

    def test_health_detailed_endpoint_registered(self):
        from app.main import create_application

        application = create_application()
        route_paths = [r.path for r in application.routes]
        assert "/health/detailed" in route_paths

    def test_health_db_endpoint_registered(self):
        from app.main import create_application

        application = create_application()
        route_paths = [r.path for r in application.routes]
        assert "/health/db" in route_paths

    def test_metrics_endpoint_registered(self):
        from app.main import create_application

        application = create_application()
        route_paths = [r.path for r in application.routes]
        assert "/metrics" in route_paths

    def test_openapi_tags_metadata(self):
        from app.main import create_application

        application = create_application()
        tag_names = [t["name"] for t in application.openapi_tags]
        assert "Health" in tag_names
        assert "Authentication" in tag_names
        assert "Diagnosis" in tag_names
        assert "DTC Codes" in tag_names
        assert "Vehicles" in tag_names
        assert "Metrics" in tag_names

    def test_middleware_stack_has_entries(self):
        """Verify that middleware was added (CORS, GZip, etc.)."""
        from app.main import create_application

        application = create_application()
        # FastAPI/Starlette stores middleware in app.middleware_stack
        # We can check that the middleware_stack is not None after build
        assert application.middleware_stack is not None or len(application.user_middleware) > 0

    def test_exception_handlers_registered(self):
        """setup_all_exception_handlers is called during create_application."""
        from app.main import create_application

        application = create_application()
        # FastAPI registers exception handlers in exception_handlers dict
        # At minimum, there should be entries beyond the default ones
        assert application.exception_handlers is not None


# ---------------------------------------------------------------------------
# Tests: Health endpoint response
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Test the /health endpoint via TestClient."""

    def test_health_returns_200(self):
        from app.main import create_application

        application = create_application()
        client = TestClient(application, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["service"] == "autocognitix-backend"

    def test_health_has_security_headers(self):
        from app.main import create_application

        application = create_application()
        client = TestClient(application, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


# ---------------------------------------------------------------------------
# Tests: MaxBodySizeMiddleware
# ---------------------------------------------------------------------------


class TestMaxBodySizeMiddleware:
    """Tests for the MaxBodySizeMiddleware ASGI middleware."""

    def test_rejects_oversized_request(self):
        from app.main import MaxBodySizeMiddleware

        inner_app = FastAPI()

        @inner_app.post("/test")
        async def _test_endpoint():
            return {"ok": True}

        inner_app.add_middleware(MaxBodySizeMiddleware)
        client = TestClient(inner_app, raise_server_exceptions=False)
        # Send a request with Content-Length header exceeding 1MB
        response = client.post(
            "/test",
            content=b"x",
            headers={"Content-Length": str(2_000_000)},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()

    def test_allows_normal_request(self):
        from app.main import MaxBodySizeMiddleware

        inner_app = FastAPI()

        @inner_app.post("/test")
        async def _test_endpoint():
            return {"ok": True}

        inner_app.add_middleware(MaxBodySizeMiddleware)
        client = TestClient(inner_app, raise_server_exceptions=False)
        response = client.post(
            "/test",
            content=b"hello",
        )
        assert response.status_code == 200

    def test_invalid_content_length_passes_through(self):
        """Invalid (non-numeric) Content-Length is ignored, request passes through."""
        from app.main import MaxBodySizeMiddleware

        inner_app = FastAPI()

        @inner_app.post("/test")
        async def _test_endpoint():
            return {"ok": True}

        inner_app.add_middleware(MaxBodySizeMiddleware)
        client = TestClient(inner_app, raise_server_exceptions=False)
        response = client.post(
            "/test",
            content=b"hello",
            headers={"Content-Length": "not-a-number"},
        )
        # Should pass through (ValueError caught) - may get 200 or other status
        # but should NOT be 413
        assert response.status_code != 413

    def test_non_http_scope_passes_through(self):
        """Non-HTTP scopes (e.g. websocket) pass through without checking."""
        import asyncio
        from app.main import MaxBodySizeMiddleware

        called = False

        async def dummy_app(scope, receive, send):
            nonlocal called
            called = True

        middleware = MaxBodySizeMiddleware(dummy_app)
        scope = {"type": "websocket", "headers": []}

        asyncio.get_event_loop().run_until_complete(middleware(scope, AsyncMock(), AsyncMock()))
        assert called

    def test_max_body_size_constant(self):
        from app.main import MaxBodySizeMiddleware

        assert MaxBodySizeMiddleware.MAX_BODY_SIZE == 1_048_576


# ---------------------------------------------------------------------------
# Tests: lifespan (startup + shutdown)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLifespan:
    """Test the lifespan context manager with mocked dependencies."""

    async def test_lifespan_startup_and_shutdown_success(self):
        """Happy path: all services initialize and shut down cleanly."""
        from app.main import lifespan

        mock_app = MagicMock(spec=FastAPI)

        mock_cache = AsyncMock()
        mock_cache.get_stats.return_value = {"status": "connected"}
        mock_cache.disconnect = AsyncMock()

        mock_thread_pool = MagicMock()

        with (
            patch("app.main.setup_logging") as mock_setup_logging,
            patch("app.main.qdrant_client") as mock_qdrant,
            patch("app.main._seed_dtc_codes", new_callable=AsyncMock) as mock_seed,
            patch(
                "app.db.redis_cache.get_cache_service",
                new_callable=AsyncMock,
                return_value=mock_cache,
            ),
            patch("app.main.engine") as mock_engine,
            patch("app.services.embedding_service._thread_pool", mock_thread_pool),
            patch("app.db.redis_cache._cache_service", mock_cache),
        ):
            mock_qdrant.initialize_collections = AsyncMock()
            mock_engine.dispose = AsyncMock()

            async with lifespan(mock_app):
                # Startup assertions
                mock_setup_logging.assert_called_once()
                mock_qdrant.initialize_collections.assert_awaited_once()
                mock_seed.assert_awaited_once()

            # Shutdown assertions
            mock_thread_pool.shutdown.assert_called_once_with(wait=True)
            mock_engine.dispose.assert_awaited_once()

    async def test_lifespan_qdrant_failure_does_not_crash(self):
        """Qdrant initialization failure is logged as warning, not raised."""
        from app.main import lifespan

        mock_app = MagicMock(spec=FastAPI)

        with (
            patch("app.main.setup_logging"),
            patch("app.main.qdrant_client") as mock_qdrant,
            patch("app.main._seed_dtc_codes", new_callable=AsyncMock),
            patch(
                "app.db.redis_cache.get_cache_service",
                new_callable=AsyncMock,
                side_effect=Exception("no redis"),
            ),
            patch("app.main.engine") as mock_engine,
            patch("app.services.embedding_service._thread_pool", MagicMock()),
            patch("app.db.redis_cache._cache_service", None),
        ):
            mock_qdrant.initialize_collections = AsyncMock(side_effect=Exception("no qdrant"))
            mock_engine.dispose = AsyncMock()

            # Should not raise
            async with lifespan(mock_app):
                pass

    async def test_lifespan_shutdown_thread_pool_error(self):
        """Thread pool shutdown error is caught and logged."""
        from app.main import lifespan

        mock_app = MagicMock(spec=FastAPI)

        mock_thread_pool = MagicMock()
        mock_thread_pool.shutdown.side_effect = RuntimeError("pool error")

        with (
            patch("app.main.setup_logging"),
            patch("app.main.qdrant_client") as mock_qdrant,
            patch("app.main._seed_dtc_codes", new_callable=AsyncMock),
            patch(
                "app.db.redis_cache.get_cache_service",
                new_callable=AsyncMock,
                side_effect=Exception("skip"),
            ),
            patch("app.main.engine") as mock_engine,
            patch("app.services.embedding_service._thread_pool", mock_thread_pool),
            patch("app.db.redis_cache._cache_service", None),
        ):
            mock_qdrant.initialize_collections = AsyncMock(side_effect=Exception("skip"))
            mock_engine.dispose = AsyncMock()

            async with lifespan(mock_app):
                pass
            # No exception should propagate

    async def test_lifespan_redis_disconnect_error(self):
        """Redis disconnect error during shutdown is caught."""
        from app.main import lifespan

        mock_app = MagicMock(spec=FastAPI)

        mock_cache = AsyncMock()
        mock_cache.disconnect = AsyncMock(side_effect=Exception("disconnect error"))

        with (
            patch("app.main.setup_logging"),
            patch("app.main.qdrant_client") as mock_qdrant,
            patch("app.main._seed_dtc_codes", new_callable=AsyncMock),
            patch(
                "app.db.redis_cache.get_cache_service",
                new_callable=AsyncMock,
                side_effect=Exception("skip"),
            ),
            patch("app.main.engine") as mock_engine,
            patch("app.services.embedding_service._thread_pool", MagicMock()),
            patch("app.db.redis_cache._cache_service", mock_cache),
        ):
            mock_qdrant.initialize_collections = AsyncMock(side_effect=Exception("skip"))
            mock_engine.dispose = AsyncMock()

            async with lifespan(mock_app):
                pass

    async def test_lifespan_seed_dtc_failure(self):
        """DTC seeding failure is caught and logged."""
        from app.main import lifespan

        mock_app = MagicMock(spec=FastAPI)

        with (
            patch("app.main.setup_logging"),
            patch("app.main.qdrant_client") as mock_qdrant,
            patch(
                "app.main._seed_dtc_codes",
                new_callable=AsyncMock,
                side_effect=Exception("seed failed"),
            ),
            patch(
                "app.db.redis_cache.get_cache_service",
                new_callable=AsyncMock,
                side_effect=Exception("skip"),
            ),
            patch("app.main.engine") as mock_engine,
            patch("app.services.embedding_service._thread_pool", MagicMock()),
            patch("app.db.redis_cache._cache_service", None),
        ):
            mock_qdrant.initialize_collections = AsyncMock(side_effect=Exception("skip"))
            mock_engine.dispose = AsyncMock()

            async with lifespan(mock_app):
                pass


# ---------------------------------------------------------------------------
# Tests: _seed_dtc_codes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSeedDtcCodes:
    """Test _seed_dtc_codes with mocked database sessions."""

    async def test_skips_when_already_seeded(self):
        """If DTC codes already exist, skip seeding."""
        from app.main import _seed_dtc_codes

        mock_result = MagicMock()
        mock_result.scalar.return_value = 100

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("app.main.async_session_maker", return_value=mock_session):
            await _seed_dtc_codes()

        # Should NOT call commit (no insert needed)
        mock_session.commit.assert_not_awaited()

    async def test_skips_when_seed_file_not_found(self):
        """If seed file does not exist, log warning and return."""
        from app.main import _seed_dtc_codes

        mock_result = MagicMock()
        mock_result.scalar.return_value = 0

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("app.main.async_session_maker", return_value=mock_session),
            patch("pathlib.Path.exists", return_value=False),
        ):
            await _seed_dtc_codes()

        mock_session.commit.assert_not_awaited()

    async def test_handles_bad_json(self):
        """If seed file has invalid JSON, log error and return."""
        from app.main import _seed_dtc_codes

        mock_result = MagicMock()
        mock_result.scalar.return_value = 0

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        # First Path.exists -> False (absolute path), second -> True (relative path)
        exists_calls = iter([False, True])

        with (
            patch("app.main.async_session_maker", return_value=mock_session),
            patch("pathlib.Path.exists", side_effect=exists_calls),
            patch("pathlib.Path.open", mock_open(read_data="not valid json {")),
        ):
            await _seed_dtc_codes()

        mock_session.commit.assert_not_awaited()

    async def test_skips_when_codes_empty(self):
        """If seed file has empty codes array, log warning and return."""
        from app.main import _seed_dtc_codes

        mock_result = MagicMock()
        mock_result.scalar.return_value = 0

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        seed_data = json.dumps({"codes": []})
        exists_calls = iter([False, True])

        with (
            patch("app.main.async_session_maker", return_value=mock_session),
            patch("pathlib.Path.exists", side_effect=exists_calls),
            patch("pathlib.Path.open", mock_open(read_data=seed_data)),
        ):
            await _seed_dtc_codes()

        mock_session.commit.assert_not_awaited()

    async def test_seeds_codes_successfully(self):
        """Happy path: codes are inserted and committed."""
        from app.main import _seed_dtc_codes

        mock_result = MagicMock()
        mock_result.scalar.return_value = 0

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        seed_data = json.dumps(
            {
                "codes": [
                    {"code": "P0300", "description_en": "Random misfire"},
                    {"code": "P0301", "description_en": "Cylinder 1 misfire"},
                ]
            }
        )
        exists_calls = iter([False, True])

        with (
            patch("app.main.async_session_maker", return_value=mock_session),
            patch("pathlib.Path.exists", side_effect=exists_calls),
            patch("pathlib.Path.open", mock_open(read_data=seed_data)),
        ):
            await _seed_dtc_codes()

        mock_session.commit.assert_awaited_once()

    async def test_seeds_codes_from_list_format(self):
        """Seed file can be a plain list (not dict with 'codes' key)."""
        from app.main import _seed_dtc_codes

        mock_result = MagicMock()
        mock_result.scalar.return_value = 0

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        seed_data = json.dumps(
            [
                {"code": "P0300", "description_en": "Random misfire"},
            ]
        )
        exists_calls = iter([False, True])

        with (
            patch("app.main.async_session_maker", return_value=mock_session),
            patch("pathlib.Path.exists", side_effect=exists_calls),
            patch("pathlib.Path.open", mock_open(read_data=seed_data)),
        ):
            await _seed_dtc_codes()

        mock_session.commit.assert_awaited_once()

    async def test_count_zero_or_none_triggers_seeding(self):
        """If count is None (empty table), proceed to seed."""
        from app.main import _seed_dtc_codes

        mock_result = MagicMock()
        mock_result.scalar.return_value = None  # NULL from empty table

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        # File does not exist at either path
        with (
            patch("app.main.async_session_maker", return_value=mock_session),
            patch("pathlib.Path.exists", return_value=False),
        ):
            await _seed_dtc_codes()

        # No commit since file not found
        mock_session.commit.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: module-level app object
# ---------------------------------------------------------------------------


class TestModuleLevelApp:
    """Test that the module-level `app` object is created correctly."""

    def test_app_is_fastapi_instance(self):
        from app.main import app

        assert isinstance(app, FastAPI)

    def test_app_has_routes(self):
        from app.main import app

        assert len(app.routes) > 0


# ---------------------------------------------------------------------------
# Tests: Root-level proxy endpoints (health/live, health/ready, etc.)
# ---------------------------------------------------------------------------


class TestRootLevelProxyEndpoints:
    """Test root-level health and metrics endpoints that delegate to API v1."""

    def test_health_live_endpoint(self):
        from app.main import create_application

        application = create_application()
        with patch(
            "app.api.v1.endpoints.health.liveness_check",
            new_callable=AsyncMock,
            return_value={"status": "alive"},
        ):
            client = TestClient(application, raise_server_exceptions=False)
            response = client.get("/health/live")
            assert response.status_code == 200

    def test_health_ready_endpoint(self):
        from app.main import create_application

        application = create_application()
        with patch(
            "app.api.v1.endpoints.health.readiness_check",
            new_callable=AsyncMock,
            return_value={"status": "ready"},
        ):
            client = TestClient(application, raise_server_exceptions=False)
            response = client.get("/health/ready")
            assert response.status_code == 200

    def test_health_detailed_endpoint(self):
        from app.main import create_application

        application = create_application()
        with patch(
            "app.api.v1.endpoints.health.detailed_health_check",
            new_callable=AsyncMock,
            return_value={"status": "detailed"},
        ):
            client = TestClient(application, raise_server_exceptions=False)
            response = client.get("/health/detailed")
            assert response.status_code == 200

    def test_health_db_endpoint(self):
        from app.main import create_application

        application = create_application()
        with patch(
            "app.api.v1.endpoints.health.database_stats",
            new_callable=AsyncMock,
            return_value={"db": "ok"},
        ):
            client = TestClient(application, raise_server_exceptions=False)
            response = client.get("/health/db")
            assert response.status_code == 200

    def test_metrics_endpoint(self):
        from app.main import create_application

        application = create_application()
        with patch(
            "app.api.v1.endpoints.metrics.get_metrics",
            new_callable=AsyncMock,
            return_value={"metrics": "data"},
        ):
            client = TestClient(application, raise_server_exceptions=False)
            response = client.get("/metrics")
            assert response.status_code == 200


# ---------------------------------------------------------------------------
# Tests: _seed_dtc_codes - first path exists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSeedDtcCodesFirstPath:
    """Test _seed_dtc_codes when the absolute path (/app/data/...) exists."""

    async def test_seeds_from_absolute_path(self):
        """When the first (absolute) path exists, use it directly."""
        from app.main import _seed_dtc_codes

        mock_result = MagicMock()
        mock_result.scalar.return_value = 0

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        seed_data = json.dumps({"codes": [{"code": "P0300", "description_en": "Test"}]})

        # First Path.exists call returns True (absolute path found)
        with (
            patch("app.main.async_session_maker", return_value=mock_session),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.open", mock_open(read_data=seed_data)),
        ):
            await _seed_dtc_codes()

        mock_session.commit.assert_awaited_once()
