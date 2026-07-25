"""Tests for app.core.metrics module."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.metrics import (
    ACTIVE_SESSIONS,
    DB_CONNECTION_POOL,
    DB_QUERY_COUNT,
    DB_QUERY_ERRORS,
    DIAGNOSIS_REQUEST_COUNT,
    DTC_CODES_TOTAL,
    DTC_LOOKUP_COUNT,
    EMBEDDING_GENERATION_COUNT,
    ERROR_COUNT,
    EXCEPTION_COUNT,
    EXTERNAL_API_CALLS,
    EXTERNAL_API_ERRORS,
    LLM_REQUESTS,
    LLM_TOKENS,
    MetricsMiddleware,
    REQUEST_COUNT,
    USERS_TOTAL,
    VECTOR_SEARCH_COUNT,
    VEHICLE_DECODE_COUNT,
    VEHICLES_TOTAL,
    generate_metrics_response,
    get_metrics_middleware,
    get_metrics_summary,
    set_active_sessions,
    set_data_metrics,
    set_db_pool_metrics,
    track_auth_attempt,
    track_database_query,
    track_diagnosis_request,
    track_dtc_lookup,
    track_embedding_generation,
    track_error,
    track_exception,
    track_external_api_call,
    track_llm_request,
    track_request_complete,
    track_request_start,
    track_vector_search,
    track_vehicle_decode,
    update_system_metrics,
)


# =============================================================================
# Context Manager Tracker Tests
# =============================================================================


class TestTrackDatabaseQuery:
    """Tests for track_database_query context manager."""

    def test_successful_query_increments_count(self):
        before = DB_QUERY_COUNT.labels(
            database="postgres", operation="select", table="users"
        )._value.get()
        with track_database_query("postgres", "select", "users"):
            pass
        after = DB_QUERY_COUNT.labels(
            database="postgres", operation="select", table="users"
        )._value.get()
        assert after == before + 1

    def test_successful_query_records_latency(self):
        with track_database_query("postgres", "select", "users"):
            pass
        # Latency should have been observed (no error = success)

    def test_failed_query_increments_error_count(self):
        before = DB_QUERY_ERRORS.labels(
            database="postgres", operation="insert", error_type="ValueError"
        )._value.get()
        with pytest.raises(ValueError), track_database_query("postgres", "insert", "users"):
            raise ValueError("test error")
        after = DB_QUERY_ERRORS.labels(
            database="postgres", operation="insert", error_type="ValueError"
        )._value.get()
        assert after == before + 1

    def test_failed_query_still_increments_count(self):
        before = DB_QUERY_COUNT.labels(
            database="neo4j", operation="match", table="nodes"
        )._value.get()
        with pytest.raises(RuntimeError), track_database_query("neo4j", "match", "nodes"):
            raise RuntimeError("connection lost")
        after = DB_QUERY_COUNT.labels(
            database="neo4j", operation="match", table="nodes"
        )._value.get()
        assert after == before + 1

    def test_default_table_is_unknown(self):
        before = DB_QUERY_COUNT.labels(
            database="redis", operation="get", table="unknown"
        )._value.get()
        with track_database_query("redis", "get"):
            pass
        after = DB_QUERY_COUNT.labels(
            database="redis", operation="get", table="unknown"
        )._value.get()
        assert after == before + 1


class TestTrackEmbeddingGeneration:
    """Tests for track_embedding_generation context manager."""

    def test_successful_generation(self):
        before = EMBEDDING_GENERATION_COUNT.labels(model="hubert", status="success")._value.get()
        with track_embedding_generation("hubert", batch_size=5):
            pass
        after = EMBEDDING_GENERATION_COUNT.labels(model="hubert", status="success")._value.get()
        assert after == before + 1

    def test_failed_generation(self):
        before = EMBEDDING_GENERATION_COUNT.labels(model="hubert", status="error")._value.get()
        with pytest.raises(RuntimeError), track_embedding_generation("hubert", batch_size=10):
            raise RuntimeError("model error")
        after = EMBEDDING_GENERATION_COUNT.labels(model="hubert", status="error")._value.get()
        assert after == before + 1


class TestTrackVectorSearch:
    """Tests for track_vector_search context manager."""

    def test_successful_search(self):
        before = VECTOR_SEARCH_COUNT.labels(
            collection="dtc_embeddings", status="success"
        )._value.get()
        with track_vector_search("dtc_embeddings") as ctx:
            ctx["results_count"] = 15
        after = VECTOR_SEARCH_COUNT.labels(
            collection="dtc_embeddings", status="success"
        )._value.get()
        assert after == before + 1

    def test_failed_search(self):
        before = VECTOR_SEARCH_COUNT.labels(
            collection="dtc_embeddings", status="error"
        )._value.get()
        with pytest.raises(ConnectionError), track_vector_search("dtc_embeddings"):
            raise ConnectionError("qdrant down")
        after = VECTOR_SEARCH_COUNT.labels(collection="dtc_embeddings", status="error")._value.get()
        assert after == before + 1

    def test_context_default_results_count(self):
        with track_vector_search("test_collection") as ctx:
            assert ctx["results_count"] == 0


class TestTrackDiagnosisRequest:
    """Tests for track_diagnosis_request context manager."""

    def test_successful_diagnosis(self):
        before = DIAGNOSIS_REQUEST_COUNT.labels(status="success", language="hu")._value.get()
        with track_diagnosis_request("hu"):
            pass
        after = DIAGNOSIS_REQUEST_COUNT.labels(status="success", language="hu")._value.get()
        assert after == before + 1

    def test_failed_diagnosis(self):
        before = DIAGNOSIS_REQUEST_COUNT.labels(status="error", language="en")._value.get()
        with pytest.raises(ValueError), track_diagnosis_request("en"):
            raise ValueError("bad input")
        after = DIAGNOSIS_REQUEST_COUNT.labels(status="error", language="en")._value.get()
        assert after == before + 1

    def test_default_language_is_hu(self):
        before = DIAGNOSIS_REQUEST_COUNT.labels(status="success", language="hu")._value.get()
        with track_diagnosis_request():
            pass
        after = DIAGNOSIS_REQUEST_COUNT.labels(status="success", language="hu")._value.get()
        assert after == before + 1


class TestTrackExternalApiCall:
    """Tests for track_external_api_call context manager."""

    def test_successful_api_call(self):
        before = EXTERNAL_API_CALLS.labels(
            service="nhtsa", endpoint="/decode", status_code="200"
        )._value.get()
        with track_external_api_call("nhtsa", "/decode") as ctx:
            ctx["status_code"] = 200
        after = EXTERNAL_API_CALLS.labels(
            service="nhtsa", endpoint="/decode", status_code="200"
        )._value.get()
        assert after == before + 1

    def test_failed_api_call_records_error(self):
        before = EXTERNAL_API_ERRORS.labels(service="nhtsa", error_type="TimeoutError")._value.get()
        with pytest.raises(TimeoutError), track_external_api_call("nhtsa", "/decode"):
            raise TimeoutError("timed out")
        after = EXTERNAL_API_ERRORS.labels(service="nhtsa", error_type="TimeoutError")._value.get()
        assert after == before + 1

    def test_default_status_code_zero(self):
        with track_external_api_call("test_service") as ctx:
            assert ctx["status_code"] == 0


class TestTrackLlmRequest:
    """Tests for track_llm_request context manager."""

    def test_successful_llm_request(self):
        before = LLM_REQUESTS.labels(
            provider="anthropic", model="claude-3", status="success"
        )._value.get()
        with track_llm_request("anthropic", "claude-3") as ctx:
            ctx["input_tokens"] = 100
            ctx["output_tokens"] = 50
        after = LLM_REQUESTS.labels(
            provider="anthropic", model="claude-3", status="success"
        )._value.get()
        assert after == before + 1

    def test_failed_llm_request(self):
        before = LLM_REQUESTS.labels(provider="openai", model="gpt-4", status="error")._value.get()
        with pytest.raises(RuntimeError), track_llm_request("openai", "gpt-4"):
            raise RuntimeError("api error")
        after = LLM_REQUESTS.labels(provider="openai", model="gpt-4", status="error")._value.get()
        assert after == before + 1

    def test_token_counting(self):
        before_input = LLM_TOKENS.labels(
            provider="anthropic", model="claude-3-test", type="input"
        )._value.get()
        before_output = LLM_TOKENS.labels(
            provider="anthropic", model="claude-3-test", type="output"
        )._value.get()
        with track_llm_request("anthropic", "claude-3-test") as ctx:
            ctx["input_tokens"] = 200
            ctx["output_tokens"] = 100
        after_input = LLM_TOKENS.labels(
            provider="anthropic", model="claude-3-test", type="input"
        )._value.get()
        after_output = LLM_TOKENS.labels(
            provider="anthropic", model="claude-3-test", type="output"
        )._value.get()
        assert after_input == before_input + 200
        assert after_output == before_output + 100

    def test_no_tokens_if_zero(self):
        """Tokens should not be incremented if left at 0."""
        with track_llm_request("test_prov", "test_model"):
            pass
        # Should not raise, and tokens should remain at default


# =============================================================================
# Simple Tracking Function Tests
# =============================================================================


class TestTrackRequestFunctions:
    """Tests for track_request_start and track_request_complete."""

    def test_track_request_start(self):
        track_request_start("GET", "/api/test")
        # Should not raise

    def test_track_request_complete(self):
        # First increment so decrement doesn't go negative
        track_request_start("POST", "/api/data")
        before = REQUEST_COUNT.labels(
            method="POST", endpoint="/api/data", status_code="201"
        )._value.get()
        track_request_complete(
            method="POST",
            endpoint="/api/data",
            status_code=201,
            duration=0.5,
            request_size=1024,
            response_size=2048,
        )
        after = REQUEST_COUNT.labels(
            method="POST", endpoint="/api/data", status_code="201"
        )._value.get()
        assert after == before + 1

    def test_track_request_complete_without_sizes(self):
        track_request_start("GET", "/api/no-size")
        track_request_complete(
            method="GET",
            endpoint="/api/no-size",
            status_code=200,
            duration=0.1,
        )
        # Should not raise


class TestSimpleTrackers:
    """Tests for simple metric tracking functions."""

    def test_track_error(self):
        before = ERROR_COUNT.labels(error_type="ValueError", endpoint="/api/test")._value.get()
        track_error("ValueError", "/api/test")
        after = ERROR_COUNT.labels(error_type="ValueError", endpoint="/api/test")._value.get()
        assert after == before + 1

    def test_track_error_default_endpoint(self):
        before = ERROR_COUNT.labels(error_type="TypeError", endpoint="unknown")._value.get()
        track_error("TypeError")
        after = ERROR_COUNT.labels(error_type="TypeError", endpoint="unknown")._value.get()
        assert after == before + 1

    def test_track_exception(self):
        before = EXCEPTION_COUNT.labels(
            exception_type="RuntimeError", endpoint="/api/crash"
        )._value.get()
        track_exception("RuntimeError", "/api/crash")
        after = EXCEPTION_COUNT.labels(
            exception_type="RuntimeError", endpoint="/api/crash"
        )._value.get()
        assert after == before + 1

    def test_track_auth_attempt_success(self):
        track_auth_attempt("password", True)
        # Should not raise

    def test_track_auth_attempt_failure(self):
        track_auth_attempt("password", False)
        # Should not raise

    def test_set_active_sessions(self):
        set_active_sessions(42)
        assert ACTIVE_SESSIONS._value.get() == 42

    def test_track_dtc_lookup_found(self):
        before = DTC_LOOKUP_COUNT.labels(found="yes")._value.get()
        track_dtc_lookup(True)
        after = DTC_LOOKUP_COUNT.labels(found="yes")._value.get()
        assert after == before + 1

    def test_track_dtc_lookup_not_found(self):
        before = DTC_LOOKUP_COUNT.labels(found="no")._value.get()
        track_dtc_lookup(False)
        after = DTC_LOOKUP_COUNT.labels(found="no")._value.get()
        assert after == before + 1

    def test_track_vehicle_decode_success(self):
        before = VEHICLE_DECODE_COUNT.labels(status="success")._value.get()
        track_vehicle_decode(True)
        after = VEHICLE_DECODE_COUNT.labels(status="success")._value.get()
        assert after == before + 1

    def test_track_vehicle_decode_error(self):
        before = VEHICLE_DECODE_COUNT.labels(status="error")._value.get()
        track_vehicle_decode(False)
        after = VEHICLE_DECODE_COUNT.labels(status="error")._value.get()
        assert after == before + 1

    def test_set_data_metrics(self):
        set_data_metrics(dtc_count=63, vehicle_count=100, user_count=10)
        assert DTC_CODES_TOTAL._value.get() == 63
        assert VEHICLES_TOTAL._value.get() == 100
        assert USERS_TOTAL._value.get() == 10

    def test_set_db_pool_metrics(self):
        set_db_pool_metrics("postgres", active=5, idle=10, waiting=2)
        assert DB_CONNECTION_POOL.labels(database="postgres", state="active")._value.get() == 5
        assert DB_CONNECTION_POOL.labels(database="postgres", state="idle")._value.get() == 10
        assert DB_CONNECTION_POOL.labels(database="postgres", state="waiting")._value.get() == 2


# =============================================================================
# System Metrics Tests
# =============================================================================


class TestUpdateSystemMetrics:
    """Tests for update_system_metrics function."""

    @patch("app.core.metrics.psutil")
    def test_update_system_metrics(self, mock_psutil):
        mock_psutil.cpu_percent.return_value = 25.0

        mock_memory = MagicMock()
        mock_memory.percent = 60.0
        mock_memory.total = 16_000_000_000
        mock_memory.available = 8_000_000_000
        mock_memory.used = 8_000_000_000
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 5.0
        mock_mem_info = MagicMock()
        mock_mem_info.rss = 100_000_000
        mock_mem_info.vms = 200_000_000
        mock_process.memory_info.return_value = mock_mem_info
        mock_psutil.Process.return_value = mock_process

        update_system_metrics()

        mock_psutil.cpu_percent.assert_called_once()
        mock_psutil.virtual_memory.assert_called_once()

    @patch("app.core.metrics.psutil")
    def test_update_system_metrics_handles_exception(self, mock_psutil):
        mock_psutil.cpu_percent.side_effect = RuntimeError("no access")
        # Should not raise
        update_system_metrics()


# =============================================================================
# MetricsMiddleware Tests
# =============================================================================


class TestMetricsMiddleware:
    """Tests for MetricsMiddleware."""

    def test_normalize_endpoint_basic(self):
        middleware = MetricsMiddleware(app=MagicMock())
        assert middleware._normalize_endpoint("/api/v1/diagnosis") == "/api/v1/diagnosis"

    def test_normalize_endpoint_uuid(self):
        middleware = MetricsMiddleware(app=MagicMock())
        result = middleware._normalize_endpoint(
            "/api/v1/users/550e8400-e29b-41d4-a716-446655440000"
        )
        assert "{id}" in result

    def test_normalize_endpoint_digit_id(self):
        middleware = MetricsMiddleware(app=MagicMock())
        result = middleware._normalize_endpoint("/api/v1/items/12345")
        assert "{id}" in result

    def test_normalize_endpoint_vin(self):
        middleware = MetricsMiddleware(app=MagicMock())
        result = middleware._normalize_endpoint("/api/v1/vehicles/1HGBH41JXMN109186")
        assert "{vin}" in result

    def test_normalize_endpoint_dtc_code(self):
        middleware = MetricsMiddleware(app=MagicMock())
        result = middleware._normalize_endpoint("/api/v1/dtc/P0300")
        assert "{dtc_code}" in result

    @pytest.mark.parametrize("code", ["P26B7", "P090C", "P0A94", "B00A0", "p0a94"])
    def test_normalize_endpoint_hex_dtc_code(self, code: str):
        """Hex DTCs are real codes and must collapse into the {dtc_code} label."""
        middleware = MetricsMiddleware(app=MagicMock())
        assert middleware._normalize_endpoint(f"/api/v1/dtc/{code}") == "/api/v1/dtc/{dtc_code}"

    @pytest.mark.parametrize("segment", ["CHEVROLET", "PEACE", "P9324", "UA80E", "B-1234"])
    def test_normalize_endpoint_does_not_collapse_non_dtc_segments(self, segment: str):
        """Segments that merely start with P/B/C/U must keep their own label.

        The old guard (``part[0] in "PBCU"`` plus ``isalnum()``) folded make
        names and any hex-ish word into {dtc_code}, silently merging unrelated
        endpoints into one metric series.
        """
        middleware = MetricsMiddleware(app=MagicMock())
        result = middleware._normalize_endpoint(f"/api/v1/vehicles/{segment}/models")
        assert "{dtc_code}" not in result
        assert segment in result

    def test_normalize_endpoint_root(self):
        middleware = MetricsMiddleware(app=MagicMock())
        assert middleware._normalize_endpoint("/") == "/"

    def test_normalize_endpoint_empty_parts(self):
        middleware = MetricsMiddleware(app=MagicMock())
        result = middleware._normalize_endpoint("/api//v1/")
        assert result == "/api/v1"

    # -- label-cardinality catch-all on the /dtc routes -------------------
    #
    # /api/v1/dtc/{code} is unauthenticated. Recognising only real SAE codes is
    # correct, but falling through to the RAW segment meant every DTC-SHAPED
    # miss opened its own Prometheus series: [PBCU] x [4-9A-F] x hex^3 ~= 164k
    # reachable label values one sprayer can create. The endpoint answers 400;
    # the middleware runs first.

    @pytest.mark.parametrize("segment", ["P4AAA", "P5AAA", "UA80E", "PEACE", "P93AF", "P9324"])
    def test_a_dtc_shaped_miss_does_not_enter_the_label(self, segment: str):
        middleware = MetricsMiddleware(app=MagicMock())
        result = middleware._normalize_endpoint(f"/api/v1/dtc/{segment}")
        assert result == "/api/v1/dtc/{invalid_dtc}"
        assert segment not in result

    def test_sprayed_junk_collapses_to_one_series(self):
        """The property that actually caps the registry: N inputs, 1 label."""
        middleware = MetricsMiddleware(app=MagicMock())
        sprayed = {
            middleware._normalize_endpoint(f"/api/v1/dtc/P{h1}{h2}{h3}{h4}")
            for h1 in "456789ABCDEF"
            for h2 in "0123456789ABCDEF"
            for h3 in "0F"
            for h4 in "0F"
        }
        assert sprayed == {"/api/v1/dtc/{invalid_dtc}"}

    def test_the_invalid_label_is_distinct_from_the_valid_one(self):
        """{dtc_code} must keep meaning "a real code was requested"."""
        middleware = MetricsMiddleware(app=MagicMock())
        assert middleware._normalize_endpoint("/api/v1/dtc/P0300") == "/api/v1/dtc/{dtc_code}"
        assert middleware._normalize_endpoint("/api/v1/dtc/PEACE") == "/api/v1/dtc/{invalid_dtc}"

    def test_the_related_subpath_is_covered_too(self):
        middleware = MetricsMiddleware(app=MagicMock())
        assert (
            middleware._normalize_endpoint("/api/v1/dtc/PEACE/related")
            == "/api/v1/dtc/{invalid_dtc}/related"
        )

    @pytest.mark.parametrize("literal", ["search", "categories", "bulk"])
    def test_literal_sibling_routes_keep_their_own_label(self, literal: str):
        """Collapsing /dtc/search into {invalid_dtc} would erase a real route."""
        middleware = MetricsMiddleware(app=MagicMock())
        assert middleware._normalize_endpoint(f"/api/v1/dtc/{literal}") == f"/api/v1/dtc/{literal}"

    def test_the_catch_all_is_scoped_to_the_code_position(self):
        """A make named PACED is not the {code} parameter and keeps its label."""
        middleware = MetricsMiddleware(app=MagicMock())
        result = middleware._normalize_endpoint("/api/v1/vehicles/PACED/models")
        assert result == "/api/v1/vehicles/PACED/models"

    def test_a_numeric_segment_under_dtc_is_still_an_id(self):
        middleware = MetricsMiddleware(app=MagicMock())
        assert middleware._normalize_endpoint("/api/v1/dtc/12345") == "/api/v1/dtc/{id}"

    def test_dtc_static_subpaths_match_the_router(self):
        """Drift guard: a new literal route under /dtc must be registered here.

        Otherwise it silently starts reporting as {invalid_dtc} and disappears
        from the dashboards.
        """
        from app.api.v1.endpoints.dtc_codes import router
        from app.core.metrics_paths import DTC_STATIC_SUBPATHS

        literals = {
            route.path.strip("/").split("/")[0]  # type: ignore[attr-defined]
            for route in router.routes
            if not route.path.strip("/").split("/")[0].startswith("{")  # type: ignore[attr-defined]
        }
        literals.discard("")  # the POST "/" collection route
        assert literals == set(DTC_STATIC_SUBPATHS)

    def test_is_uuid_valid(self):
        assert MetricsMiddleware._is_uuid("550e8400-e29b-41d4-a716-446655440000") is True

    def test_is_uuid_invalid(self):
        assert MetricsMiddleware._is_uuid("not-a-uuid") is False
        assert MetricsMiddleware._is_uuid("") is False

    def test_excluded_endpoints(self):
        assert "/health" in MetricsMiddleware.EXCLUDED_ENDPOINTS
        assert "/metrics" in MetricsMiddleware.EXCLUDED_ENDPOINTS
        assert "/health/live" in MetricsMiddleware.EXCLUDED_ENDPOINTS
        assert "/health/ready" in MetricsMiddleware.EXCLUDED_ENDPOINTS

    @pytest.mark.asyncio
    async def test_dispatch_excluded_endpoint(self):
        mock_app = MagicMock()
        middleware = MetricsMiddleware(app=mock_app)

        request = MagicMock()
        request.url.path = "/health"
        request.method = "GET"

        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_call_next(req):
            return mock_response

        response = await middleware.dispatch(request, mock_call_next)
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_dispatch_normal_endpoint(self):
        mock_app = MagicMock()
        middleware = MetricsMiddleware(app=mock_app)

        request = MagicMock()
        request.url.path = "/api/v1/diagnosis"
        request.method = "POST"
        req_headers = MagicMock()
        req_headers.get = lambda key, default=None: {"Content-Length": "512"}.get(key, default)
        request.headers = req_headers

        mock_response = MagicMock()
        mock_response.status_code = 200
        resp_headers = MagicMock()
        resp_headers.get = lambda key, default=None: {"Content-Length": "1024"}.get(key, default)
        mock_response.headers = resp_headers

        async def mock_call_next(req):
            return mock_response

        response = await middleware.dispatch(request, mock_call_next)
        assert response == mock_response

    @pytest.mark.asyncio
    async def test_dispatch_handles_exception(self):
        mock_app = MagicMock()
        middleware = MetricsMiddleware(app=mock_app)

        request = MagicMock()
        request.url.path = "/api/v1/crash"
        request.method = "GET"
        req_headers = MagicMock()
        req_headers.get = lambda key, default=None: None
        request.headers = req_headers

        async def mock_call_next(req):
            raise RuntimeError("server error")

        with pytest.raises(RuntimeError, match="server error"):
            await middleware.dispatch(request, mock_call_next)


# =============================================================================
# Metrics Export Tests
# =============================================================================


class TestMetricsExport:
    """Tests for metrics export functions."""

    @pytest.mark.asyncio
    @patch("app.core.metrics.update_system_metrics")
    async def test_generate_metrics_response(self, mock_update):
        response = await generate_metrics_response()
        mock_update.assert_called_once()
        assert response.status_code == 200
        assert b"autocognitix" in response.body

    @patch("app.core.metrics.update_system_metrics")
    def test_get_metrics_summary(self, mock_update):
        summary = get_metrics_summary()
        mock_update.assert_called_once()
        assert "timestamp" in summary
        assert "service" in summary
        assert "environment" in summary
        assert "system" in summary
        assert "endpoints" in summary
        assert summary["endpoints"]["metrics"] == "/metrics"
        assert summary["endpoints"]["health"] == "/health"

    def test_get_metrics_middleware_returns_class(self):
        result = get_metrics_middleware()
        assert result is MetricsMiddleware
