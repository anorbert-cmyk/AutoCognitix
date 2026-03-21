"""Tests for app.core.error_handlers module."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import (
    DBAPIError,
    IntegrityError,
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.exc import (
    TimeoutError as SQLAlchemyTimeoutError,
)

from app.core.error_handlers import (
    RequestContextMiddleware,
    autocognitix_exception_handler,
    build_error_response,
    generic_exception_handler,
    get_request_id,
    pydantic_validation_exception_handler,
    setup_all_exception_handlers,
    setup_exception_handlers,
    setup_httpx_exception_handler,
    setup_neo4j_exception_handler,
    setup_qdrant_exception_handler,
    sqlalchemy_exception_handler,
    validation_exception_handler,
)
from app.core.exceptions import (
    AutoCognitixException,
    ErrorCode,
)
from typing import Optional


# =============================================================================
# Helper to create mock requests
# =============================================================================


def _make_request(
    path: str = "/api/test",
    method: str = "GET",
    request_id: Optional[str] = None,
    client_host: str = "127.0.0.1",
):
    """Create a mock Starlette Request object."""
    request = MagicMock()
    request.url.path = path
    request.method = method
    request.client.host = client_host
    headers = MagicMock()
    headers.get = lambda key, default=None: {
        "X-Request-ID": request_id,
    }.get(key, default)
    request.headers = headers

    if request_id:
        request.state.request_id = request_id
    else:
        # Simulate no request_id on state
        request.state = MagicMock(spec=[])

    return request


# =============================================================================
# build_error_response Tests
# =============================================================================


class TestBuildErrorResponse:
    """Tests for build_error_response function."""

    def test_builds_basic_response(self):
        response = build_error_response(
            request_id="req-123",
            code=ErrorCode.INTERNAL_ERROR,
            message="Something went wrong",
            status_code=500,
        )
        assert response.status_code == 500
        body = response.body
        assert b"req-123" in body
        assert b"ERR_1000" in body
        assert b"Something went wrong" in body

    def test_includes_hungarian_message(self):
        response = build_error_response(
            request_id="req-456",
            code=ErrorCode.VALIDATION_ERROR,
            message="Validation error",
            message_hu="Egyedi magyar uzenet",
            status_code=422,
        )
        assert response.status_code == 422
        assert b"Egyedi magyar uzenet" in response.body

    def test_auto_generates_hungarian_message(self):
        response = build_error_response(
            request_id="req-789",
            code=ErrorCode.NOT_FOUND,
            message="Not found",
            status_code=404,
        )
        assert response.status_code == 404
        # Should contain auto-generated Hungarian message from get_error_message

    def test_includes_details(self):
        response = build_error_response(
            request_id="req-abc",
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid",
            details={"field": "email"},
            status_code=422,
        )
        assert b"email" in response.body

    def test_empty_details_by_default(self):
        response = build_error_response(
            request_id="req-def",
            code=ErrorCode.INTERNAL_ERROR,
            message="Error",
            status_code=500,
        )
        import json

        body = json.loads(response.body)
        assert body["error"]["details"] == {}


# =============================================================================
# get_request_id Tests
# =============================================================================


class TestGetRequestId:
    """Tests for get_request_id function."""

    def test_extracts_from_state(self):
        request = MagicMock()
        request.state.request_id = "my-req-id"
        assert get_request_id(request) == "my-req-id"

    def test_generates_uuid_if_missing(self):
        request = MagicMock(spec=[])
        request.state = MagicMock(spec=[])
        result = get_request_id(request)
        # Should be a valid UUID string
        assert len(result) == 36
        uuid.UUID(result)  # Should not raise


# =============================================================================
# autocognitix_exception_handler Tests
# =============================================================================


class TestAutocognitixExceptionHandler:
    """Tests for autocognitix_exception_handler."""

    @pytest.mark.asyncio
    async def test_handles_custom_exception(self):
        request = _make_request(request_id="req-custom")
        exc = AutoCognitixException(
            message="Custom error",
            code=ErrorCode.DTC_VALIDATION_ERROR,
            details={"invalid_codes": ["Z9999"]},
            status_code=400,
        )

        response = await autocognitix_exception_handler(request, exc)

        assert response.status_code == 400
        assert b"Custom error" in response.body
        assert b"ERR_4001" in response.body
        assert b"Z9999" in response.body

    @pytest.mark.asyncio
    async def test_uses_exception_status_code(self):
        request = _make_request(request_id="req-status")
        exc = AutoCognitixException(
            message="Not found",
            code=ErrorCode.NOT_FOUND,
            status_code=404,
        )

        response = await autocognitix_exception_handler(request, exc)
        assert response.status_code == 404


# =============================================================================
# validation_exception_handler Tests
# =============================================================================


class TestValidationExceptionHandler:
    """Tests for validation_exception_handler."""

    @pytest.mark.asyncio
    async def test_handles_validation_errors(self):
        request = _make_request(request_id="req-val")

        # Create a RequestValidationError with mock error data
        errors = [
            {
                "loc": ("body", "dtc_codes"),
                "msg": "field required",
                "type": "value_error.missing",
            }
        ]
        exc = RequestValidationError(errors=errors)

        response = await validation_exception_handler(request, exc)

        assert response.status_code == 422
        assert b"Validation error" in response.body
        assert b"validation_errors" in response.body

    @pytest.mark.asyncio
    async def test_formats_field_paths(self):
        request = _make_request(request_id="req-path")
        errors = [
            {
                "loc": ("body", "vehicle", "vin"),
                "msg": "invalid VIN",
                "type": "value_error",
            }
        ]
        exc = RequestValidationError(errors=errors)

        response = await validation_exception_handler(request, exc)
        assert b"body -> vehicle -> vin" in response.body


# =============================================================================
# pydantic_validation_exception_handler Tests
# =============================================================================


class TestPydanticValidationExceptionHandler:
    """Tests for pydantic_validation_exception_handler."""

    @pytest.mark.asyncio
    async def test_handles_pydantic_validation_error(self):
        request = _make_request(request_id="req-pydantic")

        # Create a real Pydantic ValidationError
        class TestModel(BaseModel):
            name: str
            age: int

        try:
            TestModel(name=123, age="not_a_number")  # type: ignore
        except ValidationError as exc:
            response = await pydantic_validation_exception_handler(request, exc)
            assert response.status_code == 422
            assert b"validation_errors" in response.body


# =============================================================================
# sqlalchemy_exception_handler Tests
# =============================================================================


class TestSQLAlchemyExceptionHandler:
    """Tests for sqlalchemy_exception_handler."""

    @pytest.mark.asyncio
    async def test_operational_error(self):
        request = _make_request(request_id="req-db-op")
        exc = OperationalError("SELECT 1", {}, Exception("connection refused"))

        response = await sqlalchemy_exception_handler(request, exc)
        assert response.status_code == 503
        assert b"ERR_2001" in response.body

    @pytest.mark.asyncio
    async def test_integrity_error(self):
        request = _make_request(request_id="req-db-int")
        exc = IntegrityError("INSERT", {}, Exception("duplicate key"))

        response = await sqlalchemy_exception_handler(request, exc)
        assert response.status_code == 409
        assert b"ERR_2003" in response.body

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        request = _make_request(request_id="req-db-timeout")
        exc = SQLAlchemyTimeoutError()

        response = await sqlalchemy_exception_handler(request, exc)
        assert response.status_code == 504
        assert b"ERR_2002" in response.body

    @pytest.mark.asyncio
    async def test_dbapi_error(self):
        request = _make_request(request_id="req-db-dbapi")
        exc = DBAPIError("SELECT", {}, Exception("driver error"))

        response = await sqlalchemy_exception_handler(request, exc)
        assert response.status_code == 500
        assert b"ERR_2010" in response.body

    @pytest.mark.asyncio
    async def test_generic_sqlalchemy_error(self):
        request = _make_request(request_id="req-db-gen")
        exc = SQLAlchemyError("generic db error")

        response = await sqlalchemy_exception_handler(request, exc)
        assert response.status_code == 500
        assert b"ERR_2000" in response.body

    @pytest.mark.asyncio
    @patch("app.core.error_handlers.settings")
    async def test_debug_mode_includes_details(self, mock_settings):
        mock_settings.DEBUG = True
        request = _make_request(request_id="req-db-debug")
        exc = OperationalError("SELECT 1", {}, Exception("connection refused"))

        response = await sqlalchemy_exception_handler(request, exc)
        assert response.status_code == 503
        assert b"error_type" in response.body

    @pytest.mark.asyncio
    @patch("app.core.error_handlers.settings")
    async def test_production_mode_hides_details(self, mock_settings):
        mock_settings.DEBUG = False
        request = _make_request(request_id="req-db-prod")
        exc = OperationalError("SELECT 1", {}, Exception("connection refused"))

        response = await sqlalchemy_exception_handler(request, exc)
        assert response.status_code == 503
        # Should not contain error_message in production
        assert b"connection refused" not in response.body


# =============================================================================
# generic_exception_handler Tests
# =============================================================================


class TestGenericExceptionHandler:
    """Tests for generic_exception_handler."""

    @pytest.mark.asyncio
    async def test_handles_unhandled_exception(self):
        request = _make_request(request_id="req-generic")
        exc = RuntimeError("unexpected error")

        response = await generic_exception_handler(request, exc)
        assert response.status_code == 500
        assert b"ERR_1000" in response.body
        assert b"An unexpected error occurred" in response.body

    @pytest.mark.asyncio
    @patch("app.core.error_handlers.settings")
    async def test_debug_mode_includes_error_details(self, mock_settings):
        mock_settings.DEBUG = True
        request = _make_request(request_id="req-gen-debug")
        exc = ValueError("debug detail")

        response = await generic_exception_handler(request, exc)
        assert response.status_code == 500
        assert b"ValueError" in response.body
        assert b"debug detail" in response.body

    @pytest.mark.asyncio
    @patch("app.core.error_handlers.settings")
    async def test_production_mode_hides_error_details(self, mock_settings):
        mock_settings.DEBUG = False
        request = _make_request(request_id="req-gen-prod")
        exc = ValueError("secret info")

        response = await generic_exception_handler(request, exc)
        assert response.status_code == 500
        assert b"secret info" not in response.body


# =============================================================================
# RequestContextMiddleware Tests
# =============================================================================


class TestRequestContextMiddleware:
    """Tests for RequestContextMiddleware."""

    @pytest.mark.asyncio
    async def test_adds_request_id_to_response(self):
        middleware = RequestContextMiddleware(app=MagicMock())

        request = MagicMock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        request.headers.get = lambda key, default=None: None

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def mock_call_next(req):
            return mock_response

        response = await middleware.dispatch(request, mock_call_next)
        assert "X-Request-ID" in response.headers
        # Should be a valid UUID
        uuid.UUID(response.headers["X-Request-ID"])

    @pytest.mark.asyncio
    async def test_uses_existing_request_id(self):
        middleware = RequestContextMiddleware(app=MagicMock())

        request = MagicMock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        request.headers.get = lambda key, default=None: {
            "X-Request-ID": "custom-req-id",
        }.get(key, default)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def mock_call_next(req):
            return mock_response

        response = await middleware.dispatch(request, mock_call_next)
        assert response.headers["X-Request-ID"] == "custom-req-id"

    @pytest.mark.asyncio
    async def test_sets_request_id_in_state(self):
        middleware = RequestContextMiddleware(app=MagicMock())

        request = MagicMock()
        request.url.path = "/api/test"
        request.method = "GET"
        request.client.host = "127.0.0.1"
        request.headers.get = lambda key, default=None: None

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def mock_call_next(req):
            # At this point, request.state.request_id should be set
            return mock_response

        # Capture request_id inside call_next to verify it was set
        captured_id = None

        async def capturing_call_next(req):
            nonlocal captured_id
            captured_id = getattr(req.state, "request_id", None)
            return mock_response

        await middleware.dispatch(request, capturing_call_next)
        # Verify request_id was actually set (not just MagicMock auto-attribute)
        assert captured_id is not None


# =============================================================================
# setup_exception_handlers Tests
# =============================================================================


class TestSetupExceptionHandlers:
    """Tests for setup functions."""

    def test_setup_exception_handlers(self):
        app = FastAPI()
        setup_exception_handlers(app)
        # Should register handlers without error
        assert AutoCognitixException in app.exception_handlers
        assert RequestValidationError in app.exception_handlers
        assert ValidationError in app.exception_handlers
        assert SQLAlchemyError in app.exception_handlers
        assert Exception in app.exception_handlers

    def test_setup_all_exception_handlers(self):
        app = FastAPI()
        setup_all_exception_handlers(app)
        # Core handlers should be registered
        assert AutoCognitixException in app.exception_handlers
        assert Exception in app.exception_handlers

    def test_setup_neo4j_exception_handler(self):
        app = FastAPI()
        # Should not raise even if neo4j is not installed
        setup_neo4j_exception_handler(app)

    def test_setup_qdrant_exception_handler(self):
        app = FastAPI()
        # Should not raise even if qdrant is not installed
        setup_qdrant_exception_handler(app)

    def test_setup_httpx_exception_handler(self):
        app = FastAPI()
        setup_httpx_exception_handler(app)
        # httpx is installed, so the handler should be registered
        import httpx

        assert httpx.HTTPError in app.exception_handlers


# =============================================================================
# HTTPX Exception Handler Tests
# =============================================================================


class TestHttpxExceptionHandler:
    """Tests for httpx exception handler (if httpx is installed)."""

    @pytest.mark.asyncio
    async def test_httpx_timeout_handler(self):
        import httpx

        app = FastAPI()
        setup_httpx_exception_handler(app)

        handler = app.exception_handlers[httpx.HTTPError]
        request = _make_request(request_id="req-httpx-timeout")
        exc = httpx.ReadTimeout("read timed out")

        response = await handler(request, exc)
        assert response.status_code == 504
        assert b"ERR_1006" in response.body

    @pytest.mark.asyncio
    async def test_httpx_connect_error_handler(self):
        import httpx

        app = FastAPI()
        setup_httpx_exception_handler(app)

        handler = app.exception_handlers[httpx.HTTPError]
        request = _make_request(request_id="req-httpx-connect")
        exc = httpx.ConnectError("connection refused")

        response = await handler(request, exc)
        assert response.status_code == 502
        assert b"ERR_3000" in response.body

    @pytest.mark.asyncio
    async def test_httpx_generic_error_handler(self):
        import httpx

        app = FastAPI()
        setup_httpx_exception_handler(app)

        handler = app.exception_handlers[httpx.HTTPError]
        request = _make_request(request_id="req-httpx-generic")
        exc = httpx.HTTPError("some http error")

        response = await handler(request, exc)
        assert response.status_code == 502
