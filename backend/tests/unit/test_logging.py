"""Tests for app.core.logging module."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.core.logging import (
    ErrorContext,
    LogLevel,
    PerformanceLogger,
    RequestLoggingMiddleware,
    SpanContext,
    StructuredJsonFormatter,
    create_child_span,
    correlation_id_var,
    critical,
    debug,
    enrich_error_context,
    error,
    get_current_trace_context,
    get_logger,
    info,
    inject_trace_headers,
    log_database_operation,
    log_event,
    log_external_api_call,
    log_with_context,
    parent_span_id_var,
    request_id_var,
    setup_logging,
    span_id_var,
    trace_id_var,
    user_id_var,
    warning,
)


# =============================================================================
# ErrorContext Tests
# =============================================================================


class TestErrorContext:
    """Tests for ErrorContext thread-safe storage."""

    def setup_method(self):
        ErrorContext.clear()

    def teardown_method(self):
        ErrorContext.clear()

    def test_set_and_get(self):
        ErrorContext.set("key1", "value1")
        assert ErrorContext.get("key1") == "value1"

    def test_get_missing_key_returns_default(self):
        assert ErrorContext.get("missing") is None
        assert ErrorContext.get("missing", "fallback") == "fallback"

    def test_clear(self):
        ErrorContext.set("key1", "value1")
        ErrorContext.clear()
        assert ErrorContext.get("key1") is None

    def test_get_all(self):
        ErrorContext.set("a", 1)
        ErrorContext.set("b", 2)
        result = ErrorContext.get_all()
        assert result == {"a": 1, "b": 2}

    def test_get_all_returns_copy(self):
        ErrorContext.set("a", 1)
        result = ErrorContext.get_all()
        result["c"] = 3
        assert ErrorContext.get("c") is None

    def test_update(self):
        ErrorContext.update({"x": 10, "y": 20})
        assert ErrorContext.get("x") == 10
        assert ErrorContext.get("y") == 20


# =============================================================================
# StructuredJsonFormatter Tests
# =============================================================================


class TestStructuredJsonFormatter:
    """Tests for StructuredJsonFormatter."""

    def setup_method(self):
        ErrorContext.clear()

    def teardown_method(self):
        ErrorContext.clear()
        # Reset context vars
        request_id_var.set(None)
        user_id_var.set(None)
        correlation_id_var.set(None)
        trace_id_var.set(None)
        span_id_var.set(None)
        parent_span_id_var.set(None)

    def _make_record(self, msg="test message", level=logging.INFO, exc_info=None):
        record = logging.LogRecord(
            name="test.logger",
            level=level,
            pathname="/test/file.py",
            lineno=42,
            msg=msg,
            args=(),
            exc_info=exc_info,
        )
        return record

    @patch("app.core.logging.settings")
    def test_add_fields_basic(self, mock_settings):
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"
        formatter = StructuredJsonFormatter()
        record = self._make_record()
        log_record = {}
        formatter.add_fields(log_record, record, {})

        assert "timestamp" in log_record
        assert "@timestamp" in log_record
        assert log_record["level"] == "INFO"
        assert log_record["severity"] == "info"
        assert log_record["level_num"] == logging.INFO
        assert log_record["logger"] == "test.logger"
        assert log_record["logger_path"] == ["test", "logger"]
        assert log_record["service"]["name"] == "TestApp"
        assert log_record["service"]["environment"] == "test"
        assert log_record["environment"] == "test"

    @patch("app.core.logging.settings")
    def test_add_fields_with_request_context(self, mock_settings):
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"
        request_id_var.set("req-123")
        user_id_var.set("user-456")
        correlation_id_var.set("corr-789")

        formatter = StructuredJsonFormatter()
        record = self._make_record()
        log_record = {}
        formatter.add_fields(log_record, record, {})

        assert log_record["request_id"] == "req-123"
        assert log_record["user_id"] == "user-456"
        assert log_record["correlation_id"] == "corr-789"

    @patch("app.core.logging.settings")
    def test_add_fields_without_request_context(self, mock_settings):
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"

        formatter = StructuredJsonFormatter()
        record = self._make_record()
        log_record = {}
        formatter.add_fields(log_record, record, {})

        assert "request_id" not in log_record
        assert "user_id" not in log_record
        assert "correlation_id" not in log_record

    @patch("app.core.logging.settings")
    def test_add_fields_with_trace_context(self, mock_settings):
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"
        trace_id_var.set("trace-abc")
        span_id_var.set("span-def")
        parent_span_id_var.set("parent-ghi")

        formatter = StructuredJsonFormatter()
        record = self._make_record()
        log_record = {}
        formatter.add_fields(log_record, record, {})

        assert log_record["trace"]["trace_id"] == "trace-abc"
        assert log_record["trace"]["span_id"] == "span-def"
        assert log_record["trace"]["parent_span_id"] == "parent-ghi"

    @patch("app.core.logging.settings")
    def test_add_fields_with_error_context(self, mock_settings):
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"
        ErrorContext.set("request_method", "GET")
        ErrorContext.set("request_path", "/test")

        formatter = StructuredJsonFormatter()
        record = self._make_record()
        log_record = {}
        formatter.add_fields(log_record, record, {})

        assert log_record["context"]["request_method"] == "GET"
        assert log_record["context"]["request_path"] == "/test"

    @patch("app.core.logging.settings")
    def test_add_fields_with_exception(self, mock_settings):
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"

        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        formatter = StructuredJsonFormatter()
        record = self._make_record(exc_info=exc_info)
        log_record = {}
        formatter.add_fields(log_record, record, {})

        assert log_record["error"]["type"] == "ValueError"
        assert log_record["error"]["message"] == "test error"
        assert log_record["error"]["module"] == "builtins"
        assert "stack_trace" in log_record["error"]
        assert "frames" in log_record["error"]

    @patch("app.core.logging.settings")
    def test_add_fields_with_chained_exception(self, mock_settings):
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"

        try:
            try:
                raise TypeError("original")
            except TypeError as orig:
                raise ValueError("chained") from orig
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        formatter = StructuredJsonFormatter()
        record = self._make_record(exc_info=exc_info)
        log_record = {}
        formatter.add_fields(log_record, record, {})

        assert log_record["error"]["cause"]["type"] == "TypeError"
        assert log_record["error"]["cause"]["message"] == "original"

    def test_extract_stack_frames_none(self):
        formatter = StructuredJsonFormatter()
        assert formatter._extract_stack_frames(None) == []

    def test_remove_none_values(self):
        formatter = StructuredJsonFormatter()
        d = {"a": 1, "b": None, "c": {"d": None, "e": 5}}
        formatter._remove_none_values(d)
        assert d == {"a": 1, "c": {"e": 5}}

    @patch("app.core.logging.settings")
    def test_host_info(self, mock_settings):
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"
        formatter = StructuredJsonFormatter()
        record = self._make_record()
        log_record = {}
        formatter.add_fields(log_record, record, {})

        assert "host" in log_record
        assert "name" in log_record["host"]
        assert "pid" in log_record["host"]

    @patch("app.core.logging.settings")
    def test_source_info(self, mock_settings):
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"
        formatter = StructuredJsonFormatter()
        record = self._make_record()
        log_record = {}
        formatter.add_fields(log_record, record, {})

        assert log_record["source"]["line"] == 42
        assert log_record["source"]["file"] == "/test/file.py"


# =============================================================================
# setup_logging Tests
# =============================================================================


class TestSetupLogging:
    """Tests for setup_logging function."""

    @patch("app.core.logging.settings")
    def test_setup_logging_json_format(self, mock_settings):
        mock_settings.LOG_LEVEL = "DEBUG"
        mock_settings.LOG_FORMAT = "json"
        mock_settings.SENTRY_DSN = None
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"

        setup_logging()

        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, StructuredJsonFormatter)

    @patch("app.core.logging.settings")
    def test_setup_logging_text_format(self, mock_settings):
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.LOG_FORMAT = "text"
        mock_settings.SENTRY_DSN = None
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"

        setup_logging()

        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, logging.Formatter)
        assert not isinstance(root.handlers[0].formatter, StructuredJsonFormatter)

    @patch("app.core.logging.settings")
    def test_setup_logging_configures_module_levels(self, mock_settings):
        mock_settings.LOG_LEVEL = "DEBUG"
        mock_settings.LOG_FORMAT = "text"
        mock_settings.SENTRY_DSN = None
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"

        setup_logging()

        assert logging.getLogger("uvicorn").level == logging.INFO
        assert logging.getLogger("uvicorn.access").level == logging.WARNING
        assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING

    @patch("app.core.logging.settings")
    def test_setup_logging_with_sentry_import_error(self, mock_settings):
        mock_settings.LOG_LEVEL = "INFO"
        mock_settings.LOG_FORMAT = "text"
        mock_settings.SENTRY_DSN = "https://fake@sentry.io/123"
        mock_settings.PROJECT_NAME = "TestApp"
        mock_settings.ENVIRONMENT = "test"

        with patch.dict("sys.modules", {"sentry_sdk": None}):
            # Should not raise even if sentry_sdk import fails
            setup_logging()


# =============================================================================
# get_logger and convenience functions Tests
# =============================================================================


class TestGetLogger:
    """Tests for get_logger and convenience logging functions."""

    def test_get_logger_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_log_event(self):
        logger = MagicMock()
        log_event(logger, logging.INFO, "test_event", "Test message", key="value")
        logger.log.assert_called_once_with(
            logging.INFO, "Test message", extra={"event": "test_event", "key": "value"}
        )

    def test_debug_convenience(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            debug("debug msg", extra_key="val")
            mock_logger.debug.assert_called_once_with("debug msg", extra={"extra_key": "val"})

    def test_info_convenience(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            info("info msg")
            mock_logger.info.assert_called_once_with("info msg", extra={})

    def test_warning_convenience(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            warning("warning msg")
            mock_logger.warning.assert_called_once_with("warning msg", extra={})

    def test_error_convenience(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            error("error msg", exc_info=True)
            mock_logger.error.assert_called_once_with("error msg", exc_info=True, extra={})

    def test_critical_convenience(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            critical("critical msg")
            mock_logger.critical.assert_called_once_with("critical msg", exc_info=True, extra={})

    def test_log_with_context(self):
        logger = MagicMock()
        ctx = {"key": "value"}
        log_with_context(logger, logging.WARNING, "msg", ctx, exc_info=True)
        logger.log.assert_called_once_with(logging.WARNING, "msg", extra=ctx, exc_info=True)


# =============================================================================
# log_database_operation Tests
# =============================================================================


class TestLogDatabaseOperation:
    """Tests for log_database_operation function."""

    def test_successful_fast_operation(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            log_database_operation("select", "users", 50.0, rows_affected=5)
            mock_logger.debug.assert_called_once()

    def test_slow_operation_warns(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            log_database_operation("select", "users", 1500.0, rows_affected=5)
            mock_logger.warning.assert_called_once()

    def test_failed_operation_errors(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            log_database_operation("insert", "users", 100.0, success=False, error="Duplicate key")
            mock_logger.error.assert_called_once()


# =============================================================================
# log_external_api_call Tests
# =============================================================================


class TestLogExternalApiCall:
    """Tests for log_external_api_call function."""

    def test_successful_call(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            log_external_api_call("nhtsa", "/decode", "GET", 200, 150.0)
            mock_logger.info.assert_called_once()

    def test_client_error_warns(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            log_external_api_call("nhtsa", "/decode", "GET", 404, 150.0)
            mock_logger.warning.assert_called_once()

    def test_failed_call_errors(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            log_external_api_call(
                "nhtsa", "/decode", "GET", 500, 150.0, success=False, error="Timeout"
            )
            mock_logger.error.assert_called_once()


# =============================================================================
# LogLevel Tests
# =============================================================================


class TestLogLevel:
    """Tests for LogLevel constants."""

    def test_log_levels(self):
        assert LogLevel.DEBUG == logging.DEBUG
        assert LogLevel.INFO == logging.INFO
        assert LogLevel.WARNING == logging.WARNING
        assert LogLevel.ERROR == logging.ERROR
        assert LogLevel.CRITICAL == logging.CRITICAL


# =============================================================================
# PerformanceLogger Tests
# =============================================================================


class TestPerformanceLogger:
    """Tests for PerformanceLogger context manager and decorator."""

    def test_context_manager_fast_operation(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            with PerformanceLogger("test_op"):
                pass
            mock_logger.debug.assert_called_once()

    def test_context_manager_slow_operation(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            with PerformanceLogger("test_op", warn_threshold_ms=0):
                pass
            mock_logger.warning.assert_called_once()

    def test_context_manager_critical_slow(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            with PerformanceLogger("test_op", warn_threshold_ms=0, error_threshold_ms=0):
                pass
            mock_logger.error.assert_called_once()

    def test_context_manager_with_exception(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            with pytest.raises(ValueError), PerformanceLogger("test_op"):
                raise ValueError("boom")
            mock_logger.error.assert_called_once()
            call_kwargs = mock_logger.error.call_args
            assert "failed" in call_kwargs[0][0].lower()

    def test_track_decorator_sync(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            @PerformanceLogger.track("sync_op")
            def my_func(x):
                return x * 2

            result = my_func(5)
            assert result == 10
            mock_logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_decorator_async(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            @PerformanceLogger.track("async_op")
            async def my_async_func(x):
                return x * 3

            result = await my_async_func(5)
            assert result == 15
            mock_logger.debug.assert_called_once()

    def test_extra_fields_passed(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            with PerformanceLogger("test_op", operation="custom_detail"):
                pass
            call_kwargs = mock_logger.debug.call_args
            assert call_kwargs[1]["extra"]["operation"] == "custom_detail"


# =============================================================================
# SpanContext Tests
# =============================================================================


class TestSpanContext:
    """Tests for SpanContext distributed tracing."""

    def setup_method(self):
        span_id_var.set(None)
        parent_span_id_var.set(None)
        trace_id_var.set(None)

    def teardown_method(self):
        span_id_var.set(None)
        parent_span_id_var.set(None)
        trace_id_var.set(None)

    def test_span_context_creates_new_span(self):
        span_id_var.set("original-span")
        with SpanContext("test_operation") as span:
            assert span.span_id is not None
            assert span.span_id != "original-span"
            assert span_id_var.get() == span.span_id
            assert parent_span_id_var.get() == "original-span"

        # Restores previous context
        assert span_id_var.get() == "original-span"

    def test_span_context_with_exception(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger

            with pytest.raises(RuntimeError), SpanContext("fail_op"):
                raise RuntimeError("span error")

            mock_logger.error.assert_called()

    def test_span_set_attribute(self):
        with SpanContext("test_op") as span:
            span.set_attribute("rows", 42)
            assert span.attributes["rows"] == 42

    def test_span_context_success_logs_debug(self):
        with patch("app.core.logging.get_logger") as mock_get:
            mock_logger = MagicMock()
            mock_get.return_value = mock_logger
            with SpanContext("success_op"):
                pass
            # Should log debug for span_start and span_end
            assert mock_logger.debug.call_count == 2


# =============================================================================
# Trace Context and Headers Tests
# =============================================================================


class TestTraceContext:
    """Tests for trace context functions."""

    def setup_method(self):
        request_id_var.set(None)
        correlation_id_var.set(None)
        trace_id_var.set(None)
        span_id_var.set(None)
        parent_span_id_var.set(None)

    def teardown_method(self):
        request_id_var.set(None)
        correlation_id_var.set(None)
        trace_id_var.set(None)
        span_id_var.set(None)
        parent_span_id_var.set(None)

    def test_create_child_span(self):
        span_id_var.set("parent-span")
        new_span = create_child_span()
        assert len(new_span) == 16
        assert span_id_var.get() == new_span
        assert parent_span_id_var.get() == "parent-span"

    def test_get_current_trace_context(self):
        request_id_var.set("req-1")
        trace_id_var.set("trace-1")
        span_id_var.set("span-1")
        parent_span_id_var.set("parent-1")
        correlation_id_var.set("corr-1")

        ctx = get_current_trace_context()
        assert ctx["request_id"] == "req-1"
        assert ctx["trace_id"] == "trace-1"
        assert ctx["span_id"] == "span-1"
        assert ctx["parent_span_id"] == "parent-1"
        assert ctx["correlation_id"] == "corr-1"

    def test_get_current_trace_context_empty(self):
        ctx = get_current_trace_context()
        assert ctx["request_id"] is None
        assert ctx["trace_id"] is None

    def test_inject_trace_headers(self):
        request_id_var.set("req-1")
        trace_id_var.set("trace-1")
        span_id_var.set("span-1")
        correlation_id_var.set("corr-1")

        headers = {}
        result = inject_trace_headers(headers)
        assert result["X-Request-ID"] == "req-1"
        assert result["X-Trace-ID"] == "trace-1"
        assert result["X-Span-ID"] == "span-1"
        assert result["X-Correlation-ID"] == "corr-1"
        assert "traceparent" in result
        assert "trace-1" in result["traceparent"]
        assert "span-1" in result["traceparent"]

    def test_inject_trace_headers_empty_context(self):
        headers = {"Existing": "header"}
        result = inject_trace_headers(headers)
        assert result["Existing"] == "header"
        assert "X-Request-ID" not in result
        assert "traceparent" not in result

    def test_enrich_error_context(self):
        ErrorContext.clear()
        enrich_error_context(user_email="test@example.com", action="login")
        assert ErrorContext.get("user_email") == "test@example.com"
        assert ErrorContext.get("action") == "login"
        ErrorContext.clear()


# =============================================================================
# RequestLoggingMiddleware Tests
# =============================================================================


class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware helper methods."""

    def test_is_private_ip_private(self):
        assert RequestLoggingMiddleware._is_private_ip("192.168.1.1") is True
        assert RequestLoggingMiddleware._is_private_ip("10.0.0.1") is True
        assert RequestLoggingMiddleware._is_private_ip("127.0.0.1") is True

    def test_is_private_ip_public(self):
        assert RequestLoggingMiddleware._is_private_ip("8.8.8.8") is False
        assert RequestLoggingMiddleware._is_private_ip("1.1.1.1") is False

    def test_is_private_ip_invalid(self):
        assert RequestLoggingMiddleware._is_private_ip("invalid") is False

    def test_get_client_ip_cf_header(self):
        middleware = RequestLoggingMiddleware(app=MagicMock())
        request = MagicMock()
        request.headers = {"CF-Connecting-IP": "203.0.113.1"}
        assert middleware._get_client_ip(request) == "203.0.113.1"

    def test_get_client_ip_forwarded_for(self):
        middleware = RequestLoggingMiddleware(app=MagicMock())
        request = MagicMock()
        headers_data = {
            "CF-Connecting-IP": None,
            "X-Forwarded-For": "203.0.113.1, 10.0.0.1",
            "X-Real-IP": None,
        }
        mock_headers = MagicMock()
        mock_headers.get = lambda key, default=None: headers_data.get(key, default)
        mock_headers.__contains__ = lambda self, key: key in headers_data
        request.headers = mock_headers
        assert middleware._get_client_ip(request) == "203.0.113.1"

    def test_get_client_ip_forwarded_for_all_private(self):
        middleware = RequestLoggingMiddleware(app=MagicMock())
        request = MagicMock()
        request.headers.get = lambda key, default=None: {
            "CF-Connecting-IP": None,
            "X-Forwarded-For": "192.168.1.1, 10.0.0.1",
            "X-Real-IP": None,
        }.get(key, default)
        assert middleware._get_client_ip(request) == "192.168.1.1"

    def test_get_client_ip_real_ip(self):
        middleware = RequestLoggingMiddleware(app=MagicMock())
        request = MagicMock()
        request.headers.get = lambda key, default=None: {
            "CF-Connecting-IP": None,
            "X-Forwarded-For": None,
            "X-Real-IP": "203.0.113.5",
        }.get(key, default)
        assert middleware._get_client_ip(request) == "203.0.113.5"

    def test_get_client_ip_direct_client(self):
        middleware = RequestLoggingMiddleware(app=MagicMock())
        request = MagicMock()
        request.headers.get = lambda key, default=None: None
        request.client.host = "127.0.0.1"
        assert middleware._get_client_ip(request) == "127.0.0.1"

    def test_get_client_ip_no_client(self):
        middleware = RequestLoggingMiddleware(app=MagicMock())
        request = MagicMock()
        request.headers.get = lambda key, default=None: None
        request.client = None
        assert middleware._get_client_ip(request) == "unknown"

    def test_excluded_paths(self):
        assert "/health" in RequestLoggingMiddleware.EXCLUDED_PATHS
        assert "/metrics" in RequestLoggingMiddleware.EXCLUDED_PATHS
