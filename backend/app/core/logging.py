"""
Enhanced logging configuration for AutoCognitix.

Provides:
- Structured JSON logging with request correlation
- Request ID tracking across the request lifecycle
- User ID tracking for authenticated requests
- Performance metrics integration
- Configurable log levels per module
- Error context enrichment with stack traces
- Distributed tracing support
- Log aggregation compatibility (ELK, Loki)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime, UTC
from functools import wraps
from typing import Any, Optional, TypeVar, TYPE_CHECKING

from pythonjsonlogger import jsonlogger
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

if TYPE_CHECKING:
    from starlette.responses import Response
    from starlette.requests import Request

# Context variables for request correlation and distributed tracing
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
parent_span_id_var: ContextVar[Optional[str]] = ContextVar("parent_span_id", default=None)

# Type variable for generic function decorator
F = TypeVar("F", bound=Callable[..., Any])


# Error context storage for enrichment
class ErrorContext:
    """Thread-safe storage for error context enrichment."""

    _context: dict[str, Any] = {}

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a context value for error enrichment."""
        cls._context[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return cls._context.get(key, default)

    @classmethod
    def clear(cls) -> None:
        """Clear all context values."""
        cls._context.clear()

    @classmethod
    def get_all(cls) -> dict[str, Any]:
        """Get all context values."""
        return cls._context.copy()

    @classmethod
    def update(cls, data: dict[str, Any]) -> None:
        """Update context with multiple values."""
        cls._context.update(data)


class StructuredJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter with comprehensive context fields.

    Adds:
    - Timestamp in ISO format (RFC 3339 compliant)
    - Log level with severity number
    - Logger name and hierarchical path
    - Service name, version, and environment
    - Request ID, user ID, and correlation IDs from context
    - Distributed tracing fields (trace_id, span_id)
    - Exception info with full stack trace when present
    - Error context enrichment data
    - Host and process information
    - Custom fields from log record
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hostname = os.uname().nodename if hasattr(os, 'uname') else "unknown"
        self._pid = os.getpid()

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Timestamp in RFC 3339 format with timezone
        log_record["timestamp"] = datetime.now(UTC).isoformat()
        log_record["@timestamp"] = log_record["timestamp"]  # ELK compatibility

        # Log level with severity
        log_record["level"] = record.levelname
        log_record["severity"] = record.levelname.lower()
        log_record["level_num"] = record.levelno

        # Logger hierarchy
        log_record["logger"] = record.name
        log_record["logger_path"] = record.name.split(".")

        # Service metadata
        log_record["service"] = {
            "name": settings.PROJECT_NAME,
            "version": "0.1.0",
            "environment": settings.ENVIRONMENT,
        }

        # Host and process info
        log_record["host"] = {
            "name": self._hostname,
            "pid": self._pid,
        }

        # Source location info
        log_record["source"] = {
            "module": record.module,
            "function": record.funcName,
            "file": record.pathname,
            "line": record.lineno,
        }

        # Request correlation from context variables
        request_id = request_id_var.get()
        if request_id:
            log_record["request_id"] = request_id

        user_id = user_id_var.get()
        if user_id:
            log_record["user_id"] = user_id

        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_record["correlation_id"] = correlation_id

        # Distributed tracing fields
        trace_id = trace_id_var.get()
        span_id = span_id_var.get()
        parent_span_id = parent_span_id_var.get()

        if trace_id or span_id:
            log_record["trace"] = {}
            if trace_id:
                log_record["trace"]["trace_id"] = trace_id
            if span_id:
                log_record["trace"]["span_id"] = span_id
            if parent_span_id:
                log_record["trace"]["parent_span_id"] = parent_span_id

        # Error context enrichment
        error_context = ErrorContext.get_all()
        if error_context:
            log_record["context"] = error_context

        # Exception info with enhanced details
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            log_record["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "module": exc_type.__module__,
                "stack_trace": self.formatException(record.exc_info),
                "frames": self._extract_stack_frames(exc_tb),
            }

            # Include exception chain for chained exceptions
            if exc_value.__cause__:
                log_record["error"]["cause"] = {
                    "type": type(exc_value.__cause__).__name__,
                    "message": str(exc_value.__cause__),
                }
            elif exc_value.__context__ and not exc_value.__suppress_context__:
                log_record["error"]["context_exception"] = {
                    "type": type(exc_value.__context__).__name__,
                    "message": str(exc_value.__context__),
                }

        # Flatten for backward compatibility
        log_record["environment"] = settings.ENVIRONMENT

        # Remove None values for cleaner output
        self._remove_none_values(log_record)

    def _extract_stack_frames(self, tb, limit: int = 10) -> list[dict[str, Any]]:
        """Extract structured stack frame information."""
        frames = []
        if tb is None:
            return frames

        for frame_info in traceback.extract_tb(tb, limit=limit):
            frames.append({
                "file": frame_info.filename,
                "line": frame_info.lineno,
                "function": frame_info.name,
                "code": frame_info.line,
            })
        return frames

    def _remove_none_values(self, d: dict[str, Any]) -> None:
        """Recursively remove None values from dictionary."""
        keys_to_remove = []
        for key, value in d.items():
            if value is None:
                keys_to_remove.append(key)
            elif isinstance(value, dict):
                self._remove_none_values(value)
        for key in keys_to_remove:
            del d[key]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request logging and correlation.

    Features:
    - Generates unique request ID for each request
    - Extracts user ID from authenticated requests
    - Logs request start and completion with timing
    - Propagates request ID to response headers
    - Distributed tracing support (trace_id, span_id)
    - Error context enrichment
    - Performance timing with breakdown
    """

    # Paths to exclude from detailed logging (high-frequency health checks)
    EXCLUDED_PATHS = {"/health", "/health/live", "/health/ready", "/metrics"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        correlation_id = request.headers.get("X-Correlation-ID")

        # Distributed tracing headers (W3C Trace Context compatible)
        trace_id = request.headers.get("X-Trace-ID") or request.headers.get("traceparent", "").split("-")[1] if "-" in request.headers.get("traceparent", "") else str(uuid.uuid4()).replace("-", "")
        span_id = str(uuid.uuid4())[:16]
        parent_span_id = request.headers.get("X-Span-ID") or request.headers.get("X-Parent-Span-ID")

        # Set context variables
        request_id_token = request_id_var.set(request_id)
        correlation_token = correlation_id_var.set(correlation_id)
        trace_id_token = trace_id_var.set(trace_id)
        span_id_token = span_id_var.set(span_id)
        parent_span_id_token = parent_span_id_var.set(parent_span_id)

        # Extract user ID from request state if authenticated
        user_id = None
        if hasattr(request.state, "user") and request.state.user:
            user_id = str(getattr(request.state.user, "id", None))
        user_id_token = user_id_var.set(user_id)

        # Set error context for enrichment
        ErrorContext.clear()
        ErrorContext.update({
            "request_method": request.method,
            "request_path": request.url.path,
            "request_query": str(request.query_params) if request.query_params else None,
            "client_ip": self._get_client_ip(request),
        })

        logger = get_logger("request")

        # Check if this path should be logged in detail
        should_log_detailed = request.url.path not in self.EXCLUDED_PATHS

        # Log request start
        start_time = time.time()

        if should_log_detailed:
            logger.info(
                "Request started",
                extra={
                    "event": "request_start",
                    "http": {
                        "method": request.method,
                        "path": request.url.path,
                        "query": str(request.query_params) if request.query_params else None,
                        "scheme": request.url.scheme,
                        "host": request.url.hostname,
                    },
                    "client": {
                        "ip": self._get_client_ip(request),
                        "user_agent": request.headers.get("User-Agent"),
                        "referer": request.headers.get("Referer"),
                    },
                    "trace": {
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_span_id": parent_span_id,
                    },
                }
            )

        try:
            response = await call_next(request)

            # Calculate request duration
            duration_ms = (time.time() - start_time) * 1000
            duration_seconds = duration_ms / 1000

            # Determine log level based on status code and duration
            if response.status_code >= 500:
                log_level = logging.ERROR
            elif response.status_code >= 400 or duration_ms > 5000:
                log_level = logging.WARNING
            else:
                log_level = logging.INFO

            # Log request completion
            if should_log_detailed:
                logger.log(
                    log_level,
                    f"Request completed: {request.method} {request.url.path} - {response.status_code} ({duration_ms:.2f}ms)",
                    extra={
                        "event": "request_complete",
                        "http": {
                            "method": request.method,
                            "path": request.url.path,
                            "status_code": response.status_code,
                            "status_class": f"{response.status_code // 100}xx",
                        },
                        "timing": {
                            "duration_ms": round(duration_ms, 2),
                            "duration_seconds": round(duration_seconds, 4),
                        },
                        "response": {
                            "size_bytes": int(response.headers.get("Content-Length", 0)) if response.headers.get("Content-Length") else None,
                            "content_type": response.headers.get("Content-Type"),
                        },
                        "trace": {
                            "trace_id": trace_id,
                            "span_id": span_id,
                        },
                    }
                )

            # Add tracing headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Span-ID"] = span_id
            if correlation_id:
                response.headers["X-Correlation-ID"] = correlation_id

            # Add timing header
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Enrich error context
            ErrorContext.update({
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_ms": round(duration_ms, 2),
            })

            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "event": "request_error",
                    "http": {
                        "method": request.method,
                        "path": request.url.path,
                    },
                    "timing": {
                        "duration_ms": round(duration_ms, 2),
                    },
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                    },
                    "trace": {
                        "trace_id": trace_id,
                        "span_id": span_id,
                    },
                },
                exc_info=True,
            )
            raise

        finally:
            # Reset context variables
            request_id_var.reset(request_id_token)
            correlation_id_var.reset(correlation_token)
            user_id_var.reset(user_id_token)
            trace_id_var.reset(trace_id_token)
            span_id_var.reset(span_id_token)
            parent_span_id_var.reset(parent_span_id_token)
            ErrorContext.clear()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, considering proxy headers."""
        # Check Cloudflare header first
        cf_connecting_ip = request.headers.get("CF-Connecting-IP")
        if cf_connecting_ip:
            return cf_connecting_ip

        # Check X-Forwarded-For (take first non-private IP)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ips = [ip.strip() for ip in forwarded.split(",")]
            for ip in ips:
                if not self._is_private_ip(ip):
                    return ip
            return ips[0]  # Return first if all are private

        # Check X-Real-IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client
        if request.client:
            return request.client.host
        return "unknown"

    @staticmethod
    def _is_private_ip(ip: str) -> bool:
        """Check if IP address is private."""
        try:
            import ipaddress
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except ValueError:
            return False


class PerformanceLogger:
    """
    Context manager and decorator for logging performance metrics.

    Usage as context manager:
        with PerformanceLogger("database_query", operation="select_users"):
            result = await db.execute(query)

    Usage as decorator:
        @PerformanceLogger.track("embedding_generation")
        async def generate_embedding(text: str):
            ...
    """

    def __init__(
        self,
        operation_name: str,
        logger_name: str = "performance",
        warn_threshold_ms: float = 1000.0,
        error_threshold_ms: float = 5000.0,
        **extra_fields: Any,
    ):
        self.operation_name = operation_name
        self.logger = get_logger(logger_name)
        self.warn_threshold_ms = warn_threshold_ms
        self.error_threshold_ms = error_threshold_ms
        self.extra_fields = extra_fields
        self.start_time: float | None = None

    def __enter__(self) -> PerformanceLogger:
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0

        log_data = {
            "event": "performance_metric",
            "operation": self.operation_name,
            "duration_ms": round(duration_ms, 2),
            "success": exc_type is None,
            **self.extra_fields,
        }

        if exc_type:
            log_data["error_type"] = exc_type.__name__
            log_data["error_message"] = str(exc_val)
            self.logger.error(f"Operation failed: {self.operation_name}", extra=log_data)
        elif duration_ms >= self.error_threshold_ms:
            self.logger.error(f"Operation critically slow: {self.operation_name}", extra=log_data)
        elif duration_ms >= self.warn_threshold_ms:
            self.logger.warning(f"Operation slow: {self.operation_name}", extra=log_data)
        else:
            self.logger.debug(f"Operation completed: {self.operation_name}", extra=log_data)

    @classmethod
    def track(
        cls,
        operation_name: str,
        warn_threshold_ms: float = 1000.0,
        error_threshold_ms: float = 5000.0,
    ) -> Callable[[F], F]:
        """Decorator for tracking function performance."""
        def decorator(func: F) -> F:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with cls(operation_name, warn_threshold_ms=warn_threshold_ms, error_threshold_ms=error_threshold_ms):
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with cls(operation_name, warn_threshold_ms=warn_threshold_ms, error_threshold_ms=error_threshold_ms):
                    return func(*args, **kwargs)

            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator


class LogLevel:
    """Log level constants for convenience."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Logger configuration by module
LOGGER_CONFIG: dict[str, int] = {
    "uvicorn": logging.INFO,
    "uvicorn.access": logging.WARNING,
    "uvicorn.error": logging.ERROR,
    "sqlalchemy.engine": logging.WARNING,
    "sqlalchemy.pool": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "neo4j": logging.WARNING,
    "qdrant_client": logging.WARNING,
    "transformers": logging.WARNING,
    "sentence_transformers": logging.WARNING,
    "torch": logging.WARNING,
    "langchain": logging.INFO,
}


def setup_logging() -> None:
    """
    Configure application logging with structured JSON output.

    Features:
    - JSON format for production, human-readable for development
    - Configurable log levels per module
    - Sentry integration when configured
    - Request correlation support
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if settings.LOG_FORMAT == "json":
        # JSON format for production
        formatter = StructuredJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set specific logger levels from config
    for logger_name, level in LOGGER_CONFIG.items():
        logging.getLogger(logger_name).setLevel(level)

    # Initialize Sentry if DSN is provided
    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.logging import LoggingIntegration
            from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

            # Configure Sentry logging integration
            sentry_logging = LoggingIntegration(
                level=logging.INFO,        # Capture info and above as breadcrumbs
                event_level=logging.ERROR  # Send errors and above as events
            )

            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.ENVIRONMENT,
                release="autocognitix@0.1.0",
                integrations=[
                    FastApiIntegration(),
                    SqlalchemyIntegration(),
                    sentry_logging,
                ],
                traces_sample_rate=0.1 if settings.ENVIRONMENT == "production" else 1.0,
                profiles_sample_rate=0.1 if settings.ENVIRONMENT == "production" else 0.0,
                attach_stacktrace=True,
                send_default_pii=False,  # Don't send PII by default
            )
            logging.info("Sentry SDK initialized successfully")
        except ImportError:
            logging.warning("Sentry SDK not installed, skipping initialization")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    The logger will inherit the root logger's configuration
    and include request correlation fields in log output.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    **extra_fields: Any,
) -> None:
    """
    Log a structured event with additional context.

    Args:
        logger: Logger instance to use
        level: Log level (use LogLevel constants)
        event: Event type identifier
        message: Human-readable message
        **extra_fields: Additional fields to include in log
    """
    logger.log(
        level,
        message,
        extra={"event": event, **extra_fields}
    )


def log_database_operation(
    operation: str,
    table: str,
    duration_ms: float,
    rows_affected: int = 0,
    success: bool = True,
    error: str | None = None,
) -> None:
    """
    Log a database operation with standard fields.

    Args:
        operation: Type of operation (select, insert, update, delete)
        table: Table name
        duration_ms: Operation duration in milliseconds
        rows_affected: Number of rows affected
        success: Whether operation succeeded
        error: Error message if failed
    """
    logger = get_logger("database")

    extra = {
        "event": "database_operation",
        "operation": operation,
        "table": table,
        "duration_ms": round(duration_ms, 2),
        "rows_affected": rows_affected,
        "success": success,
    }

    if error:
        extra["error"] = error
        logger.error(f"Database {operation} on {table} failed", extra=extra)
    elif duration_ms > 1000:
        logger.warning(f"Slow database {operation} on {table}", extra=extra)
    else:
        logger.debug(f"Database {operation} on {table} completed", extra=extra)


def log_external_api_call(
    service: str,
    endpoint: str,
    method: str,
    status_code: int,
    duration_ms: float,
    success: bool = True,
    error: str | None = None,
) -> None:
    """
    Log an external API call with standard fields.

    Args:
        service: Name of external service (e.g., "nhtsa", "qdrant")
        endpoint: API endpoint called
        method: HTTP method
        status_code: Response status code
        duration_ms: Call duration in milliseconds
        success: Whether call succeeded
        error: Error message if failed
    """
    logger = get_logger("external_api")

    extra = {
        "event": "external_api_call",
        "service": service,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "success": success,
    }

    if error:
        extra["error"] = error
        logger.error(f"External API call to {service} failed", extra=extra)
    elif status_code >= 400:
        logger.warning(f"External API call to {service} returned {status_code}", extra=extra)
    else:
        logger.info(f"External API call to {service} completed", extra=extra)


# Convenience functions for common log levels
def debug(message: str, **extra: Any) -> None:
    """Log a debug message with optional extra fields."""
    get_logger("app").debug(message, extra=extra)


def info(message: str, **extra: Any) -> None:
    """Log an info message with optional extra fields."""
    get_logger("app").info(message, extra=extra)


def warning(message: str, **extra: Any) -> None:
    """Log a warning message with optional extra fields."""
    get_logger("app").warning(message, extra=extra)


def error(message: str, exc_info: bool = False, **extra: Any) -> None:
    """Log an error message with optional exception info."""
    get_logger("app").error(message, exc_info=exc_info, extra=extra)


def critical(message: str, exc_info: bool = True, **extra: Any) -> None:
    """Log a critical message with exception info by default."""
    get_logger("app").critical(message, exc_info=exc_info, extra=extra)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    context: dict[str, Any],
    exc_info: bool = False,
) -> None:
    """
    Log a message with structured context data.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        context: Context dictionary to include
        exc_info: Whether to include exception info
    """
    logger.log(level, message, extra=context, exc_info=exc_info)


def create_child_span() -> str:
    """
    Create a new child span ID for distributed tracing.

    Returns:
        New span ID (16 character hex string)
    """
    new_span_id = str(uuid.uuid4())[:16]
    parent_span_id_var.set(span_id_var.get())
    span_id_var.set(new_span_id)
    return new_span_id


class SpanContext:
    """
    Context manager for creating child spans in distributed tracing.

    Usage:
        with SpanContext("database_query") as span:
            result = await db.execute(query)
            span.set_attribute("rows", len(result))
    """

    def __init__(self, operation_name: str, **attributes: Any):
        self.operation_name = operation_name
        self.attributes = attributes
        self.start_time: float | None = None
        self.old_span_id: str | None = None
        self.old_parent_span_id: str | None = None
        self.span_id: str | None = None
        self.logger = get_logger("tracing")

    def __enter__(self) -> SpanContext:
        self.start_time = time.time()
        self.old_span_id = span_id_var.get()
        self.old_parent_span_id = parent_span_id_var.get()

        # Create new span
        self.span_id = str(uuid.uuid4())[:16]
        parent_span_id_var.set(self.old_span_id)
        span_id_var.set(self.span_id)

        self.logger.debug(
            f"Span started: {self.operation_name}",
            extra={
                "event": "span_start",
                "operation": self.operation_name,
                "span_id": self.span_id,
                "parent_span_id": self.old_span_id,
                "attributes": self.attributes,
            }
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0

        log_data = {
            "event": "span_end",
            "operation": self.operation_name,
            "span_id": self.span_id,
            "parent_span_id": self.old_span_id,
            "duration_ms": round(duration_ms, 2),
            "success": exc_type is None,
            "attributes": self.attributes,
        }

        if exc_type:
            log_data["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_val),
            }
            self.logger.error(f"Span failed: {self.operation_name}", extra=log_data)
        else:
            self.logger.debug(f"Span completed: {self.operation_name}", extra=log_data)

        # Restore previous span context
        span_id_var.set(self.old_span_id)
        parent_span_id_var.set(self.old_parent_span_id)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        self.attributes[key] = value


def enrich_error_context(**kwargs: Any) -> None:
    """
    Add context data that will be included in error logs.

    Usage:
        enrich_error_context(user_email="user@example.com", action="login")
    """
    ErrorContext.update(kwargs)


def get_current_trace_context() -> dict[str, str | None]:
    """
    Get the current distributed tracing context.

    Returns:
        Dictionary with trace_id, span_id, parent_span_id
    """
    return {
        "trace_id": trace_id_var.get(),
        "span_id": span_id_var.get(),
        "parent_span_id": parent_span_id_var.get(),
        "request_id": request_id_var.get(),
        "correlation_id": correlation_id_var.get(),
    }


def inject_trace_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Inject trace context into outgoing request headers.

    Args:
        headers: Existing headers dictionary

    Returns:
        Headers with trace context added
    """
    context = get_current_trace_context()

    if context["trace_id"]:
        headers["X-Trace-ID"] = context["trace_id"]
    if context["span_id"]:
        headers["X-Span-ID"] = context["span_id"]
    if context["request_id"]:
        headers["X-Request-ID"] = context["request_id"]
    if context["correlation_id"]:
        headers["X-Correlation-ID"] = context["correlation_id"]

    # W3C Trace Context format
    if context["trace_id"] and context["span_id"]:
        headers["traceparent"] = f"00-{context['trace_id']}-{context['span_id']}-01"

    return headers
