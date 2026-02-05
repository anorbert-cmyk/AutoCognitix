"""
Comprehensive metrics middleware for AutoCognitix.

Provides Prometheus-compatible metrics for:
- Request count by method, endpoint, status code
- Response time histograms with configurable buckets
- Error rate tracking by type
- Active connections gauge
- Request/response size tracking
- Endpoint-specific latency percentiles

This middleware is designed to be lightweight and production-ready,
with features like endpoint normalization to prevent cardinality explosion.

Usage:
    from app.middleware.metrics import MetricsMiddleware, get_metrics_middleware

    # In FastAPI app setup
    app.add_middleware(MetricsMiddleware)

    # Or get the middleware class
    middleware_class = get_metrics_middleware()
    app.add_middleware(middleware_class)
"""

import re
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Optional, Set

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import settings
from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)


# =============================================================================
# Prometheus Metrics Definitions
# =============================================================================

# Application info metric
APP_INFO = Info(
    "autocognitix_app_info",
    "AutoCognitix application information",
)
APP_INFO.info({
    "version": "0.1.0",
    "environment": settings.ENVIRONMENT,
    "service": settings.PROJECT_NAME,
    "python_version": "3.11",
})

# Request metrics
HTTP_REQUESTS_TOTAL = Counter(
    "autocognitix_http_requests_total",
    "Total HTTP request count",
    ["method", "endpoint", "status_code", "status_class"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "autocognitix_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[
        0.005, 0.01, 0.025, 0.05, 0.075,
        0.1, 0.25, 0.5, 0.75,
        1.0, 2.5, 5.0, 7.5, 10.0,
        15.0, 30.0, 60.0,
    ],
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "autocognitix_http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["method", "endpoint"],
)

HTTP_REQUEST_SIZE_BYTES = Summary(
    "autocognitix_http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
)

HTTP_RESPONSE_SIZE_BYTES = Summary(
    "autocognitix_http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
)

# Error metrics
HTTP_ERRORS_TOTAL = Counter(
    "autocognitix_http_errors_total",
    "Total HTTP error count",
    ["method", "endpoint", "error_type", "status_code"],
)

EXCEPTIONS_TOTAL = Counter(
    "autocognitix_exceptions_total",
    "Total unhandled exceptions",
    ["method", "endpoint", "exception_type"],
)

# Connection metrics
ACTIVE_CONNECTIONS = Gauge(
    "autocognitix_active_connections",
    "Number of active connections",
)

# Endpoint-specific latency percentiles
ENDPOINT_LATENCY_PERCENTILES = Summary(
    "autocognitix_endpoint_latency_seconds",
    "Endpoint latency percentiles",
    ["endpoint"],
)


# =============================================================================
# Endpoint Normalization
# =============================================================================

class EndpointNormalizer:
    """
    Normalizes endpoint paths to prevent metric cardinality explosion.

    Replaces dynamic path segments (UUIDs, IDs, VINs, DTC codes) with placeholders.
    """

    # Patterns for dynamic segments
    UUID_PATTERN = re.compile(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    NUMERIC_ID_PATTERN = re.compile(r"^\d+$")
    VIN_PATTERN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$", re.IGNORECASE)
    DTC_PATTERN = re.compile(r"^[PBCU][0-9A-F]{4}$", re.IGNORECASE)

    # Known static paths that should not be normalized
    STATIC_PATHS: Set[str] = {
        "/health",
        "/health/live",
        "/health/ready",
        "/health/detailed",
        "/health/db",
        "/metrics",
        "/metrics/summary",
        "/metrics/prometheus",
        "/api/v1/docs",
        "/api/v1/redoc",
        "/api/v1/openapi.json",
    }

    @classmethod
    def normalize(cls, path: str) -> str:
        """
        Normalize an endpoint path.

        Args:
            path: Original request path

        Returns:
            Normalized path with dynamic segments replaced
        """
        # Return static paths as-is
        if path in cls.STATIC_PATHS:
            return path

        # Split path and normalize each segment
        segments = path.split("/")
        normalized = []

        for segment in segments:
            if not segment:
                continue

            # Check patterns in order of specificity
            if cls.UUID_PATTERN.match(segment):
                normalized.append("{uuid}")
            elif cls.VIN_PATTERN.match(segment):
                normalized.append("{vin}")
            elif cls.DTC_PATTERN.match(segment):
                normalized.append("{dtc_code}")
            elif cls.NUMERIC_ID_PATTERN.match(segment):
                normalized.append("{id}")
            else:
                normalized.append(segment)

        result = "/" + "/".join(normalized) if normalized else "/"
        return result


# =============================================================================
# Request Tracking Context Manager
# =============================================================================

@contextmanager
def track_request(
    method: str,
    endpoint: str,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for tracking request metrics.

    Args:
        method: HTTP method
        endpoint: Normalized endpoint path

    Yields:
        Context dictionary for storing request-specific data

    Usage:
        with track_request("GET", "/api/v1/dtc/{id}") as ctx:
            response = await process_request()
            ctx["status_code"] = response.status_code
    """
    context: Dict[str, Any] = {
        "status_code": 0,
        "request_size": 0,
        "response_size": 0,
        "error_type": None,
    }

    # Track in-progress request
    HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
    ACTIVE_CONNECTIONS.inc()

    start_time = time.time()

    try:
        yield context
    except Exception as e:
        context["error_type"] = type(e).__name__
        context["status_code"] = 500
        raise
    finally:
        duration = time.time() - start_time
        status_code = context.get("status_code", 0)
        status_class = f"{status_code // 100}xx" if status_code else "unknown"

        # Decrement in-progress counters
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
        ACTIVE_CONNECTIONS.dec()

        # Record request metrics
        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
            status_class=status_class,
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        ENDPOINT_LATENCY_PERCENTILES.labels(
            endpoint=endpoint,
        ).observe(duration)

        # Record size metrics
        if context.get("request_size"):
            HTTP_REQUEST_SIZE_BYTES.labels(
                method=method,
                endpoint=endpoint,
            ).observe(context["request_size"])

        if context.get("response_size"):
            HTTP_RESPONSE_SIZE_BYTES.labels(
                method=method,
                endpoint=endpoint,
            ).observe(context["response_size"])

        # Record errors
        if context.get("error_type"):
            HTTP_ERRORS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                error_type=context["error_type"],
                status_code=str(status_code),
            ).inc()


# =============================================================================
# Metrics Middleware
# =============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware for Prometheus metrics collection.

    Features:
    - Automatic request counting and latency tracking
    - Endpoint normalization to prevent cardinality explosion
    - Request/response size tracking
    - Error and exception tracking
    - In-progress request gauge
    - Configurable excluded endpoints

    The middleware is designed to be lightweight and have minimal
    impact on request processing time.
    """

    # Endpoints to exclude from metrics (health checks, etc.)
    EXCLUDED_ENDPOINTS: Set[str] = {
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
    }

    def __init__(self, app: ASGIApp, exclude_paths: Optional[Set[str]] = None):
        """
        Initialize metrics middleware.

        Args:
            app: ASGI application
            exclude_paths: Optional set of paths to exclude from metrics
        """
        super().__init__(app)
        if exclude_paths:
            self.EXCLUDED_ENDPOINTS = self.EXCLUDED_ENDPOINTS.union(exclude_paths)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        # Get normalized endpoint
        path = request.url.path
        endpoint = EndpointNormalizer.normalize(path)
        method = request.method

        # Skip excluded endpoints
        if endpoint in self.EXCLUDED_ENDPOINTS:
            return await call_next(request)

        # Track request with context manager
        with track_request(method, endpoint) as ctx:
            # Get request size
            content_length = request.headers.get("Content-Length")
            if content_length:
                try:
                    ctx["request_size"] = int(content_length)
                except ValueError:
                    pass

            try:
                response = await call_next(request)
                ctx["status_code"] = response.status_code

                # Get response size
                response_length = response.headers.get("Content-Length")
                if response_length:
                    try:
                        ctx["response_size"] = int(response_length)
                    except ValueError:
                        pass

                # Track errors (4xx and 5xx)
                if response.status_code >= 400:
                    if response.status_code >= 500:
                        ctx["error_type"] = "server_error"
                    else:
                        ctx["error_type"] = "client_error"

                return response

            except Exception as e:
                # Track unhandled exceptions
                ctx["error_type"] = type(e).__name__
                ctx["status_code"] = 500

                EXCEPTIONS_TOTAL.labels(
                    method=method,
                    endpoint=endpoint,
                    exception_type=type(e).__name__,
                ).inc()

                logger.error(
                    f"Unhandled exception in request: {method} {path}",
                    extra={
                        "event": "unhandled_exception",
                        "method": method,
                        "path": path,
                        "endpoint": endpoint,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                    },
                    exc_info=True,
                )
                raise


def get_metrics_middleware() -> type:
    """
    Get the MetricsMiddleware class.

    Returns:
        MetricsMiddleware class for use with FastAPI
    """
    return MetricsMiddleware


# =============================================================================
# Metrics Generation
# =============================================================================

async def generate_metrics_response() -> Response:
    """
    Generate Prometheus-format metrics response.

    Returns:
        Response with Prometheus text format metrics
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def get_metrics_text() -> bytes:
    """
    Get metrics in Prometheus text format.

    Returns:
        Bytes containing Prometheus metrics
    """
    return generate_latest()


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a human-readable metrics summary.

    Returns:
        Dictionary with key metrics for dashboards
    """
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "version": "0.1.0",
        "endpoints": {
            "metrics": "/metrics",
            "metrics_summary": "/metrics/summary",
            "health": "/health",
            "health_detailed": "/health/detailed",
        },
    }
