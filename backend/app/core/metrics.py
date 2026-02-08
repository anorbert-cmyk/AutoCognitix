"""
Comprehensive metrics collection for AutoCognitix.

Provides Prometheus-format metrics for:
- HTTP request count and latency by endpoint
- Database operation metrics (query times, connection pool)
- Embedding generation metrics
- Diagnosis request tracking
- Error rates and active users
- System resource metrics

Usage:
    # In middleware or endpoint handlers
    from app.core.metrics import (
        track_request,
        track_database_query,
        track_embedding_generation,
        get_metrics_middleware,
    )

    # Track a database query
    with track_database_query("postgres", "select"):
        result = await db.execute(query)
"""

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from datetime import datetime
from typing import Any

import psutil
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

from app.core.config import settings

# =============================================================================
# Application Info
# =============================================================================

APP_INFO = Info(
    "autocognitix_app",
    "AutoCognitix application information",
)
APP_INFO.info(
    {
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "service": settings.PROJECT_NAME,
    }
)

# =============================================================================
# HTTP Request Metrics
# =============================================================================

REQUEST_COUNT = Counter(
    "autocognitix_http_requests_total",
    "Total HTTP request count",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "autocognitix_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUEST_IN_PROGRESS = Gauge(
    "autocognitix_http_requests_in_progress",
    "Number of HTTP requests currently in progress",
    ["method", "endpoint"],
)

REQUEST_SIZE = Summary(
    "autocognitix_http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint"],
)

RESPONSE_SIZE = Summary(
    "autocognitix_http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
)

# =============================================================================
# Database Metrics
# =============================================================================

DB_QUERY_COUNT = Counter(
    "autocognitix_db_queries_total",
    "Total database query count",
    ["database", "operation", "table"],
)

DB_QUERY_LATENCY = Histogram(
    "autocognitix_db_query_duration_seconds",
    "Database query latency in seconds",
    ["database", "operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

DB_QUERY_ERRORS = Counter(
    "autocognitix_db_query_errors_total",
    "Total database query errors",
    ["database", "operation", "error_type"],
)

DB_CONNECTION_POOL = Gauge(
    "autocognitix_db_connections",
    "Database connection pool metrics",
    ["database", "state"],
)

DB_ROWS_AFFECTED = Counter(
    "autocognitix_db_rows_affected_total",
    "Total rows affected by database operations",
    ["database", "operation"],
)

# =============================================================================
# Embedding & Vector Search Metrics
# =============================================================================

EMBEDDING_GENERATION_COUNT = Counter(
    "autocognitix_embedding_generations_total",
    "Total embedding generations",
    ["model", "status"],
)

EMBEDDING_GENERATION_LATENCY = Histogram(
    "autocognitix_embedding_generation_duration_seconds",
    "Embedding generation latency in seconds",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

EMBEDDING_BATCH_SIZE = Histogram(
    "autocognitix_embedding_batch_size",
    "Batch size for embedding generations",
    ["model"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

VECTOR_SEARCH_COUNT = Counter(
    "autocognitix_vector_searches_total",
    "Total vector search operations",
    ["collection", "status"],
)

VECTOR_SEARCH_LATENCY = Histogram(
    "autocognitix_vector_search_duration_seconds",
    "Vector search latency in seconds",
    ["collection"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

VECTOR_SEARCH_RESULTS = Histogram(
    "autocognitix_vector_search_results_count",
    "Number of results returned by vector search",
    ["collection"],
    buckets=[0, 1, 5, 10, 25, 50, 100],
)

# =============================================================================
# Diagnosis & Business Metrics
# =============================================================================

DIAGNOSIS_REQUEST_COUNT = Counter(
    "autocognitix_diagnosis_requests_total",
    "Total diagnosis requests",
    ["status", "language"],
)

DIAGNOSIS_LATENCY = Histogram(
    "autocognitix_diagnosis_duration_seconds",
    "Diagnosis request latency in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 60.0],
)

DTC_LOOKUP_COUNT = Counter(
    "autocognitix_dtc_lookups_total",
    "Total DTC code lookups",
    ["found"],
)

VEHICLE_DECODE_COUNT = Counter(
    "autocognitix_vehicle_decodes_total",
    "Total VIN decode requests",
    ["status"],
)

# =============================================================================
# Authentication & User Metrics
# =============================================================================

AUTH_ATTEMPTS = Counter(
    "autocognitix_auth_attempts_total",
    "Total authentication attempts",
    ["method", "status"],
)

ACTIVE_SESSIONS = Gauge(
    "autocognitix_active_sessions",
    "Number of active user sessions",
)

USER_REGISTRATIONS = Counter(
    "autocognitix_user_registrations_total",
    "Total user registrations",
    ["status"],
)

# =============================================================================
# External API Metrics
# =============================================================================

EXTERNAL_API_CALLS = Counter(
    "autocognitix_external_api_calls_total",
    "Total external API calls",
    ["service", "endpoint", "status_code"],
)

EXTERNAL_API_LATENCY = Histogram(
    "autocognitix_external_api_duration_seconds",
    "External API call latency in seconds",
    ["service"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

EXTERNAL_API_ERRORS = Counter(
    "autocognitix_external_api_errors_total",
    "Total external API errors",
    ["service", "error_type"],
)

# =============================================================================
# LLM Metrics
# =============================================================================

LLM_REQUESTS = Counter(
    "autocognitix_llm_requests_total",
    "Total LLM API requests",
    ["provider", "model", "status"],
)

LLM_LATENCY = Histogram(
    "autocognitix_llm_duration_seconds",
    "LLM API request latency in seconds",
    ["provider", "model"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 60.0, 120.0],
)

LLM_TOKENS = Counter(
    "autocognitix_llm_tokens_total",
    "Total LLM tokens used",
    ["provider", "model", "type"],  # token type: input/output
)

# =============================================================================
# Error Metrics
# =============================================================================

ERROR_COUNT = Counter(
    "autocognitix_errors_total",
    "Total error count",
    ["error_type", "endpoint"],
)

EXCEPTION_COUNT = Counter(
    "autocognitix_exceptions_total",
    "Total unhandled exceptions",
    ["exception_type", "endpoint"],
)

# =============================================================================
# System Resource Metrics
# =============================================================================

SYSTEM_CPU_PERCENT = Gauge(
    "autocognitix_system_cpu_percent",
    "System CPU usage percentage",
)

SYSTEM_MEMORY_PERCENT = Gauge(
    "autocognitix_system_memory_percent",
    "System memory usage percentage",
)

SYSTEM_MEMORY_BYTES = Gauge(
    "autocognitix_system_memory_bytes",
    "System memory usage in bytes",
    ["type"],
)

PROCESS_CPU_PERCENT = Gauge(
    "autocognitix_process_cpu_percent",
    "Process CPU usage percentage",
)

PROCESS_MEMORY_BYTES = Gauge(
    "autocognitix_process_memory_bytes",
    "Process memory usage in bytes",
    ["type"],
)

# =============================================================================
# Data Metrics
# =============================================================================

DTC_CODES_TOTAL = Gauge(
    "autocognitix_dtc_codes_total",
    "Total DTC codes in database",
)

VEHICLES_TOTAL = Gauge(
    "autocognitix_vehicles_total",
    "Total vehicles in database",
)

USERS_TOTAL = Gauge(
    "autocognitix_users_total",
    "Total registered users",
)


# =============================================================================
# Metric Collection Functions
# =============================================================================


def update_system_metrics() -> None:
    """Update system resource metrics."""
    try:
        # System-wide metrics
        cpu_percent = psutil.cpu_percent()
        SYSTEM_CPU_PERCENT.set(cpu_percent)

        memory = psutil.virtual_memory()
        SYSTEM_MEMORY_PERCENT.set(memory.percent)
        SYSTEM_MEMORY_BYTES.labels(type="total").set(memory.total)
        SYSTEM_MEMORY_BYTES.labels(type="available").set(memory.available)
        SYSTEM_MEMORY_BYTES.labels(type="used").set(memory.used)

        # Process-specific metrics
        process = psutil.Process()
        PROCESS_CPU_PERCENT.set(process.cpu_percent())

        mem_info = process.memory_info()
        PROCESS_MEMORY_BYTES.labels(type="rss").set(mem_info.rss)
        PROCESS_MEMORY_BYTES.labels(type="vms").set(mem_info.vms)

    except Exception:
        pass  # Silently ignore errors in metrics collection


@contextmanager
def track_database_query(
    database: str,
    operation: str,
    table: str = "unknown",
) -> Generator[None, None, None]:
    """
    Context manager for tracking database query metrics.

    Args:
        database: Database name (postgres, neo4j, qdrant, redis)
        operation: Operation type (select, insert, update, delete, search)
        table: Table/collection name

    Usage:
        with track_database_query("postgres", "select", "dtc_codes"):
            result = await db.execute(query)
    """
    start_time = time.time()
    error_occurred = False
    error_type = None

    try:
        yield
    except Exception as e:
        error_occurred = True
        error_type = type(e).__name__
        raise
    finally:
        duration = time.time() - start_time

        # Record metrics
        DB_QUERY_COUNT.labels(
            database=database,
            operation=operation,
            table=table,
        ).inc()

        DB_QUERY_LATENCY.labels(
            database=database,
            operation=operation,
        ).observe(duration)

        if error_occurred:
            DB_QUERY_ERRORS.labels(
                database=database,
                operation=operation,
                error_type=error_type,
            ).inc()


@contextmanager
def track_embedding_generation(
    model: str = "hubert",
    batch_size: int = 1,
) -> Generator[None, None, None]:
    """
    Context manager for tracking embedding generation metrics.

    Args:
        model: Embedding model name
        batch_size: Number of texts in batch

    Usage:
        with track_embedding_generation("hubert", batch_size=10):
            embeddings = model.encode(texts)
    """
    start_time = time.time()
    success = True

    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration = time.time() - start_time

        EMBEDDING_GENERATION_COUNT.labels(
            model=model,
            status="success" if success else "error",
        ).inc()

        EMBEDDING_GENERATION_LATENCY.labels(model=model).observe(duration)
        EMBEDDING_BATCH_SIZE.labels(model=model).observe(batch_size)


@contextmanager
def track_vector_search(
    collection: str,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager for tracking vector search metrics.

    Args:
        collection: Qdrant collection name

    Usage:
        with track_vector_search("dtc_embeddings") as ctx:
            results = await qdrant.search(...)
            ctx["results_count"] = len(results)
    """
    start_time = time.time()
    context: dict[str, Any] = {"results_count": 0}
    success = True

    try:
        yield context
    except Exception:
        success = False
        raise
    finally:
        duration = time.time() - start_time

        VECTOR_SEARCH_COUNT.labels(
            collection=collection,
            status="success" if success else "error",
        ).inc()

        VECTOR_SEARCH_LATENCY.labels(collection=collection).observe(duration)
        VECTOR_SEARCH_RESULTS.labels(collection=collection).observe(context.get("results_count", 0))


@contextmanager
def track_diagnosis_request(
    language: str = "hu",
) -> Generator[None, None, None]:
    """
    Context manager for tracking diagnosis request metrics.

    Args:
        language: Request language (hu, en)

    Usage:
        with track_diagnosis_request("hu"):
            diagnosis = await diagnosis_service.analyze(...)
    """
    start_time = time.time()
    success = True

    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration = time.time() - start_time

        DIAGNOSIS_REQUEST_COUNT.labels(
            status="success" if success else "error",
            language=language,
        ).inc()

        DIAGNOSIS_LATENCY.observe(duration)


@contextmanager
def track_external_api_call(
    service: str,
    endpoint: str = "",
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager for tracking external API call metrics.

    Args:
        service: External service name (nhtsa, carmd, etc.)
        endpoint: API endpoint

    Usage:
        with track_external_api_call("nhtsa", "/vehicles/decode-vin") as ctx:
            response = await client.get(url)
            ctx["status_code"] = response.status_code
    """
    start_time = time.time()
    context: dict[str, Any] = {"status_code": 0}

    try:
        yield context
    except Exception as e:
        EXTERNAL_API_ERRORS.labels(
            service=service,
            error_type=type(e).__name__,
        ).inc()
        raise
    finally:
        duration = time.time() - start_time

        EXTERNAL_API_CALLS.labels(
            service=service,
            endpoint=endpoint,
            status_code=str(context.get("status_code", 0)),
        ).inc()

        EXTERNAL_API_LATENCY.labels(service=service).observe(duration)


@contextmanager
def track_llm_request(
    provider: str,
    model: str,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager for tracking LLM request metrics.

    Args:
        provider: LLM provider (anthropic, openai, ollama)
        model: Model name

    Usage:
        with track_llm_request("anthropic", "claude-3") as ctx:
            response = await llm.generate(...)
            ctx["input_tokens"] = response.usage.input_tokens
            ctx["output_tokens"] = response.usage.output_tokens
    """
    start_time = time.time()
    context: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    success = True

    try:
        yield context
    except Exception:
        success = False
        raise
    finally:
        duration = time.time() - start_time

        LLM_REQUESTS.labels(
            provider=provider,
            model=model,
            status="success" if success else "error",
        ).inc()

        LLM_LATENCY.labels(provider=provider, model=model).observe(duration)

        if context.get("input_tokens"):
            LLM_TOKENS.labels(
                provider=provider,
                model=model,
                type="input",
            ).inc(context["input_tokens"])

        if context.get("output_tokens"):
            LLM_TOKENS.labels(
                provider=provider,
                model=model,
                type="output",
            ).inc(context["output_tokens"])


def track_request_start(method: str, endpoint: str) -> None:
    """Track request start for in-progress gauge."""
    REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()


def track_request_complete(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float,
    request_size: int = 0,
    response_size: int = 0,
) -> None:
    """Track request completion metrics."""
    REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()

    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code),
    ).inc()

    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    if request_size:
        REQUEST_SIZE.labels(method=method, endpoint=endpoint).observe(request_size)

    if response_size:
        RESPONSE_SIZE.labels(method=method, endpoint=endpoint).observe(response_size)


def track_error(error_type: str, endpoint: str = "unknown") -> None:
    """Track an error occurrence."""
    ERROR_COUNT.labels(error_type=error_type, endpoint=endpoint).inc()


def track_exception(exception_type: str, endpoint: str = "unknown") -> None:
    """Track an unhandled exception."""
    EXCEPTION_COUNT.labels(exception_type=exception_type, endpoint=endpoint).inc()


def track_auth_attempt(method: str, success: bool) -> None:
    """Track authentication attempt."""
    AUTH_ATTEMPTS.labels(
        method=method,
        status="success" if success else "failure",
    ).inc()


def set_active_sessions(count: int) -> None:
    """Set active session count."""
    ACTIVE_SESSIONS.set(count)


def track_dtc_lookup(found: bool) -> None:
    """Track DTC code lookup."""
    DTC_LOOKUP_COUNT.labels(found="yes" if found else "no").inc()


def track_vehicle_decode(success: bool) -> None:
    """Track VIN decode request."""
    VEHICLE_DECODE_COUNT.labels(status="success" if success else "error").inc()


def set_data_metrics(dtc_count: int = 0, vehicle_count: int = 0, user_count: int = 0) -> None:
    """Set data count metrics."""
    if dtc_count >= 0:
        DTC_CODES_TOTAL.set(dtc_count)
    if vehicle_count >= 0:
        VEHICLES_TOTAL.set(vehicle_count)
    if user_count >= 0:
        USERS_TOTAL.set(user_count)


def set_db_pool_metrics(database: str, active: int, idle: int, waiting: int = 0) -> None:
    """Set database connection pool metrics."""
    DB_CONNECTION_POOL.labels(database=database, state="active").set(active)
    DB_CONNECTION_POOL.labels(database=database, state="idle").set(idle)
    DB_CONNECTION_POOL.labels(database=database, state="waiting").set(waiting)


# =============================================================================
# Metrics Middleware
# =============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic request metrics collection.

    Tracks:
    - Request count by method, endpoint, status
    - Request latency
    - Requests in progress
    - Request/response sizes
    """

    # Endpoints to exclude from detailed metrics (to prevent cardinality explosion)
    EXCLUDED_ENDPOINTS = {"/health", "/health/live", "/health/ready", "/metrics"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Normalize endpoint for metrics (avoid high cardinality)
        endpoint = self._normalize_endpoint(request.url.path)

        # Skip excluded endpoints
        if endpoint in self.EXCLUDED_ENDPOINTS:
            return await call_next(request)

        method = request.method

        # Track request start
        track_request_start(method, endpoint)

        # Get request size
        request_size = 0
        content_length = request.headers.get("Content-Length")
        if content_length:
            with suppress(ValueError):
                request_size = int(content_length)

        start_time = time.time()

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Get response size
            response_size = 0
            content_length = response.headers.get("Content-Length")
            if content_length:
                with suppress(ValueError):
                    response_size = int(content_length)

            # Track completion
            track_request_complete(
                method=method,
                endpoint=endpoint,
                status_code=response.status_code,
                duration=duration,
                request_size=request_size,
                response_size=response_size,
            )

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Track error
            track_request_complete(
                method=method,
                endpoint=endpoint,
                status_code=500,
                duration=duration,
                request_size=request_size,
            )

            track_exception(type(e).__name__, endpoint)
            raise

    def _normalize_endpoint(self, path: str) -> str:
        """
        Normalize endpoint path to prevent cardinality explosion.

        Replaces dynamic path parameters with placeholders.
        """
        parts = path.split("/")
        normalized_parts = []

        for part in parts:
            if not part:
                continue

            # Replace UUIDs with placeholder
            if self._is_uuid(part) or part.isdigit():
                normalized_parts.append("{id}")
            # Replace VIN patterns with placeholder
            elif len(part) == 17 and part.isalnum():
                normalized_parts.append("{vin}")
            # Replace DTC code patterns with placeholder
            elif len(part) >= 4 and part[0] in "PBCU" and part[1:].replace("-", "").isalnum():
                normalized_parts.append("{dtc_code}")
            else:
                normalized_parts.append(part)

        return "/" + "/".join(normalized_parts) if normalized_parts else "/"

    @staticmethod
    def _is_uuid(value: str) -> bool:
        """Check if value looks like a UUID."""
        try:
            import uuid as uuid_module

            uuid_module.UUID(value)
            return True
        except (ValueError, AttributeError):
            return False


def get_metrics_middleware() -> Callable:
    """
    Get the metrics middleware for FastAPI.

    Returns:
        MetricsMiddleware class
    """
    return MetricsMiddleware


# =============================================================================
# Metrics Export
# =============================================================================


async def generate_metrics_response() -> Response:
    """
    Generate Prometheus metrics response.

    Updates system metrics before generating response.
    """
    # Update system metrics
    update_system_metrics()

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def get_metrics_summary() -> dict[str, Any]:
    """
    Get human-readable metrics summary.

    Returns:
        Dictionary with key metrics
    """
    update_system_metrics()

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "system": {
            "cpu_percent": SYSTEM_CPU_PERCENT._value._value
            if hasattr(SYSTEM_CPU_PERCENT._value, "_value")
            else 0,
            "memory_percent": SYSTEM_MEMORY_PERCENT._value._value
            if hasattr(SYSTEM_MEMORY_PERCENT._value, "_value")
            else 0,
        },
        "endpoints": {
            "metrics": "/metrics",
            "health": "/health",
            "health_live": "/health/live",
            "health_ready": "/health/ready",
        },
    }
