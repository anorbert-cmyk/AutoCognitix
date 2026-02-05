"""
Global exception handlers for FastAPI application.

This module provides:
- Centralized exception handling
- Structured error responses
- Request ID tracing
- Error logging with context
"""

import traceback
import uuid
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import (
    DBAPIError,
    IntegrityError,
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLAlchemyTimeoutError,
)
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.exceptions import (
    AutoCognitixException,
    DatabaseException,
    ErrorCode,
    PostgresConnectionException,
    PostgresException,
    get_error_message,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Request ID Middleware
# =============================================================================


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context (ID, timing) to each request."""

    async def dispatch(self, request: Request, call_next: Callable):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store request ID in state for access in handlers
        request.state.request_id = request_id

        # Log request start
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
            },
        )

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log request completion
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
            },
        )

        return response


# =============================================================================
# Error Response Builder
# =============================================================================


def build_error_response(
    request_id: str,
    code: ErrorCode,
    message: str,
    message_hu: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> JSONResponse:
    """
    Build a standardized error response.

    Args:
        request_id: Unique request identifier
        code: Error code enum
        message: English error message
        message_hu: Hungarian error message (auto-generated if not provided)
        details: Additional error details
        status_code: HTTP status code

    Returns:
        JSONResponse with structured error body
    """
    content = {
        "error": {
            "code": code.value,
            "message": message,
            "message_hu": message_hu or get_error_message(code, message),
            "details": details or {},
            "request_id": request_id,
        }
    }

    return JSONResponse(
        status_code=status_code,
        content=content,
    )


def get_request_id(request: Request) -> str:
    """Extract request ID from request state."""
    return getattr(request.state, "request_id", str(uuid.uuid4()))


# =============================================================================
# Exception Handlers
# =============================================================================


async def autocognitix_exception_handler(
    request: Request,
    exc: AutoCognitixException,
) -> JSONResponse:
    """Handle AutoCognitix custom exceptions."""
    request_id = get_request_id(request)

    logger.warning(
        f"AutoCognitix exception: {exc.message}",
        extra={
            "request_id": request_id,
            "error_code": exc.code.value,
            "details": exc.details,
            "path": request.url.path,
        },
    )

    return build_error_response(
        request_id=request_id,
        code=exc.code,
        message=exc.message,
        details=exc.details,
        status_code=exc.status_code,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = get_request_id(request)

    # Extract validation error details
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
        })

    logger.warning(
        "Validation error",
        extra={
            "request_id": request_id,
            "errors": errors,
            "path": request.url.path,
        },
    )

    return build_error_response(
        request_id=request_id,
        code=ErrorCode.VALIDATION_ERROR,
        message="Validation error",
        message_hu="Ervenytelen adatok. Kerem, ellenorizze a bevitt adatokat.",
        details={"validation_errors": errors},
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def pydantic_validation_exception_handler(
    request: Request,
    exc: ValidationError,
) -> JSONResponse:
    """Handle Pydantic ValidationError (different from RequestValidationError)."""
    request_id = get_request_id(request)

    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"],
        })

    logger.warning(
        "Pydantic validation error",
        extra={
            "request_id": request_id,
            "errors": errors,
            "path": request.url.path,
        },
    )

    return build_error_response(
        request_id=request_id,
        code=ErrorCode.VALIDATION_ERROR,
        message="Validation error",
        message_hu="Ervenytelen adatok.",
        details={"validation_errors": errors},
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def sqlalchemy_exception_handler(
    request: Request,
    exc: SQLAlchemyError,
) -> JSONResponse:
    """Handle SQLAlchemy database errors."""
    request_id = get_request_id(request)

    # Determine specific error type
    if isinstance(exc, OperationalError):
        code = ErrorCode.DATABASE_CONNECTION
        message = "Database connection error"
        message_hu = "Adatbazis kapcsolati hiba."
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, IntegrityError):
        code = ErrorCode.DATABASE_INTEGRITY
        message = "Database integrity error"
        message_hu = "Adatintegritasi hiba."
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(exc, SQLAlchemyTimeoutError):
        code = ErrorCode.DATABASE_TIMEOUT
        message = "Database timeout"
        message_hu = "Adatbazis idotullpes."
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
    elif isinstance(exc, DBAPIError):
        code = ErrorCode.POSTGRES_ERROR
        message = "Database error"
        message_hu = "Adatbazis hiba."
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    else:
        code = ErrorCode.DATABASE_ERROR
        message = "Database error"
        message_hu = "Adatbazis hiba tortent."
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    # Log with full error in non-production
    log_details = {
        "request_id": request_id,
        "error_type": type(exc).__name__,
        "path": request.url.path,
    }

    if settings.DEBUG:
        log_details["error_message"] = str(exc)
        log_details["traceback"] = traceback.format_exc()

    logger.error(f"Database error: {type(exc).__name__}", extra=log_details)

    # Only include error details in debug mode
    details = {}
    if settings.DEBUG:
        details["error_type"] = type(exc).__name__
        details["error_message"] = str(exc)[:200]  # Truncate for security

    return build_error_response(
        request_id=request_id,
        code=code,
        message=message,
        message_hu=message_hu,
        details=details,
        status_code=status_code,
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle all unhandled exceptions."""
    request_id = get_request_id(request)

    # Log full traceback
    log_details = {
        "request_id": request_id,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "path": request.url.path,
        "method": request.method,
    }

    if settings.DEBUG:
        log_details["traceback"] = traceback.format_exc()

    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}",
        extra=log_details,
        exc_info=True,
    )

    # Only include error details in debug mode
    details = {}
    if settings.DEBUG:
        details["error_type"] = type(exc).__name__
        details["error_message"] = str(exc)[:200]

    return build_error_response(
        request_id=request_id,
        code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred",
        message_hu="Varatlan hiba tortent. Kerem, probalkozzon kesobb.",
        details=details,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# =============================================================================
# Setup Function
# =============================================================================


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers with the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    # Add request context middleware
    app.add_middleware(RequestContextMiddleware)

    # Register custom exception handlers
    app.add_exception_handler(AutoCognitixException, autocognitix_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)

    # Generic handler for unhandled exceptions (must be last)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Exception handlers registered")


# =============================================================================
# Neo4j Exception Handler (separate due to optional import)
# =============================================================================


def setup_neo4j_exception_handler(app: FastAPI) -> None:
    """Setup Neo4j-specific exception handlers if neo4j is available."""
    try:
        from neo4j.exceptions import (
            AuthError as Neo4jAuthError,
            ServiceUnavailable as Neo4jServiceUnavailable,
            SessionExpired as Neo4jSessionExpired,
            Neo4jError,
        )

        async def neo4j_exception_handler(
            request: Request,
            exc: Neo4jError,
        ) -> JSONResponse:
            """Handle Neo4j database errors."""
            request_id = get_request_id(request)

            if isinstance(exc, Neo4jServiceUnavailable):
                code = ErrorCode.NEO4J_CONNECTION
                message = "Neo4j service unavailable"
                message_hu = "Neo4j szolgaltatas nem elerheto."
            elif isinstance(exc, Neo4jAuthError):
                code = ErrorCode.NEO4J_ERROR
                message = "Neo4j authentication error"
                message_hu = "Neo4j hitelesitesi hiba."
            elif isinstance(exc, Neo4jSessionExpired):
                code = ErrorCode.NEO4J_ERROR
                message = "Neo4j session expired"
                message_hu = "Neo4j munkamenet lejart."
            else:
                code = ErrorCode.NEO4J_ERROR
                message = "Neo4j error"
                message_hu = "Neo4j grafadatbazis hiba."

            logger.error(
                f"Neo4j error: {type(exc).__name__}",
                extra={
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "path": request.url.path,
                },
            )

            details = {}
            if settings.DEBUG:
                details["error_type"] = type(exc).__name__

            return build_error_response(
                request_id=request_id,
                code=code,
                message=message,
                message_hu=message_hu,
                details=details,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        app.add_exception_handler(Neo4jError, neo4j_exception_handler)
        logger.info("Neo4j exception handler registered")

    except ImportError:
        logger.debug("Neo4j not installed, skipping Neo4j exception handler")


def setup_qdrant_exception_handler(app: FastAPI) -> None:
    """Setup Qdrant-specific exception handlers if qdrant is available."""
    try:
        from qdrant_client.http.exceptions import (
            UnexpectedResponse as QdrantUnexpectedResponse,
            ResponseHandlingException as QdrantResponseError,
        )

        async def qdrant_exception_handler(
            request: Request,
            exc: Exception,
        ) -> JSONResponse:
            """Handle Qdrant vector database errors."""
            request_id = get_request_id(request)

            logger.error(
                f"Qdrant error: {type(exc).__name__}",
                extra={
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "path": request.url.path,
                },
            )

            details = {}
            if settings.DEBUG:
                details["error_type"] = type(exc).__name__

            return build_error_response(
                request_id=request_id,
                code=ErrorCode.QDRANT_ERROR,
                message="Vector database error",
                message_hu="Vektor adatbazis hiba.",
                details=details,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        app.add_exception_handler(QdrantUnexpectedResponse, qdrant_exception_handler)
        app.add_exception_handler(QdrantResponseError, qdrant_exception_handler)
        logger.info("Qdrant exception handler registered")

    except ImportError:
        logger.debug("Qdrant client not installed, skipping Qdrant exception handler")


def setup_httpx_exception_handler(app: FastAPI) -> None:
    """Setup HTTPX exception handlers for external API calls."""
    try:
        import httpx

        async def httpx_exception_handler(
            request: Request,
            exc: httpx.HTTPError,
        ) -> JSONResponse:
            """Handle HTTPX errors (external API calls)."""
            request_id = get_request_id(request)

            if isinstance(exc, httpx.TimeoutException):
                code = ErrorCode.REQUEST_TIMEOUT
                message = "External API timeout"
                message_hu = "Kulso szolgaltatas idotullpes."
                status_code = status.HTTP_504_GATEWAY_TIMEOUT
            elif isinstance(exc, httpx.ConnectError):
                code = ErrorCode.EXTERNAL_API_ERROR
                message = "External API connection error"
                message_hu = "Kulso szolgaltatas kapcsolati hiba."
                status_code = status.HTTP_502_BAD_GATEWAY
            else:
                code = ErrorCode.EXTERNAL_API_ERROR
                message = "External API error"
                message_hu = "Kulso szolgaltatas hiba."
                status_code = status.HTTP_502_BAD_GATEWAY

            logger.error(
                f"HTTPX error: {type(exc).__name__}",
                extra={
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "path": request.url.path,
                },
            )

            return build_error_response(
                request_id=request_id,
                code=code,
                message=message,
                message_hu=message_hu,
                details={},
                status_code=status_code,
            )

        app.add_exception_handler(httpx.HTTPError, httpx_exception_handler)
        logger.info("HTTPX exception handler registered")

    except ImportError:
        logger.debug("HTTPX not installed, skipping HTTPX exception handler")


def setup_all_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers including optional ones.

    This is the main function to call from the application startup.

    Args:
        app: FastAPI application instance
    """
    # Core exception handlers
    setup_exception_handlers(app)

    # Optional handlers for specific libraries
    setup_neo4j_exception_handler(app)
    setup_qdrant_exception_handler(app)
    setup_httpx_exception_handler(app)

    logger.info("All exception handlers configured")
