# Core module
"""
Core module for AutoCognitix backend.

This module provides:
- Configuration management (config.py)
- Custom exceptions with Hungarian messages (exceptions.py)
- Global error handlers (error_handlers.py)
- Structured logging (logging.py)
- Security utilities (security.py)
- Retry utilities (retry.py)
"""

from app.core.config import get_settings, settings
from app.core.exceptions import (
    # Authentication exceptions
    AuthenticationException,
    # Base exceptions
    AutoCognitixException,
    # Database exceptions
    DatabaseException,
    DiagnosisException,
    # Business logic exceptions
    DTCValidationException,
    EmbeddingException,
    # Error codes
    ErrorCode,
    # External API exceptions
    ExternalAPIException,
    ForbiddenException,
    InvalidCredentialsException,
    InvalidTokenException,
    LLMException,
    LLMRateLimitException,
    LLMUnavailableException,
    Neo4jConnectionException,
    Neo4jException,
    NHTSAException,
    NHTSARateLimitException,
    NotFoundException,
    PostgresConnectionException,
    PostgresException,
    QdrantConnectionException,
    QdrantException,
    RAGException,
    RateLimitException,
    RedisConnectionException,
    RedisException,
    TokenExpiredException,
    ValidationException,
    VINValidationException,
    get_error_message,
)
from app.core.logging import (
    PerformanceLogger,
    get_logger,
    log_database_operation,
    log_external_api_call,
    setup_logging,
)
from app.core.retry import (
    DEFAULT_CONFIG,
    LLM_CONFIG,
    NHTSA_CONFIG,
    RetryConfig,
    RetryContext,
    retry_async,
    retry_sync,
)

__all__ = [
    "DEFAULT_CONFIG",
    "LLM_CONFIG",
    "NHTSA_CONFIG",
    "AuthenticationException",
    # Exceptions
    "AutoCognitixException",
    "DTCValidationException",
    "DatabaseException",
    "DiagnosisException",
    "EmbeddingException",
    "ErrorCode",
    "ExternalAPIException",
    "ForbiddenException",
    "InvalidCredentialsException",
    "InvalidTokenException",
    "LLMException",
    "LLMRateLimitException",
    "LLMUnavailableException",
    "NHTSAException",
    "NHTSARateLimitException",
    "Neo4jConnectionException",
    "Neo4jException",
    "NotFoundException",
    "PerformanceLogger",
    "PostgresConnectionException",
    "PostgresException",
    "QdrantConnectionException",
    "QdrantException",
    "RAGException",
    "RateLimitException",
    "RedisConnectionException",
    "RedisException",
    # Retry
    "RetryConfig",
    "RetryContext",
    "TokenExpiredException",
    "VINValidationException",
    "ValidationException",
    "get_error_message",
    "get_logger",
    "get_settings",
    "log_database_operation",
    "log_external_api_call",
    "retry_async",
    "retry_sync",
    # Config
    "settings",
    # Logging
    "setup_logging",
]
