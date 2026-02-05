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

from app.core.config import settings, get_settings
from app.core.exceptions import (
    # Base exceptions
    AutoCognitixException,
    ValidationException,
    NotFoundException,
    # Database exceptions
    DatabaseException,
    PostgresException,
    PostgresConnectionException,
    Neo4jException,
    Neo4jConnectionException,
    QdrantException,
    QdrantConnectionException,
    RedisException,
    RedisConnectionException,
    # External API exceptions
    ExternalAPIException,
    NHTSAException,
    NHTSARateLimitException,
    LLMException,
    LLMRateLimitException,
    LLMUnavailableException,
    # Business logic exceptions
    DTCValidationException,
    VINValidationException,
    DiagnosisException,
    EmbeddingException,
    RAGException,
    # Authentication exceptions
    AuthenticationException,
    InvalidCredentialsException,
    TokenExpiredException,
    InvalidTokenException,
    ForbiddenException,
    RateLimitException,
    # Error codes
    ErrorCode,
    get_error_message,
)
from app.core.logging import (
    setup_logging,
    get_logger,
    log_database_operation,
    log_external_api_call,
    PerformanceLogger,
)
from app.core.retry import (
    RetryConfig,
    retry_async,
    retry_sync,
    RetryContext,
    DEFAULT_CONFIG,
    NHTSA_CONFIG,
    LLM_CONFIG,
)

__all__ = [
    # Config
    "settings",
    "get_settings",
    # Exceptions
    "AutoCognitixException",
    "ValidationException",
    "NotFoundException",
    "DatabaseException",
    "PostgresException",
    "PostgresConnectionException",
    "Neo4jException",
    "Neo4jConnectionException",
    "QdrantException",
    "QdrantConnectionException",
    "RedisException",
    "RedisConnectionException",
    "ExternalAPIException",
    "NHTSAException",
    "NHTSARateLimitException",
    "LLMException",
    "LLMRateLimitException",
    "LLMUnavailableException",
    "DTCValidationException",
    "VINValidationException",
    "DiagnosisException",
    "EmbeddingException",
    "RAGException",
    "AuthenticationException",
    "InvalidCredentialsException",
    "TokenExpiredException",
    "InvalidTokenException",
    "ForbiddenException",
    "RateLimitException",
    "ErrorCode",
    "get_error_message",
    # Logging
    "setup_logging",
    "get_logger",
    "log_database_operation",
    "log_external_api_call",
    "PerformanceLogger",
    # Retry
    "RetryConfig",
    "retry_async",
    "retry_sync",
    "RetryContext",
    "DEFAULT_CONFIG",
    "NHTSA_CONFIG",
    "LLM_CONFIG",
]
