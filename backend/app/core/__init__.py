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

Importing this package is side-effect free
------------------------------------------
``app.core.config`` builds the Pydantic ``Settings`` object at *import* time,
and that object refuses to construct without ``SECRET_KEY`` / ``JWT_SECRET_KEY``.
While this ``__init__`` imported it eagerly, **every** ``app.core.<submodule>``
import inherited that requirement transitively - including
``app.core.dtc_codes``, whose only dependency is stdlib ``re``. That is why the
standalone scripts under ``scripts/`` could not simply
``from app.core.dtc_codes import ...``: the import died on a missing secret
before it ever reached the module. The workaround was to copy the DTC rules
into each script, which is exactly how ten divergent regexes accumulated.

The convenience re-exports below are therefore resolved lazily (PEP 562
module ``__getattr__``). ``from app.core import settings`` still works and
still builds the settings object on first use; importing a *submodule* no
longer drags in configuration, logging and retry machinery.
"""

import importlib
from typing import Any, Dict, List

# name -> submodule that defines it. Kept explicit (rather than probing every
# submodule) so a typo raises AttributeError instead of silently importing the
# whole package looking for it.
_LAZY_EXPORTS: Dict[str, str] = {}
for _submodule, _names in (
    ("config", ("get_settings", "settings")),
    (
        "exceptions",
        (
            "AuthenticationException",
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
            "PostgresConnectionException",
            "PostgresException",
            "QdrantConnectionException",
            "QdrantException",
            "RAGException",
            "RateLimitException",
            "RedisConnectionException",
            "RedisException",
            "TokenExpiredException",
            "VINValidationException",
            "ValidationException",
            "get_error_message",
        ),
    ),
    (
        "logging",
        (
            "PerformanceLogger",
            "get_logger",
            "log_database_operation",
            "log_external_api_call",
            "setup_logging",
        ),
    ),
    (
        "retry",
        (
            "DEFAULT_CONFIG",
            "LLM_CONFIG",
            "NHTSA_CONFIG",
            "RetryConfig",
            "RetryContext",
            "retry_async",
            "retry_sync",
        ),
    ),
):
    _LAZY_EXPORTS.update(dict.fromkeys(_names, f"app.core.{_submodule}"))
del _submodule, _names


def __getattr__(name: str) -> Any:
    """Resolve a re-exported name on first access (PEP 562)."""
    submodule = _LAZY_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(submodule), name)
    globals()[name] = value  # cache: __getattr__ is only consulted on a miss
    return value


def __dir__() -> List[str]:
    return sorted(__all__)


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
