"""
Custom exception classes for AutoCognitix.

This module defines a hierarchy of exceptions with:
- Structured error responses
- Hungarian error messages
- Proper HTTP status codes
- Error codes for client-side handling
"""

from enum import StrEnum
from typing import Any

from fastapi import HTTPException, status

# =============================================================================
# Error Codes Enum
# =============================================================================


class ErrorCode(StrEnum):
    """Standardized error codes for client-side handling."""

    # General errors (1xxx)
    INTERNAL_ERROR = "ERR_1000"
    VALIDATION_ERROR = "ERR_1001"
    NOT_FOUND = "ERR_1002"
    UNAUTHORIZED = "ERR_1003"
    FORBIDDEN = "ERR_1004"
    RATE_LIMITED = "ERR_1005"
    REQUEST_TIMEOUT = "ERR_1006"
    BAD_REQUEST = "ERR_1007"

    # Database errors (2xxx)
    DATABASE_ERROR = "ERR_2000"
    DATABASE_CONNECTION = "ERR_2001"
    DATABASE_TIMEOUT = "ERR_2002"
    DATABASE_INTEGRITY = "ERR_2003"
    POSTGRES_ERROR = "ERR_2010"
    NEO4J_ERROR = "ERR_2020"
    NEO4J_CONNECTION = "ERR_2021"
    QDRANT_ERROR = "ERR_2030"
    QDRANT_CONNECTION = "ERR_2031"
    REDIS_ERROR = "ERR_2040"
    REDIS_CONNECTION = "ERR_2041"

    # External API errors (3xxx)
    EXTERNAL_API_ERROR = "ERR_3000"
    NHTSA_ERROR = "ERR_3001"
    NHTSA_RATE_LIMITED = "ERR_3002"
    NHTSA_TIMEOUT = "ERR_3003"
    LLM_ERROR = "ERR_3010"
    LLM_RATE_LIMITED = "ERR_3011"
    LLM_TIMEOUT = "ERR_3012"
    LLM_UNAVAILABLE = "ERR_3013"

    # Business logic errors (4xxx)
    DTC_VALIDATION_ERROR = "ERR_4001"
    VIN_VALIDATION_ERROR = "ERR_4002"
    VEHICLE_NOT_FOUND = "ERR_4003"
    DIAGNOSIS_ERROR = "ERR_4004"
    EMBEDDING_ERROR = "ERR_4005"
    RAG_ERROR = "ERR_4006"

    # Authentication errors (5xxx)
    AUTH_ERROR = "ERR_5000"
    INVALID_CREDENTIALS = "ERR_5001"
    TOKEN_EXPIRED = "ERR_5002"
    TOKEN_INVALID = "ERR_5003"
    REFRESH_TOKEN_INVALID = "ERR_5004"


# =============================================================================
# Hungarian Error Messages
# =============================================================================


ERROR_MESSAGES_HU: dict[ErrorCode, str] = {
    # General errors
    ErrorCode.INTERNAL_ERROR: "Belso szerverhiba tortent. Kerem, probalkozzon kesobb.",
    ErrorCode.VALIDATION_ERROR: "Ervenytelen adatok. Kerem, ellenorizze a bevitt adatokat.",
    ErrorCode.NOT_FOUND: "A keresett eroforras nem talalhato.",
    ErrorCode.UNAUTHORIZED: "Bejelentkezes szukseges a muvelet vegrehajtasahoz.",
    ErrorCode.FORBIDDEN: "Nincs jogosultsaga ehhez a muvelethez.",
    ErrorCode.RATE_LIMITED: "Tul sok keres. Kerem, varjon egy kicsit, majd probalkozzon ujra.",
    ErrorCode.REQUEST_TIMEOUT: "A keres idotullpes miatt megszakadt. Kerem, probalkozzon ujra.",
    ErrorCode.BAD_REQUEST: "Hibas keres formatum.",

    # Database errors
    ErrorCode.DATABASE_ERROR: "Adatbazis hiba tortent. Kerem, probalkozzon kesobb.",
    ErrorCode.DATABASE_CONNECTION: "Nem sikerult csatlakozni az adatbazishoz.",
    ErrorCode.DATABASE_TIMEOUT: "Az adatbazis kapcsolat idotullpest szenvedett.",
    ErrorCode.DATABASE_INTEGRITY: "Adatintegritasi hiba tortent.",
    ErrorCode.POSTGRES_ERROR: "PostgreSQL adatbazis hiba.",
    ErrorCode.NEO4J_ERROR: "Neo4j grafadatbazis hiba.",
    ErrorCode.NEO4J_CONNECTION: "Nem sikerult csatlakozni a Neo4j adatbazishoz.",
    ErrorCode.QDRANT_ERROR: "Qdrant vektor adatbazis hiba.",
    ErrorCode.QDRANT_CONNECTION: "Nem sikerult csatlakozni a Qdrant adatbazishoz.",
    ErrorCode.REDIS_ERROR: "Redis cache hiba.",
    ErrorCode.REDIS_CONNECTION: "Nem sikerult csatlakozni a Redis szerverhez.",

    # External API errors
    ErrorCode.EXTERNAL_API_ERROR: "Kulso szolgaltatas hiba. Kerem, probalkozzon kesobb.",
    ErrorCode.NHTSA_ERROR: "NHTSA szolgaltatas hiba.",
    ErrorCode.NHTSA_RATE_LIMITED: "NHTSA API korlat tullepve. Kerem, varjon egy kicsit.",
    ErrorCode.NHTSA_TIMEOUT: "NHTSA szolgaltatas nem valaszol.",
    ErrorCode.LLM_ERROR: "AI szolgaltatas hiba tortent.",
    ErrorCode.LLM_RATE_LIMITED: "AI szolgaltatas korlat tullepve. Kerem, varjon.",
    ErrorCode.LLM_TIMEOUT: "AI szolgaltatas nem valaszol.",
    ErrorCode.LLM_UNAVAILABLE: "AI szolgaltatas jelenleg nem elerheto.",

    # Business logic errors
    ErrorCode.DTC_VALIDATION_ERROR: "Ervenytelen DTC hibakod formatum.",
    ErrorCode.VIN_VALIDATION_ERROR: "Ervenytelen alvazszam (VIN) formatum.",
    ErrorCode.VEHICLE_NOT_FOUND: "A megadott jarmu nem talalhato.",
    ErrorCode.DIAGNOSIS_ERROR: "Hiba tortent a diagnosztika soran.",
    ErrorCode.EMBEDDING_ERROR: "Szovegfeldolgozasi hiba tortent.",
    ErrorCode.RAG_ERROR: "Tudasbazis keresesi hiba.",

    # Authentication errors
    ErrorCode.AUTH_ERROR: "Hitelesitesi hiba.",
    ErrorCode.INVALID_CREDENTIALS: "Hibas felhasznalonev vagy jelszo.",
    ErrorCode.TOKEN_EXPIRED: "A munkamenet lejart. Kerem, jelentkezzen be ujra.",
    ErrorCode.TOKEN_INVALID: "Ervenytelen token.",
    ErrorCode.REFRESH_TOKEN_INVALID: "Ervenytelen frissitesi token.",
}


def get_error_message(code: ErrorCode, fallback: str | None = None) -> str:
    """Get Hungarian error message for error code."""
    return ERROR_MESSAGES_HU.get(code, fallback or "Ismeretlen hiba tortent.")


# =============================================================================
# Base Exception Classes
# =============================================================================


class AutoCognitixException(Exception):
    """
    Base exception class for all AutoCognitix exceptions.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        details: Additional error context
        status_code: HTTP status code
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "message_hu": get_error_message(self.code, self.message),
                "details": self.details,
            }
        }

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_dict(),
        )


# =============================================================================
# Validation Exceptions
# =============================================================================


class ValidationException(AutoCognitixException):
    """Exception for validation errors."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field

        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            details=error_details,
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class DTCValidationException(ValidationException):
    """Exception for invalid DTC codes."""

    def __init__(
        self,
        message: str,
        invalid_codes: list[str] | None = None,
    ):
        details = {}
        if invalid_codes:
            details["invalid_codes"] = invalid_codes

        super().__init__(
            message=message,
            field="dtc_codes",
            details=details,
        )
        self.code = ErrorCode.DTC_VALIDATION_ERROR


class VINValidationException(ValidationException):
    """Exception for invalid VIN."""

    def __init__(
        self,
        message: str,
        vin: str | None = None,
    ):
        details = {}
        if vin:
            details["vin"] = vin

        super().__init__(
            message=message,
            field="vin",
            details=details,
        )
        self.code = ErrorCode.VIN_VALIDATION_ERROR


# =============================================================================
# Resource Exceptions
# =============================================================================


class NotFoundException(AutoCognitixException):
    """Exception for resource not found errors."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
    ):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            message=message,
            code=ErrorCode.NOT_FOUND,
            details=details,
            status_code=status.HTTP_404_NOT_FOUND,
        )


class VehicleNotFoundException(NotFoundException):
    """Exception when vehicle is not found."""

    def __init__(self, message: str = "A megadott jarmu nem talalhato."):
        super().__init__(
            message=message,
            resource_type="vehicle",
        )
        self.code = ErrorCode.VEHICLE_NOT_FOUND


# =============================================================================
# Database Exceptions
# =============================================================================


class DatabaseException(AutoCognitixException):
    """Base exception for database errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.DATABASE_ERROR,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        error_details = details or {}
        if original_error:
            error_details["original_error"] = str(original_error)

        super().__init__(
            message=message,
            code=code,
            details=error_details,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.original_error = original_error


class PostgresException(DatabaseException):
    """Exception for PostgreSQL errors."""

    def __init__(
        self,
        message: str = "PostgreSQL adatbazis hiba.",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.POSTGRES_ERROR,
            details=details,
            original_error=original_error,
        )


class PostgresConnectionException(PostgresException):
    """Exception for PostgreSQL connection errors."""

    def __init__(
        self,
        message: str = "Nem sikerult csatlakozni a PostgreSQL adatbazishoz.",
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            details={"type": "connection"},
            original_error=original_error,
        )


class Neo4jException(DatabaseException):
    """Exception for Neo4j errors."""

    def __init__(
        self,
        message: str = "Neo4j grafadatbazis hiba.",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.NEO4J_ERROR,
            details=details,
            original_error=original_error,
        )


class Neo4jConnectionException(Neo4jException):
    """Exception for Neo4j connection errors."""

    def __init__(
        self,
        message: str = "Nem sikerult csatlakozni a Neo4j adatbazishoz.",
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            original_error=original_error,
        )
        self.code = ErrorCode.NEO4J_CONNECTION


class QdrantException(DatabaseException):
    """Exception for Qdrant errors."""

    def __init__(
        self,
        message: str = "Qdrant vektor adatbazis hiba.",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.QDRANT_ERROR,
            details=details,
            original_error=original_error,
        )


class QdrantConnectionException(QdrantException):
    """Exception for Qdrant connection errors."""

    def __init__(
        self,
        message: str = "Nem sikerult csatlakozni a Qdrant adatbazishoz.",
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            original_error=original_error,
        )
        self.code = ErrorCode.QDRANT_CONNECTION


class RedisException(DatabaseException):
    """Exception for Redis errors."""

    def __init__(
        self,
        message: str = "Redis cache hiba.",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.REDIS_ERROR,
            details=details,
            original_error=original_error,
        )


class RedisConnectionException(RedisException):
    """Exception for Redis connection errors."""

    def __init__(
        self,
        message: str = "Nem sikerult csatlakozni a Redis szerverhez.",
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            original_error=original_error,
        )
        self.code = ErrorCode.REDIS_CONNECTION


# =============================================================================
# External API Exceptions
# =============================================================================


class ExternalAPIException(AutoCognitixException):
    """Base exception for external API errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.EXTERNAL_API_ERROR,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
        retry_after: int | None = None,
    ):
        error_details = details or {}
        if original_error:
            error_details["original_error"] = str(original_error)
        if retry_after:
            error_details["retry_after_seconds"] = retry_after

        super().__init__(
            message=message,
            code=code,
            details=error_details,
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
        self.original_error = original_error
        self.retry_after = retry_after


class NHTSAException(ExternalAPIException):
    """Exception for NHTSA API errors."""

    def __init__(
        self,
        message: str = "NHTSA szolgaltatas hiba.",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.NHTSA_ERROR,
            details=details,
            original_error=original_error,
        )


class NHTSARateLimitException(NHTSAException):
    """Exception when NHTSA rate limit is exceeded."""

    def __init__(
        self,
        retry_after: int = 60,
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=f"NHTSA API korlat tullepve. Varjon {retry_after} masodpercet.",
            details={"retry_after_seconds": retry_after},
            original_error=original_error,
        )
        self.code = ErrorCode.NHTSA_RATE_LIMITED
        self.retry_after = retry_after
        self.status_code = status.HTTP_429_TOO_MANY_REQUESTS


class LLMException(ExternalAPIException):
    """Exception for LLM service errors."""

    def __init__(
        self,
        message: str = "AI szolgaltatas hiba.",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.LLM_ERROR,
            details=details,
            original_error=original_error,
        )


class LLMRateLimitException(LLMException):
    """Exception when LLM rate limit is exceeded."""

    def __init__(
        self,
        retry_after: int = 60,
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=f"AI szolgaltatas korlat tullepve. Varjon {retry_after} masodpercet.",
            details={"retry_after_seconds": retry_after},
            original_error=original_error,
        )
        self.code = ErrorCode.LLM_RATE_LIMITED
        self.retry_after = retry_after
        self.status_code = status.HTTP_429_TOO_MANY_REQUESTS


class LLMUnavailableException(LLMException):
    """Exception when LLM service is unavailable."""

    def __init__(
        self,
        message: str = "AI szolgaltatas jelenleg nem elerheto.",
        original_error: Exception | None = None,
    ):
        super().__init__(
            message=message,
            original_error=original_error,
        )
        self.code = ErrorCode.LLM_UNAVAILABLE
        self.status_code = status.HTTP_503_SERVICE_UNAVAILABLE


# =============================================================================
# Business Logic Exceptions
# =============================================================================


class DiagnosisException(AutoCognitixException):
    """Exception for diagnosis service errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        error_details = details or {}
        if original_error:
            error_details["original_error"] = str(original_error)

        super().__init__(
            message=message,
            code=ErrorCode.DIAGNOSIS_ERROR,
            details=error_details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        self.original_error = original_error


class EmbeddingException(AutoCognitixException):
    """Exception for embedding service errors."""

    def __init__(
        self,
        message: str = "Szovegfeldolgozasi hiba.",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        error_details = details or {}
        if original_error:
            error_details["original_error"] = str(original_error)

        super().__init__(
            message=message,
            code=ErrorCode.EMBEDDING_ERROR,
            details=error_details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class RAGException(AutoCognitixException):
    """Exception for RAG service errors."""

    def __init__(
        self,
        message: str = "Tudazbazis keresesi hiba.",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        error_details = details or {}
        if original_error:
            error_details["original_error"] = str(original_error)

        super().__init__(
            message=message,
            code=ErrorCode.RAG_ERROR,
            details=error_details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# =============================================================================
# Authentication Exceptions
# =============================================================================


class AuthenticationException(AutoCognitixException):
    """Base exception for authentication errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.AUTH_ERROR,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            code=code,
            details=details,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class InvalidCredentialsException(AuthenticationException):
    """Exception for invalid login credentials."""

    def __init__(
        self,
        message: str = "Hibas felhasznalonev vagy jelszo.",
    ):
        super().__init__(
            message=message,
            code=ErrorCode.INVALID_CREDENTIALS,
        )


class TokenExpiredException(AuthenticationException):
    """Exception for expired tokens."""

    def __init__(
        self,
        message: str = "A munkamenet lejart. Kerem, jelentkezzen be ujra.",
    ):
        super().__init__(
            message=message,
            code=ErrorCode.TOKEN_EXPIRED,
        )


class InvalidTokenException(AuthenticationException):
    """Exception for invalid tokens."""

    def __init__(
        self,
        message: str = "Ervenytelen token.",
    ):
        super().__init__(
            message=message,
            code=ErrorCode.TOKEN_INVALID,
        )


class ForbiddenException(AutoCognitixException):
    """Exception for forbidden access."""

    def __init__(
        self,
        message: str = "Nincs jogosultsaga ehhez a muvelethez.",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.FORBIDDEN,
            details=details,
            status_code=status.HTTP_403_FORBIDDEN,
        )


# =============================================================================
# Rate Limiting Exception
# =============================================================================


class RateLimitException(AutoCognitixException):
    """Exception when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Tul sok keres. Kerem, varjon egy kicsit.",
        retry_after: int = 60,
    ):
        super().__init__(
            message=message,
            code=ErrorCode.RATE_LIMITED,
            details={"retry_after_seconds": retry_after},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )
        self.retry_after = retry_after
