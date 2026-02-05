"""
Security utilities for authentication and authorization.

Provides JWT token management, password hashing, and token blacklisting.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set, Union

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# Python 3.11+ has datetime.UTC, for older versions use timezone.utc
UTC = timezone.utc

# Password hashing context with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory token blacklist (in production, use Redis)
# This stores JTI (JWT ID) of invalidated tokens
_token_blacklist: Set[str] = set()


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        subject: The subject of the token (usually user ID)
        expires_delta: Optional custom expiration time
        additional_claims: Optional additional claims to include

    Returns:
        Encoded JWT token string
    """
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "exp": expire,
        "iat": datetime.now(UTC),
        "sub": str(subject),
        "type": "access",
        "jti": secrets.token_urlsafe(16),  # JWT ID for blacklisting
    }

    if additional_claims:
        to_encode.update(additional_claims)

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return encoded_jwt


def create_refresh_token(
    subject: str | Any,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        subject: The subject of the token (usually user ID)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT refresh token string
    """
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )

    to_encode = {
        "exp": expire,
        "iat": datetime.now(UTC),
        "sub": str(subject),
        "type": "refresh",
        "jti": secrets.token_urlsafe(16),  # JWT ID for blacklisting
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return encoded_jwt


def create_password_reset_token(
    subject: str | Any,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a JWT password reset token.

    Args:
        subject: The subject of the token (usually user email)
        expires_delta: Optional custom expiration time (default 1 hour)

    Returns:
        Encoded JWT password reset token string
    """
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(hours=1)

    to_encode = {
        "exp": expire,
        "iat": datetime.now(UTC),
        "sub": str(subject),
        "type": "password_reset",
        "jti": secrets.token_urlsafe(16),
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)


def decode_token(token: str) -> dict[str, Any] | None:
    """
    Decode and verify a JWT token.

    Args:
        token: The JWT token to decode

    Returns:
        The decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )

        # Check if token is blacklisted
        jti = payload.get("jti")
        if jti and is_token_blacklisted(jti):
            return None

        return payload
    except JWTError:
        return None


def blacklist_token(token: str) -> bool:
    """
    Add a token to the blacklist.

    Args:
        token: The JWT token to blacklist

    Returns:
        True if successfully blacklisted, False otherwise
    """
    try:
        # Decode without verification to get JTI even if expired
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False},
        )
        jti = payload.get("jti")
        if jti:
            _token_blacklist.add(jti)
            return True
        return False
    except JWTError:
        return False


def is_token_blacklisted(jti: str) -> bool:
    """
    Check if a token JTI is blacklisted.

    Args:
        jti: The JWT ID to check

    Returns:
        True if blacklisted, False otherwise
    """
    return jti in _token_blacklist


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """
    Validate password strength.

    Args:
        password: The password to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if len(password) < 8:
        errors.append("A jelszónak legalább 8 karakter hosszúnak kell lennie")

    if len(password) > 100:
        errors.append("A jelszó maximum 100 karakter hosszú lehet")

    if not any(c.isupper() for c in password):
        errors.append("A jelszónak tartalmaznia kell legalább egy nagybetűt")

    if not any(c.islower() for c in password):
        errors.append("A jelszónak tartalmaznia kell legalább egy kisbetűt")

    if not any(c.isdigit() for c in password):
        errors.append("A jelszónak tartalmaznia kell legalább egy számot")

    # Check for special characters
    special_chars = set("!@#$%^&*()_+-=[]{}|;:,.<>?")
    if not any(c in special_chars for c in password):
        errors.append("A jelszónak tartalmaznia kell legalább egy speciális karaktert (!@#$%^&*()_+-=[]{}|;:,.<>?)")

    return len(errors) == 0, errors


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.

    Args:
        length: The length of the token in bytes (default 32)

    Returns:
        A URL-safe base64 encoded token string
    """
    return secrets.token_urlsafe(length)
