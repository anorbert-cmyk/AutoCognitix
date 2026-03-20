"""
Security utilities for authentication and authorization.

Provides JWT token management, password hashing, and token blacklisting.
"""

import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError, DecodeError
from passlib.context import CryptContext

from app.core.config import settings

# PyJWT encode returns str, decode returns dict
# passlib CryptContext.verify/hash also return Any

logger = logging.getLogger(__name__)

# Password hashing context with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "sub": str(subject),
        "type": "access",
        "jti": secrets.token_urlsafe(16),  # JWT ID for blacklisting
    }

    if additional_claims:
        # Prevent overwriting critical JWT claims (sub, exp, iat, type, jti)
        protected = {"exp", "iat", "sub", "type", "jti"}
        safe_claims = {k: v for k, v in additional_claims.items() if k not in protected}
        to_encode.update(safe_claims)

    encoded_jwt: str = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return encoded_jwt


def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
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
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "sub": str(subject),
        "type": "refresh",
        "jti": secrets.token_urlsafe(16),  # JWT ID for blacklisting
    }

    encoded_jwt: str = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return encoded_jwt


def create_password_reset_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
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
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=1)

    to_encode = {
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "sub": str(subject),
        "type": "password_reset",
        "jti": secrets.token_urlsafe(16),
    }

    encoded_jwt: str = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    result: bool = pwd_context.verify(plain_password, hashed_password)
    return result


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    hashed: str = pwd_context.hash(password)
    return hashed


async def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and verify a JWT token.

    Args:
        token: The JWT token to decode

    Returns:
        The decoded token payload or None if invalid
    """
    try:
        payload: Dict[str, Any] = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )

        # Check if token is blacklisted
        jti = payload.get("jti")
        if jti and await is_token_blacklisted(jti):
            return None

        return payload
    except (InvalidTokenError, ExpiredSignatureError, DecodeError):
        return None


async def blacklist_token(token: str) -> bool:
    """
    Add a token to the Redis blacklist with TTL matching token expiry.

    This prevents the token from being used again while avoiding
    storing expired tokens indefinitely.

    Args:
        token: The JWT token to blacklist

    Returns:
        True if successfully blacklisted, False otherwise
    """
    try:
        # Decode without verification to get JTI and expiration
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False},
        )
        jti = payload.get("jti")
        if not jti:
            logger.warning("Token missing JTI claim - cannot blacklist")
            return False

        # Calculate TTL based on token expiration
        exp = payload.get("exp", 0)
        ttl = max(0, exp - int(datetime.now(timezone.utc).timestamp()))

        # Only store if token hasn't expired yet
        if ttl > 0:
            try:
                from app.db.redis_cache import get_cache_service

                cache = await get_cache_service()
                await cache.set(f"blacklist:{jti}", "1", ttl=ttl)
                logger.info(f"Token blacklisted: {jti[:8]}... (TTL: {ttl}s)")
                return True
            except Exception as e:
                logger.error(f"Failed to blacklist token in Redis: {e}")
                return False
        else:
            logger.debug(f"Token already expired, skipping blacklist: {jti[:8]}...")
            return True  # Expired tokens are effectively blacklisted

    except (InvalidTokenError, ExpiredSignatureError, DecodeError) as e:
        logger.warning(f"Failed to decode token for blacklisting: {e}")
        return False


async def is_token_blacklisted(jti: str) -> bool:
    """
    Check if a token JTI is in the Redis blacklist.

    Args:
        jti: The JWT ID to check

    Returns:
        True if blacklisted, False otherwise
    """
    try:
        from app.db.redis_cache import get_cache_service

        cache = await get_cache_service()

        # If circuit breaker is open, fail closed (reject token)
        if cache.is_circuit_open():
            logger.critical("Redis circuit breaker open - rejecting token for safety")
            return True

        result = await cache.get(f"blacklist:{jti}")
        return result is not None
    except Exception as e:
        logger.critical(f"Token blacklist check failed - rejecting token for safety: {e}")
        return True


def check_password_strength(password: str) -> Dict[str, Any]:
    """
    Check password strength and return detailed results.

    Returns a dict with score (0-5), requirement statuses, and Hungarian feedback.

    Args:
        password: The password to check

    Returns:
        Dict with is_strong, score, requirements, and feedback_hu
    """
    special_chars = re.compile(r"[!@#$%^&*()\\_+\-=\[\]{}|;:,.<>?]")

    requirements = {
        "min_length": len(password) >= 8,
        "has_uppercase": bool(re.search(r"[A-Z]", password)),
        "has_lowercase": bool(re.search(r"[a-z]", password)),
        "has_digit": bool(re.search(r"[0-9]", password)),
        "has_special": bool(special_chars.search(password)),
    }

    score = sum(1 for v in requirements.values() if v)

    feedback_map: Dict[int, str] = {
        0: "Nagyon gyenge jelszo - egyik kovetelmeny sem teljesul",
        1: "Gyenge jelszo - tobb kovetelmeny teljesitese szukseges",
        2: "Gyenge jelszo - legalabb 3 kovetelmeny teljesitese szukseges",
        3: "Kozepes jelszo - meg erosebb lenne tobb kovetelmeny teljesitesevel",
        4: "Eros jelszo - majdnem minden kovetelmeny teljesul",
        5: "Nagyon eros jelszo - minden kovetelmeny teljesul",
    }

    return {
        "is_strong": score >= 3,
        "score": score,
        "requirements": requirements,
        "feedback_hu": feedback_map[score],
    }


def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
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
        errors.append(
            "A jelszónak tartalmaznia kell legalább egy speciális karaktert (!@#$%^&*()_+-=[]{}|;:,.<>?)"
        )

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
