"""
Security utilities for authentication and authorization.

Provides JWT token management, password hashing, and token blacklisting.
"""

import hashlib
import hmac
import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError, DecodeError
from passlib.context import CryptContext

from app.core.config import settings
from app.core.log_sanitizer import sanitize_log

# PyJWT encode returns str, decode returns dict
# passlib CryptContext.verify/hash also return Any

logger = logging.getLogger(__name__)

# Password hashing context with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Clock-skew leeway (seconds) applied when decoding tokens. A blacklisted
# token stays acceptable to decode_token until exp + leeway, so blacklist
# entries must live at least this long past exp to close the replay window.
JWT_DECODE_LEEWAY = 10


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


async def decode_token(token: str, expected_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    Decode and verify a JWT token.

    Args:
        token: The JWT token to decode
        expected_type: Expected token type claim (default "access").
            Prevents token type confusion attacks (e.g. using a
            refresh token as an access token).

    Returns:
        The decoded token payload or None if invalid
    """
    try:
        payload: Dict[str, Any] = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            leeway=JWT_DECODE_LEEWAY,
        )

        # Validate token type to prevent type confusion attacks
        if payload.get("type") != expected_type:
            logger.warning(
                f"Token type mismatch: expected={expected_type}, got={payload.get('type')}"
            )
            return None

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

        # Calculate TTL based on token expiration, extended by the decode
        # leeway. decode_token accepts tokens up to JWT_DECODE_LEEWAY seconds
        # past exp, so the blacklist entry must outlive that window; otherwise
        # a revoked token is replayable in [exp, exp + leeway].
        exp = payload.get("exp", 0)
        ttl = exp + JWT_DECODE_LEEWAY - int(datetime.now(timezone.utc).timestamp())

        # Only store if the token can still be accepted (within exp + leeway)
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

    Fail-closed: if Redis is unavailable or the circuit breaker is open,
    the token is rejected to prevent use of potentially blacklisted tokens.

    Args:
        jti: The JWT ID to check

    Returns:
        True if blacklisted (or Redis unavailable), False otherwise
    """
    try:
        from app.db.redis_cache import get_cache_service

        cache = await get_cache_service()

        # If circuit breaker is open, Redis is considered unavailable — reject token
        if cache.is_circuit_open():
            logger.warning("Redis circuit breaker open — rejecting token (fail-closed)")
            return True  # fail-closed

        result = await cache.get(f"blacklist:{jti}")
        return result is not None
    except Exception as e:
        logger.error(f"Token blacklist check failed: {sanitize_log(str(e))}")
        return True  # fail-closed


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


PASSWORD_MIN_LENGTH = 8
PASSWORD_PATTERN = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$")


def validate_password_strength(password: str) -> str:
    """
    Validate password strength. Returns password if valid, raises ValueError.

    Requirements:
    - At least 8 characters
    - At most 100 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character

    Args:
        password: The password to validate

    Returns:
        The original password if all requirements are met

    Raises:
        ValueError: If any requirement is not met
    """
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"A jelszónak legalább {PASSWORD_MIN_LENGTH} karakter hosszúnak kell lennie."
        )

    if len(password) > 100:
        raise ValueError("A jelszó maximum 100 karakter hosszú lehet.")

    if not re.search(r"[a-z]", password):
        raise ValueError("A jelszónak tartalmaznia kell legalább egy kisbetűt.")

    if not re.search(r"[A-Z]", password):
        raise ValueError("A jelszónak tartalmaznia kell legalább egy nagybetűt.")

    if not re.search(r"\d", password):
        raise ValueError("A jelszónak tartalmaznia kell legalább egy számot.")

    if not re.search(r"[!@#$%^&*()\\_+\-=\[\]{}|;:,.<>?]", password):
        raise ValueError(
            "A jelszónak tartalmaznia kell legalább egy speciális karaktert"
            " (!@#$%^&*()_+-=[]{}|;:,.<>?)."
        )

    return password


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.

    Args:
        length: The length of the token in bytes (default 32)

    Returns:
        A URL-safe base64 encoded token string
    """
    return secrets.token_urlsafe(length)


def generate_csrf_token() -> str:
    """
    Generate an HMAC-signed CSRF token (header-only, no cookie).

    Produces a random nonce signed with HMAC-SHA256 using
    settings.JWT_SECRET_KEY. The token is returned in the login
    response body and sent by the frontend as an X-CSRF-Token header
    on state-changing requests. No cookie is involved, so it works
    across cross-site deployments where samesite cookies are dropped.

    Returns:
        A "<nonce>.<mac>" signed CSRF token string
    """
    nonce = secrets.token_urlsafe(32)
    mac = hmac.new(settings.JWT_SECRET_KEY.encode(), nonce.encode(), hashlib.sha256).hexdigest()
    return f"{nonce}.{mac}"


def verify_csrf_token(token: Optional[str]) -> bool:
    """
    Verify an HMAC-signed CSRF token.

    Recomputes the HMAC-SHA256 signature of the nonce using
    settings.JWT_SECRET_KEY and compares it against the provided MAC
    in constant time. Header-only, no cookie is consulted.

    Args:
        token: The "<nonce>.<mac>" CSRF token from the X-CSRF-Token header

    Returns:
        True if the token is well-formed and its signature is valid
    """
    # A well-formed token is always ASCII ("<nonce>.<hexdigest>"). Reject
    # non-ASCII early: hmac.compare_digest raises TypeError on non-ASCII str
    # (Starlette decodes headers as latin-1), which would surface as a 500
    # instead of a clean 403.
    if not token or "." not in token or not token.isascii():
        return False
    nonce, _, provided_mac = token.partition(".")
    if not nonce or not provided_mac:
        return False
    expected_mac = hmac.new(
        settings.JWT_SECRET_KEY.encode(), nonce.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(provided_mac, expected_mac)
