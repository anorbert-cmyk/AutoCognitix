"""
Authentication schemas.

Provides Pydantic models for authentication requests and responses.
"""

import re
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

# =============================================================================
# User Role Types
# =============================================================================

UserRole = Literal["user", "mechanic", "admin"]


# =============================================================================
# Password Validation
# =============================================================================

def validate_password_strength(password: str) -> str:
    """
    Validate password strength.

    Requirements:
    - At least 8 characters
    - At most 100 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    """
    if len(password) < 8:
        raise ValueError("A jelszónak legalább 8 karakter hosszúnak kell lennie")

    if len(password) > 100:
        raise ValueError("A jelszó maximum 100 karakter hosszú lehet")

    if not re.search(r"[A-Z]", password):
        raise ValueError("A jelszónak tartalmaznia kell legalább egy nagybetűt")

    if not re.search(r"[a-z]", password):
        raise ValueError("A jelszónak tartalmaznia kell legalább egy kisbetűt")

    if not re.search(r"\d", password):
        raise ValueError("A jelszónak tartalmaznia kell legalább egy számot")

    if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]", password):
        raise ValueError(
            "A jelszónak tartalmaznia kell legalább egy speciális karaktert (!@#$%^&*()_+-=[]{}|;:,.<>?)"
        )

    return password


# =============================================================================
# User Registration & Authentication
# =============================================================================

class UserCreate(BaseModel):
    """Schema for user registration."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str | None = Field(None, max_length=100)
    role: UserRole = Field(default="user", description="User role (default: user)")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        return validate_password_strength(v)


class UserLogin(BaseModel):
    """Schema for user login (alternative to OAuth2 form)."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response."""

    id: str
    email: EmailStr
    full_name: str | None = None
    is_active: bool = True
    role: UserRole = "user"
    created_at: str | None = None

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for updating user profile."""

    full_name: str | None = Field(None, max_length=100)
    email: EmailStr | None = None


class UserPasswordUpdate(BaseModel):
    """Schema for updating user password."""

    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        return validate_password_strength(v)


# =============================================================================
# Token Schemas
# =============================================================================

class Token(BaseModel):
    """Schema for JWT token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Schema for token refresh request."""

    refresh_token: str


class TokenPayload(BaseModel):
    """Schema for decoded JWT token payload."""

    sub: str
    exp: int
    type: str
    jti: str | None = None
    role: str | None = None


# =============================================================================
# Password Reset Schemas
# =============================================================================

class ForgotPasswordRequest(BaseModel):
    """Schema for forgot password request."""

    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    """Schema for forgot password response."""

    message: str = "Ha az email cím létezik, elküldtük a jelszó visszaállítási linket"


class ResetPasswordRequest(BaseModel):
    """Schema for reset password request."""

    token: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        return validate_password_strength(v)


class ResetPasswordResponse(BaseModel):
    """Schema for reset password response."""

    message: str = "A jelszó sikeresen megváltozott"


# =============================================================================
# Logout Schema
# =============================================================================

class LogoutRequest(BaseModel):
    """Schema for logout request (optional refresh token)."""

    refresh_token: str | None = None


class LogoutResponse(BaseModel):
    """Schema for logout response."""

    message: str = "Sikeres kijelentkezés"


# =============================================================================
# Error Responses
# =============================================================================

class AuthErrorResponse(BaseModel):
    """Schema for authentication error response."""

    detail: str
    error_code: str | None = None


class ValidationErrorResponse(BaseModel):
    """Schema for validation error response."""

    detail: list[dict]
