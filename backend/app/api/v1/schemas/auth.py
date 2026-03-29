"""
Authentication schemas.

Provides Pydantic models for authentication requests and responses.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.security import validate_password_strength

# =============================================================================
# User Role Types
# =============================================================================

UserRole = Literal["user", "mechanic", "admin"]


# =============================================================================
# User Registration & Authentication
# =============================================================================


class UserCreate(BaseModel):
    """Schema for user registration."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)
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
    full_name: Optional[str] = None
    is_active: bool = True
    role: UserRole = "user"
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for updating user profile."""

    full_name: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = None


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
    csrf_token: Optional[str] = None


class TokenRefresh(BaseModel):
    """Schema for token refresh request.

    The refresh_token field is optional because browser clients
    send the token via httpOnly cookie instead of the request body.
    """

    refresh_token: Optional[str] = None


class TokenPayload(BaseModel):
    """Schema for decoded JWT token payload."""

    sub: str
    exp: int
    type: str
    jti: Optional[str] = None
    role: Optional[str] = None


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

    refresh_token: Optional[str] = None


class LogoutResponse(BaseModel):
    """Schema for logout response."""

    message: str = "Sikeres kijelentkezés"


# =============================================================================
# Error Responses
# =============================================================================


class AuthErrorResponse(BaseModel):
    """Schema for authentication error response."""

    detail: str
    error_code: Optional[str] = None


class ValidationErrorResponse(BaseModel):
    """Schema for validation error response."""

    detail: List[dict]
