"""
Authentication schemas.
"""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for user registration."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)


class UserResponse(BaseModel):
    """Schema for user response."""

    id: str
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    role: str = "user"

    class Config:
        from_attributes = True


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
