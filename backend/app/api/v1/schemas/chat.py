"""
Chat schemas - AI chat assistant request/response models.

Provides schemas for the interactive chat interface with SSE streaming support.
"""

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class VehicleContext(BaseModel):
    """Vehicle context for chat messages."""

    make: str = Field(..., min_length=1, max_length=100, description="Vehicle manufacturer")
    model: str = Field(..., min_length=1, max_length=100, description="Vehicle model")
    year: int = Field(..., ge=1900, le=2030, description="Vehicle year")
    dtc_codes: Optional[List[str]] = Field(
        None, max_length=20, description="Optional list of DTC codes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "make": "Volkswagen",
                "model": "Golf",
                "year": 2018,
                "dtc_codes": ["P0300", "P0301"],
            }
        }


# Prompt injection markers to reject
_INJECTION_MARKERS = [
    "SYSTEM:",
    "IGNORE ALL",
    "OVERRIDE",
]

_INJECTION_PATTERN = re.compile(
    "|".join(re.escape(marker) for marker in _INJECTION_MARKERS),
    re.IGNORECASE,
)


class ChatRequest(BaseModel):
    """Request schema for chat message."""

    message: str = Field(..., min_length=1, max_length=1000, description="User message text")
    conversation_id: Optional[str] = Field(
        None, max_length=64, description="Existing conversation ID to continue"
    )
    vehicle_context: Optional[VehicleContext] = Field(
        None, description="Optional vehicle context for the conversation"
    )
    diagnosis_id: Optional[str] = Field(
        None, max_length=64, description="Optional diagnosis ID for context"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Strip whitespace and check for prompt injection markers."""
        v = v.strip()
        if not v:
            msg = "Message cannot be empty after stripping whitespace"
            raise ValueError(msg)
        if _INJECTION_PATTERN.search(v):
            msg = "Message contains disallowed content"
            raise ValueError(msg)
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Mi okozhatja a P0300 hibakodot?",
                "conversation_id": None,
                "vehicle_context": {
                    "make": "Volkswagen",
                    "model": "Golf",
                    "year": 2018,
                    "dtc_codes": ["P0300"],
                },
                "diagnosis_id": None,
            }
        }


class ChatSource(BaseModel):
    """Schema for a chat response source reference."""

    title: str = Field(..., description="Source title")
    type: str = Field(..., description="Source type (database, recall, complaint, etc.)")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score (0-1)")

    @field_validator("relevance_score")
    @classmethod
    def validate_relevance_score(cls, v: float) -> float:
        """Clamp relevance score to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))


class ChatMessage(BaseModel):
    """Schema for a single chat message."""

    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp",
    )
    sources: Optional[List[ChatSource]] = Field(
        None, description="Source references for assistant messages"
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Ensure role is either user or assistant."""
        if v not in ("user", "assistant"):
            msg = "Role must be 'user' or 'assistant'"
            raise ValueError(msg)
        return v


class ChatStreamEvent(BaseModel):
    """Schema for SSE streaming chat events."""

    event_type: str = Field(
        ...,
        description="Event type: start, token, source, suggestion, complete, error",
    )
    data: Dict = Field(default_factory=dict, description="Event data payload")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        description="Event timestamp in ISO format",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "token",
                "data": {"content": "A P0300 hibakod"},
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-03-20T10:30:00Z",
            }
        }
