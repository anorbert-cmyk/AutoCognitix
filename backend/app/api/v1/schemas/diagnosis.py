"""
Diagnosis schemas - core diagnostic request/response models.
"""

import re
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.vehicle_makes import normalize_make

# Standard OBD-II DTC code shape: one system letter (P/B/C/U) + 4 hex digits
# (e.g. "P0300", "U0100"). Kept permissive on purpose - it only guarantees the
# canonical 5-character shape so raw strings cannot flow unbounded into Neo4j
# lookups, RAG cache keys and logs.
_DTC_CODE_PATTERN = re.compile(r"^[PBCU][0-9A-F]{4}$")


def normalize_dtc_codes(codes: List[str]) -> List[str]:
    """Normalize (strip + upper) and validate DTC codes at the API boundary.

    Every downstream consumer (Neo4j graph lookup, RAG cache key, log line)
    receives the same canonical uppercase form, so "p0300 " and "P0300" no
    longer cause cache misses or failed graph lookups. Malformed items are
    rejected here instead of silently reaching those code paths.
    """
    normalized: List[str] = []
    for code in codes:
        canonical = code.strip().upper()
        if not _DTC_CODE_PATTERN.match(canonical):
            raise ValueError(
                f"Invalid DTC code format: {code!r} "
                "(expected e.g. 'P0300', one of P/B/C/U followed by 4 hex digits)"
            )
        normalized.append(canonical)
    return normalized


class DiagnosisRequest(BaseModel):
    """Request schema for vehicle diagnosis."""

    vehicle_make: str = Field(..., min_length=1, max_length=100, description="Vehicle manufacturer")
    vehicle_model: str = Field(..., min_length=1, max_length=100, description="Vehicle model")
    vehicle_year: int = Field(..., ge=1900, le=2030, description="Vehicle year")
    vehicle_engine: Optional[str] = Field(None, max_length=100, description="Engine type/code")
    vin: Optional[str] = Field(
        None, min_length=17, max_length=17, description="Vehicle Identification Number"
    )

    dtc_codes: List[str] = Field(..., min_length=1, max_length=20, description="List of DTC codes")
    symptoms: str = Field(
        ..., min_length=10, max_length=2000, description="Symptom description in Hungarian"
    )
    additional_context: Optional[str] = Field(
        None, max_length=1000, description="Additional context"
    )

    @field_validator("vehicle_make")
    @classmethod
    def normalize_vehicle_make(cls, v: str) -> str:
        """Canonicalize the make once at the API boundary.

        Every downstream consumer (Qdrant filter, PostgreSQL KnownIssue
        filter, LLM prompt, NHTSA API, diagnosis history) receives the same
        canonical NHTSA spelling ("vw" -> "Volkswagen", "bmw" -> "BMW").
        Unknown makes pass through unchanged.
        """
        normalized = normalize_make(v)
        if not normalized:
            # Whitespace-only input passes min_length=1 before this validator
            # runs; an empty make would silently disable every make filter.
            raise ValueError("vehicle_make must not be blank")
        return normalized

    @field_validator("dtc_codes")
    @classmethod
    def normalize_dtc_codes(cls, v: List[str]) -> List[str]:
        """Normalize and validate DTC codes at the API boundary."""
        return normalize_dtc_codes(v)

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "vehicle_engine": "2.0 TSI",
                "dtc_codes": ["P0101", "P0171"],
                "symptoms": "A motor nehezen indul hidegben, egyenetlenül jár alapjáraton, és a fogyasztás megnőtt.",
                "additional_context": "A probléma télen rosszabb.",
            }
        }


class ProbableCause(BaseModel):
    """Schema for a probable cause in diagnosis."""

    title: str = Field(..., description="Short title of the cause")
    description: str = Field(..., description="Detailed description in Hungarian")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    related_dtc_codes: List[str] = Field(default_factory=list, description="Related DTC codes")
    components: List[str] = Field(default_factory=list, description="Affected components")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Clamp confidence score to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))


class ToolNeeded(BaseModel):
    """Schema for a tool needed for a repair step."""

    name: str = Field(..., description="Tool name in Hungarian")
    icon_hint: str = Field("handyman", description="Material Symbols icon name hint")


class PartWithPrice(BaseModel):
    """Schema for a part with price information."""

    id: str = Field(..., description="Part identifier key")
    name: str = Field(..., description="Part name in Hungarian")
    name_en: Optional[str] = Field(None, description="Part name in English")
    category: str = Field("other", description="Part category")
    price_range_min: int = Field(0, ge=0, description="Minimum price in HUF")
    price_range_max: int = Field(0, ge=0, description="Maximum price in HUF")
    labor_hours: float = Field(0.0, ge=0, description="Estimated labor hours")
    currency: str = Field("HUF", description="Currency code")


class TotalCostEstimate(BaseModel):
    """Schema for total repair cost estimate breakdown."""

    parts_min: int = Field(0, ge=0, description="Minimum parts cost")
    parts_max: int = Field(0, ge=0, description="Maximum parts cost")
    labor_min: int = Field(0, ge=0, description="Minimum labor cost")
    labor_max: int = Field(0, ge=0, description="Maximum labor cost")
    total_min: int = Field(0, ge=0, description="Total minimum cost")
    total_max: int = Field(0, ge=0, description="Total maximum cost")
    currency: str = Field("HUF", description="Currency code")
    estimated_hours: float = Field(0.0, ge=0, description="Total estimated labor hours")
    difficulty: str = Field("medium", description="Overall difficulty level")
    disclaimer: str = Field("", description="Price estimate disclaimer")

    @model_validator(mode="after")
    def validate_min_max_ranges(self) -> "TotalCostEstimate":
        """Ensure min values do not exceed max values."""
        if self.parts_min > self.parts_max:
            msg = f"parts_min ({self.parts_min}) must be <= parts_max ({self.parts_max})"
            raise ValueError(msg)
        if self.labor_min > self.labor_max:
            msg = f"labor_min ({self.labor_min}) must be <= labor_max ({self.labor_max})"
            raise ValueError(msg)
        if self.total_min > self.total_max:
            msg = f"total_min ({self.total_min}) must be <= total_max ({self.total_max})"
            raise ValueError(msg)
        return self


class RepairRecommendation(BaseModel):
    """Schema for repair recommendation."""

    title: str = Field(..., description="Short title of the repair")
    description: str = Field(..., description="Detailed repair instructions in Hungarian")
    estimated_cost_min: Optional[int] = Field(None, ge=0, description="Minimum estimated cost")
    estimated_cost_max: Optional[int] = Field(None, ge=0, description="Maximum estimated cost")
    estimated_cost_currency: str = Field("HUF", description="Currency code")
    difficulty: str = Field(
        "intermediate",
        description="Difficulty level: beginner, intermediate, advanced, professional",
    )
    parts_needed: List[str] = Field(default_factory=list, description="List of parts needed")
    estimated_time_minutes: Optional[int] = Field(
        None, ge=0, description="Estimated repair time in minutes"
    )
    tools_needed: List[ToolNeeded] = Field(
        default_factory=list, description="Tools needed for this repair step"
    )
    expert_tips: List[str] = Field(
        default_factory=list, description="Expert tips for this repair step"
    )
    root_cause_explanation: Optional[str] = Field(
        None, description="Why this repair is needed - root cause analysis"
    )


class Source(BaseModel):
    """Schema for information source."""

    type: str = Field(..., description="Source type: tsb, forum, video, manual, database")
    title: str = Field(..., description="Source title")
    url: Optional[str] = Field(None, description="Source URL if available")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score (0-1)")

    @field_validator("relevance_score")
    @classmethod
    def validate_relevance_score(cls, v: float) -> float:
        """Clamp relevance score to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))


class RelatedRecall(BaseModel):
    """Schema for related NHTSA recall information."""

    campaign_number: str = Field(..., description="NHTSA campaign number")
    component: str = Field(..., description="Affected component")
    summary: str = Field(..., description="Recall summary")
    consequence: Optional[str] = Field(None, description="Potential consequence")
    remedy: Optional[str] = Field(None, description="Recommended remedy")
    recall_date: Optional[str] = Field(None, description="Recall date")


class RelatedComplaint(BaseModel):
    """Schema for related NHTSA complaint information."""

    odi_number: Optional[str] = Field(None, description="ODI complaint number")
    components: str = Field(..., description="Affected components")
    summary: str = Field(..., description="Complaint summary")
    crash: bool = Field(False, description="Whether crash was involved")
    fire: bool = Field(False, description="Whether fire was involved")
    similarity_score: float = Field(..., ge=0, le=1, description="Similarity to current issue")

    @field_validator("similarity_score")
    @classmethod
    def validate_similarity_score(cls, v: float) -> float:
        """Clamp similarity score to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))


class UrgencyLevel(str, Enum):
    """Urgency level for diagnosis."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DiagnosisResponse(BaseModel):
    """Response schema for vehicle diagnosis."""

    id: Optional[UUID] = Field(None, description="Unique diagnosis ID (None if not persisted)")
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    dtc_codes: List[str]
    symptoms: str

    probable_causes: List[ProbableCause] = Field(..., description="Ranked list of probable causes")
    recommended_repairs: List[RepairRecommendation] = Field(..., description="Recommended repairs")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall diagnosis confidence")
    sources: List[Source] = Field(default_factory=list, description="Information sources used")

    # Enhanced fields for recalls and complaints
    similar_complaints: List[RelatedComplaint] = Field(
        default_factory=list, description="Similar NHTSA complaints found for this vehicle"
    )
    related_recalls: List[RelatedRecall] = Field(
        default_factory=list, description="Related NHTSA recalls for this vehicle"
    )

    # Urgency and safety
    urgency_level: str = Field(
        default="medium", description="Urgency level: low, medium, high, critical"
    )
    safety_warnings: List[str] = Field(
        default_factory=list, description="Safety warnings if any critical issues detected"
    )

    # Diagnostic steps
    diagnostic_steps: List[str] = Field(
        default_factory=list, description="Recommended diagnostic steps"
    )

    # Parts and cost information
    parts_with_prices: List[PartWithPrice] = Field(
        default_factory=list, description="Parts with Hungarian price ranges"
    )
    total_cost_estimate: Optional[TotalCostEstimate] = Field(
        None, description="Total repair cost estimate"
    )
    root_cause_analysis: Optional[str] = Field(
        None, description="Detailed root cause analysis paragraph"
    )

    # Processing metadata
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    model_used: Optional[str] = Field(None, description="AI model used for diagnosis")
    save_error: bool = Field(
        default=False,
        description="True if diagnosis was generated but could not be persisted to database",
    )
    used_fallback: bool = Field(False, description="Whether fallback diagnosis was used")

    # AI disclaimer (EU AI Act + liability protection)
    ai_disclaimer: str = Field(
        default=(
            "Ez az elemzés mesterséges intelligencia által készült, tájékoztató jellegű. "
            "NEM helyettesíti a szakképzett szerelő véleményét. "
            "Minden javítás előtt kérjen szakemberi diagnózist. "
            "Az AutoCognitix nem vállal felelősséget az AI alapú ajánlások alapján végzett munkákért."
        ),
        description="AI-generated diagnosis disclaimer (EU AI Act compliance)",
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Clamp confidence score to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))

    class Config:
        from_attributes = True


class DiagnosisHistoryItem(BaseModel):
    """Schema for diagnosis history list item."""

    id: UUID
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    vehicle_vin: Optional[str] = None
    dtc_codes: List[str]
    symptoms_text: str
    confidence_score: float
    created_at: datetime

    class Config:
        from_attributes = True


class DiagnosisHistoryFilter(BaseModel):
    """Filter parameters for diagnosis history queries."""

    vehicle_make: Optional[str] = Field(None, max_length=100, description="Filter by vehicle make")
    vehicle_model: Optional[str] = Field(
        None, max_length=100, description="Filter by vehicle model"
    )
    vehicle_year: Optional[int] = Field(
        None, ge=1900, le=2030, description="Filter by vehicle year"
    )
    dtc_code: Optional[str] = Field(None, max_length=10, description="Filter by DTC code")
    date_from: Optional[datetime] = Field(None, description="Filter by start date")
    date_to: Optional[datetime] = Field(None, description="Filter by end date")


class PaginatedDiagnosisHistory(BaseModel):
    """Paginated response for diagnosis history."""

    items: List[DiagnosisHistoryItem]
    total: int = Field(..., ge=0, description="Total number of items")
    skip: int = Field(..., ge=0, description="Number of items skipped")
    limit: int = Field(..., ge=1, le=100, description="Maximum items per page")
    has_more: bool = Field(..., description="Whether more items are available")

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "vehicle_make": "Volkswagen",
                        "vehicle_model": "Golf",
                        "vehicle_year": 2018,
                        "dtc_codes": ["P0101", "P0171"],
                        "symptoms_text": "Motor nehezen indul",
                        "confidence_score": 0.85,
                        "created_at": "2024-02-01T10:30:00Z",
                    }
                ],
                "total": 25,
                "skip": 0,
                "limit": 10,
                "has_more": True,
            }
        }


class VehicleDiagnosisCount(BaseModel):
    """Statistics for a specific vehicle."""

    make: str
    model: str
    count: int


class DTCFrequency(BaseModel):
    """DTC code frequency statistics."""

    code: str
    count: int


class MonthlyDiagnosisCount(BaseModel):
    """Monthly diagnosis count statistics."""

    month: str
    count: int


class DiagnosisStats(BaseModel):
    """User diagnosis statistics."""

    total_diagnoses: int = Field(..., ge=0, description="Total number of diagnoses")
    avg_confidence: float = Field(..., ge=0, le=1, description="Average confidence score")
    most_diagnosed_vehicles: List[VehicleDiagnosisCount] = Field(
        default_factory=list, description="Most diagnosed vehicle makes/models"
    )
    most_common_dtcs: List[DTCFrequency] = Field(
        default_factory=list, description="Most common DTC codes"
    )
    diagnoses_by_month: List[MonthlyDiagnosisCount] = Field(
        default_factory=list, description="Diagnosis counts by month"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_diagnoses": 42,
                "avg_confidence": 0.75,
                "most_diagnosed_vehicles": [
                    {"make": "Volkswagen", "model": "Golf", "count": 15},
                    {"make": "BMW", "model": "3 Series", "count": 8},
                ],
                "most_common_dtcs": [
                    {"code": "P0171", "count": 12},
                    {"code": "P0300", "count": 8},
                ],
                "diagnoses_by_month": [
                    {"month": "2024-02", "count": 8},
                    {"month": "2024-01", "count": 12},
                ],
            }
        }


class DeleteResponse(BaseModel):
    """Response for delete operation."""

    success: bool
    message: str
    deleted_id: Optional[UUID] = None


# =============================================================================
# Streaming Response Schemas
# =============================================================================


class StreamingEventType(str, Enum):
    """Types of streaming events."""

    START = "start"
    CONTEXT = "context"
    ANALYSIS = "analysis"
    CAUSE = "cause"
    REPAIR = "repair"
    WARNING = "warning"
    COMPLETE = "complete"
    ERROR = "error"


class StreamingEvent(BaseModel):
    """Schema for streaming diagnosis events (Server-Sent Events)."""

    event_type: str = Field(
        ...,
        description="Event type: start, context, analysis, cause, repair, warning, complete, error",
    )
    data: dict = Field(default_factory=dict, description="Event data payload")
    diagnosis_id: UUID = Field(..., description="Diagnosis session ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Event timestamp"
    )
    progress: Optional[float] = Field(None, ge=0, le=1, description="Progress indicator (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "cause",
                "data": {
                    "title": "MAF szenzor hiba",
                    "description": "A levegotomeg-mero szenzor hibas jeleket kuld.",
                    "confidence": 0.85,
                },
                "diagnosis_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-02-03T10:30:00Z",
                "progress": 0.6,
            }
        }


class DiagnosisStreamRequest(BaseModel):
    """Request schema for streaming vehicle diagnosis."""

    vehicle_make: str = Field(..., min_length=1, max_length=100, description="Vehicle manufacturer")
    vehicle_model: str = Field(..., min_length=1, max_length=100, description="Vehicle model")
    vehicle_year: int = Field(..., ge=1900, le=2030, description="Vehicle year")
    vehicle_engine: Optional[str] = Field(None, max_length=100, description="Engine type/code")
    vin: Optional[str] = Field(
        None, min_length=17, max_length=17, description="Vehicle Identification Number"
    )

    dtc_codes: List[str] = Field(..., min_length=1, max_length=20, description="List of DTC codes")
    symptoms: str = Field(
        ..., min_length=10, max_length=2000, description="Symptom description in Hungarian"
    )
    additional_context: Optional[str] = Field(
        None, max_length=1000, description="Additional context"
    )

    # Streaming options
    include_context: bool = Field(True, description="Include retrieved context in stream")
    include_progress: bool = Field(True, description="Include progress updates")

    @field_validator("vehicle_make")
    @classmethod
    def normalize_vehicle_make(cls, v: str) -> str:
        """Canonicalize the make once at the API boundary (see DiagnosisRequest)."""
        normalized = normalize_make(v)
        if not normalized:
            raise ValueError("vehicle_make must not be blank")
        return normalized

    @field_validator("dtc_codes")
    @classmethod
    def normalize_dtc_codes(cls, v: List[str]) -> List[str]:
        """Normalize and validate DTC codes at the API boundary (see DiagnosisRequest)."""
        return normalize_dtc_codes(v)

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "vehicle_engine": "2.0 TSI",
                "dtc_codes": ["P0101", "P0171"],
                "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, es a fogyasztas megnott.",
                "additional_context": "A problema telen rosszabb.",
                "include_context": True,
                "include_progress": True,
            }
        }
