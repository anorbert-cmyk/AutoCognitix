"""
Diagnosis schemas - core diagnostic request/response models.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DiagnosisRequest(BaseModel):
    """Request schema for vehicle diagnosis."""

    vehicle_make: str = Field(..., min_length=1, max_length=100, description="Vehicle manufacturer")
    vehicle_model: str = Field(..., min_length=1, max_length=100, description="Vehicle model")
    vehicle_year: int = Field(..., ge=1900, le=2030, description="Vehicle year")
    vehicle_engine: Optional[str] = Field(None, max_length=100, description="Engine type/code")
    vin: Optional[str] = Field(None, min_length=17, max_length=17, description="Vehicle Identification Number")

    dtc_codes: List[str] = Field(..., min_length=1, max_length=20, description="List of DTC codes")
    symptoms: str = Field(..., min_length=10, max_length=2000, description="Symptom description in Hungarian")
    additional_context: Optional[str] = Field(None, max_length=1000, description="Additional context")

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


class RepairRecommendation(BaseModel):
    """Schema for repair recommendation."""

    title: str = Field(..., description="Short title of the repair")
    description: str = Field(..., description="Detailed repair instructions in Hungarian")
    estimated_cost_min: Optional[int] = Field(None, ge=0, description="Minimum estimated cost")
    estimated_cost_max: Optional[int] = Field(None, ge=0, description="Maximum estimated cost")
    estimated_cost_currency: str = Field("HUF", description="Currency code")
    difficulty: str = Field("intermediate", description="Difficulty level: beginner, intermediate, advanced, professional")
    parts_needed: List[str] = Field(default_factory=list, description="List of parts needed")
    estimated_time_minutes: Optional[int] = Field(None, ge=0, description="Estimated repair time in minutes")


class Source(BaseModel):
    """Schema for information source."""

    type: str = Field(..., description="Source type: tsb, forum, video, manual, database")
    title: str = Field(..., description="Source title")
    url: Optional[str] = Field(None, description="Source URL if available")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score (0-1)")


class DiagnosisResponse(BaseModel):
    """Response schema for vehicle diagnosis."""

    id: UUID = Field(..., description="Unique diagnosis ID")
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    dtc_codes: List[str]
    symptoms: str

    probable_causes: List[ProbableCause] = Field(..., description="Ranked list of probable causes")
    recommended_repairs: List[RepairRecommendation] = Field(..., description="Recommended repairs")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall diagnosis confidence")
    sources: List[Source] = Field(default_factory=list, description="Information sources used")

    created_at: datetime = Field(default_factory=datetime.utcnow)

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
    vehicle_model: Optional[str] = Field(None, max_length=100, description="Filter by vehicle model")
    vehicle_year: Optional[int] = Field(None, ge=1900, le=2030, description="Filter by vehicle year")
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
