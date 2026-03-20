"""
Műszaki vizsga (Technical Inspection) schemas.

Pydantic models for the vehicle technical inspection evaluation endpoint.
Maps DTC codes to Hungarian MOT (Műszaki Vizsga) inspection categories
and provides risk assessment for passing/failing.

Author: AutoCognitix Team
"""

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class InspectionRiskLevel(str, Enum):
    """Overall inspection risk level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InspectionSeverity(str, Enum):
    """Severity of an individual inspection finding."""

    FAIL = "fail"
    WARNING = "warning"
    PASS = "pass"


# =============================================================================
# Request Schema
# =============================================================================


class InspectionRequest(BaseModel):
    """Request payload for technical inspection evaluation.

    Attributes:
        vehicle_make: Vehicle manufacturer (e.g. "Volkswagen").
        vehicle_model: Vehicle model name (e.g. "Golf").
        vehicle_year: Model year (1990-2030).
        vehicle_engine: Engine description (optional).
        dtc_codes: List of DTC codes to evaluate (1-20 codes).
        mileage_km: Current odometer reading in km (optional).
        symptoms: Free-text symptom description in Hungarian (optional).
    """

    vehicle_make: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Jármű gyártó",
    )
    vehicle_model: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Jármű modell",
    )
    vehicle_year: int = Field(
        ...,
        ge=1990,
        le=2030,
        description="Évjárat",
    )
    vehicle_engine: Optional[str] = Field(
        None,
        max_length=100,
        description="Motor típus (pl. 1.4 TSI)",
    )
    dtc_codes: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="DTC hibakódok listája",
    )
    mileage_km: Optional[int] = Field(
        None,
        ge=0,
        le=2000000,
        description="Kilométeróra állás",
    )
    symptoms: Optional[str] = Field(
        None,
        max_length=2000,
        description="Tünetek szöveges leírása",
    )

    @field_validator("dtc_codes")
    @classmethod
    def validate_dtc_codes(cls, v: List[str]) -> List[str]:
        """Validate DTC code format and normalize to uppercase.

        Args:
            v: List of DTC code strings.

        Returns:
            Normalized uppercase DTC codes.

        Raises:
            ValueError: If any code has invalid format.
        """
        pattern = re.compile(r"^[PBCU][0-9A-Fa-f]{4}$")
        for code in v:
            if not pattern.match(code):
                raise ValueError(
                    f"Érvénytelen DTC formátum: {code}. "
                    f"Elvárt: [P/B/C/U] + 4 hexadecimális számjegy"
                )
        return [c.upper() for c in v]


# =============================================================================
# Response Schemas
# =============================================================================


class FailingItem(BaseModel):
    """A single failing or warning item from inspection evaluation.

    Attributes:
        category: Inspection category key in English.
        category_hu: Hungarian display name of the category.
        issue: Description of the detected issue.
        related_dtc: The DTC code that triggered this finding.
        severity: FAIL, WARNING, or PASS severity level.
        fix_recommendation: Recommended repair action.
        estimated_cost_min: Minimum estimated repair cost (HUF).
        estimated_cost_max: Maximum estimated repair cost (HUF).
    """

    category: str = Field(
        ...,
        description="Vizsgálati kategória azonosító",
    )
    category_hu: str = Field(
        ...,
        description="Vizsgálati kategória magyar neve",
    )
    issue: str = Field(
        ...,
        description="Észlelt probléma leírása",
    )
    related_dtc: str = Field(
        ...,
        description="Kapcsolódó DTC kód",
    )
    severity: InspectionSeverity = Field(
        ...,
        description="Súlyosság: fail / warning / pass",
    )
    fix_recommendation: str = Field(
        ...,
        description="Javasolt javítás",
    )
    estimated_cost_min: int = Field(
        0,
        ge=0,
        description="Becsült minimális javítási költség (HUF)",
    )
    estimated_cost_max: int = Field(
        0,
        ge=0,
        description="Becsült maximális javítási költség (HUF)",
    )


class InspectionResponse(BaseModel):
    """Response payload for technical inspection evaluation.

    Attributes:
        overall_risk: HIGH / MEDIUM / LOW risk classification.
        risk_score: Numeric risk score between 0 (safe) and 1 (critical).
        failing_items: List of items that would fail or warn at inspection.
        passing_categories: Categories with no issues found.
        recommendations: Summary recommendations in Hungarian.
        estimated_total_fix_cost_min: Total minimum repair cost (HUF).
        estimated_total_fix_cost_max: Total maximum repair cost (HUF).
        vehicle_info: Human-readable vehicle description.
        dtc_count: Number of DTC codes evaluated.
    """

    overall_risk: InspectionRiskLevel = Field(
        ...,
        description="Összesített kockázati szint",
    )
    risk_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Kockázati pontszám (0=biztonságos, 1=kritikus)",
    )
    failing_items: List[FailingItem] = Field(
        ...,
        description="Hibás/figyelmeztető tételek listája",
    )
    passing_categories: List[str] = Field(
        ...,
        description="Megfelelő vizsgálati kategóriák",
    )
    recommendations: List[str] = Field(
        ...,
        description="Javaslatok magyar nyelven",
    )
    estimated_total_fix_cost_min: int = Field(
        0,
        ge=0,
        description="Összes becsült minimális javítási költség (HUF)",
    )
    estimated_total_fix_cost_max: int = Field(
        0,
        ge=0,
        description="Összes becsült maximális javítási költség (HUF)",
    )
    vehicle_info: str = Field(
        ...,
        description="Jármű leírás",
    )
    dtc_count: int = Field(
        ...,
        ge=0,
        description="Elemzett DTC kódok száma",
    )
