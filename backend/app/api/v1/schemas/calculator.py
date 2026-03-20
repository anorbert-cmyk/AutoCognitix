"""
Calculator schemas - "Megeri megjavitani?" (Worth Repairing?) calculator models.

Provides vehicle valuation vs. repair cost analysis schemas
for the Hungarian automotive market.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class VehicleCondition(str, Enum):
    """Vehicle condition levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class RecommendationType(str, Enum):
    """Repair recommendation types."""

    REPAIR = "repair"
    SELL = "sell"
    SCRAP = "scrap"


class ImpactType(str, Enum):
    """Value factor impact direction."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class CalculatorRequest(BaseModel):
    """Request schema for the repair worthiness calculator."""

    vehicle_make: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Jarmu gyartoja (pl. Volkswagen, BMW)",
    )
    vehicle_model: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Jarmu modellje (pl. Golf, 3 Series)",
    )
    vehicle_year: int = Field(
        ...,
        ge=1990,
        le=2030,
        description="Gyartasi ev",
    )
    mileage_km: int = Field(
        ...,
        ge=0,
        le=999999,
        description="Kilometerora allasa (km)",
    )
    condition: VehicleCondition = Field(
        ...,
        description="Jarmu altalanos allapota",
    )
    repair_cost_huf: Optional[int] = Field(
        None,
        ge=0,
        description="Becsult javitasi koltseg (HUF) - ha ismert",
    )
    diagnosis_id: Optional[str] = Field(
        None,
        max_length=36,
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        description="Korabbi diagnosztika azonositoja (UUID formatum)",
    )
    fuel_type: Optional[str] = Field(
        None,
        max_length=30,
        description="Uzemanyag tipusa (benzin, dizel, elektromos, hybrid)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "mileage_km": 98420,
                "condition": "good",
                "repair_cost_huf": 185000,
                "fuel_type": "benzin",
            }
        }


class CostBreakdown(BaseModel):
    """Repair cost breakdown."""

    parts_cost: int = Field(0, ge=0, description="Alkatresz koltseg (HUF)")
    labor_cost: int = Field(0, ge=0, description="Munkadij (HUF)")
    additional_costs: int = Field(0, ge=0, description="Egyeb koltsegek (HUF)")


class ValueFactor(BaseModel):
    """Factor affecting vehicle value assessment."""

    name: str = Field(..., description="Tenyezo neve")
    impact: ImpactType = Field(..., description="Hatas iranya (positive/negative)")
    description: str = Field(..., description="Reszletes leiras magyarul")


class AlternativeScenario(BaseModel):
    """Alternative outcome scenario for the vehicle."""

    scenario: str = Field(..., description="Szcenario rovid neve")
    description: str = Field(..., description="Reszletes leiras magyarul")
    estimated_value: int = Field(..., ge=0, description="Becsult ertek (HUF)")


class CalculatorResponse(BaseModel):
    """Response schema for the repair worthiness calculator."""

    # Vehicle value estimate
    vehicle_value_min: int = Field(..., ge=0, description="Jarmu becsult erteke - minimum (HUF)")
    vehicle_value_max: int = Field(..., ge=0, description="Jarmu becsult erteke - maximum (HUF)")
    vehicle_value_avg: int = Field(..., ge=0, description="Jarmu becsult erteke - atlag (HUF)")

    # Repair cost range
    repair_cost_min: int = Field(..., ge=0, description="Javitasi koltseg - minimum (HUF)")
    repair_cost_max: int = Field(..., ge=0, description="Javitasi koltseg - maximum (HUF)")

    # Analysis
    ratio: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Javitasi koltseg / jarmu ertek arany",
    )
    recommendation: RecommendationType = Field(
        ..., description="Ajanlas tipusa (repair/sell/scrap)"
    )
    recommendation_text: str = Field(..., description="Reszletes ajanlas magyarul")

    # Breakdown and factors
    breakdown: CostBreakdown = Field(..., description="Koltseg reszletezese")
    factors: List[ValueFactor] = Field(
        default_factory=list,
        description="Ertekbecslesre hato tenyezok",
    )
    alternative_scenarios: List[AlternativeScenario] = Field(
        default_factory=list,
        description="Alternativ forgatokonyvek",
    )

    # Metadata
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Becsles megbizhatosaga (0-1)",
    )
    currency: str = Field("HUF", description="Penznem")

    # AI disclaimer
    ai_disclaimer: str = Field(
        default=(
            "Ez a becsles tajekoztato jellegu, statisztikai adatokon alapul. "
            "A valos piaci ar ettol jelentosen elterhet. "
            "Dontese elott kerjuk, konzultaljon szakemberrel."
        ),
        description="Jogi nyilatkozat",
    )

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Clamp confidence score to valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_value_min": 3200000,
                "vehicle_value_max": 4100000,
                "vehicle_value_avg": 3650000,
                "repair_cost_min": 150000,
                "repair_cost_max": 220000,
                "ratio": 0.05,
                "recommendation": "repair",
                "recommendation_text": (
                    "Javitas ajanlott. A javitasi koltseg mindossze 5%-a a jarmu ertekenek."
                ),
                "breakdown": {
                    "parts_cost": 120000,
                    "labor_cost": 65000,
                    "additional_costs": 0,
                },
                "factors": [
                    {
                        "name": "Jarmu kora",
                        "impact": "negative",
                        "description": "8 eves jarmu, atlagon feluli ertekcsokkenest mutat.",
                    }
                ],
                "alternative_scenarios": [
                    {
                        "scenario": "Eladas jelenlegi allapotban",
                        "description": "A jarmu jelenlegi allapotaban torteno ertekesitese.",
                        "estimated_value": 3000000,
                    }
                ],
                "confidence_score": 0.75,
                "currency": "HUF",
            }
        }
