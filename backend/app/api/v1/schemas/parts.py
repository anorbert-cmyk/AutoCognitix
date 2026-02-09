"""
Parts and repair cost estimation schemas.

Alkatrész és javítási költség becslés sémák a diagnosztikai rendszerhez.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PartCategory(str, Enum):
    """
    Alkatrész kategóriák.

    Part categories for organizing automotive components.
    """

    SENSORS = "sensors"
    FILTERS = "filters"
    IGNITION = "ignition"
    BRAKES = "brakes"
    EXHAUST = "exhaust"
    COOLING = "cooling"
    FUEL_SYSTEM = "fuel_system"
    ELECTRICAL = "electrical"
    EMISSIONS = "emissions"
    ENGINE = "engine"
    TRANSMISSION = "transmission"
    CONSUMABLES = "consumables"


class LaborDifficulty(str, Enum):
    """
    Munkadíj nehézségi szintek.

    Labor difficulty levels for repair cost estimation.
    """

    EASY = "easy"  # Egyszerű - bárki meg tudja csinálni
    MEDIUM = "medium"  # Közepes - alapvető szerszámok kellenek
    HARD = "hard"  # Nehéz - tapasztalat szükséges
    EXPERT = "expert"  # Szakértői - szerviz vagy specialista


class PriceSource(BaseModel):
    """
    Alkatrész ár forrás információ.

    Represents a source for part pricing (webshop, dealer, etc.).
    """

    name: str = Field(
        ..., min_length=1, max_length=100, description="Forrás neve (pl. 'AutoDoc', 'EuroAuto')"
    )
    price_min: int = Field(..., ge=0, description="Minimális ár a forrásnál")
    price_max: int = Field(..., ge=0, description="Maximális ár a forrásnál")
    currency: str = Field("HUF", max_length=3, description="Pénznem (ISO 4217)")
    in_stock: Optional[bool] = Field(None, description="Készleten van-e")
    delivery_days: Optional[int] = Field(None, ge=0, le=365, description="Szállítási idő napokban")
    url: Optional[str] = Field(None, max_length=500, description="Link a termékhez")

    @field_validator("price_max")
    @classmethod
    def validate_price_range(cls, v: int, info) -> int:
        """Ensure max price is not less than min price."""
        if "price_min" in info.data and v < info.data["price_min"]:
            return info.data["price_min"]
        return v


class PartInfo(BaseModel):
    """
    Alkatrész információ.

    Detailed information about an automotive part including pricing and compatibility.
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Egyedi alkatrész azonosító (pl. 'maf_sensor')",
    )
    name: str = Field(..., min_length=1, max_length=200, description="Alkatrész neve magyarul")
    name_en: Optional[str] = Field(None, max_length=200, description="Alkatrész neve angolul")
    category: PartCategory = Field(..., description="Alkatrész kategória")

    # Cikkszámok
    part_number: Optional[str] = Field(None, max_length=50, description="Utángyártott cikkszám")
    oem_number: Optional[str] = Field(None, max_length=50, description="Gyári (OEM) cikkszám")

    # Leírás
    description: Optional[str] = Field(
        None, max_length=1000, description="Részletes leírás magyarul"
    )

    # Árazás
    price_range_min: int = Field(..., ge=0, description="Minimális ár (HUF)")
    price_range_max: int = Field(..., ge=0, description="Maximális ár (HUF)")
    currency: str = Field("HUF", max_length=3, description="Pénznem")

    # Források
    sources: List[PriceSource] = Field(default_factory=list, description="Ár források listája")

    # Minőség
    is_oem: bool = Field(False, description="OEM (gyári) alkatrész-e")
    quality_rating: Optional[float] = Field(
        None, ge=0, le=5, description="Minőségi értékelés (0-5)"
    )

    # Kompatibilitás
    compatibility_notes: Optional[str] = Field(
        None, max_length=500, description="Kompatibilitási megjegyzések"
    )

    # Beszerelés
    labor_hours: Optional[float] = Field(
        None, ge=0, le=100, description="Becsült beszerelési idő órában"
    )

    @field_validator("price_range_max")
    @classmethod
    def validate_price_range(cls, v: int, info) -> int:
        """Ensure max price is not less than min price."""
        if "price_range_min" in info.data and v < info.data["price_range_min"]:
            return info.data["price_range_min"]
        return v

    @field_validator("quality_rating")
    @classmethod
    def validate_quality_rating(cls, v: Optional[float]) -> Optional[float]:
        """Clamp quality rating to valid range."""
        if v is not None:
            return max(0.0, min(5.0, v))
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "id": "maf_sensor",
                "name": "Légtömegmérő szenzor (MAF)",
                "name_en": "Mass Air Flow Sensor",
                "category": "sensors",
                "part_number": "0280218063",
                "oem_number": "06A906461L",
                "description": "A légtömegmérő szenzor méri a motorba áramló levegő mennyiségét.",
                "price_range_min": 15000,
                "price_range_max": 85000,
                "currency": "HUF",
                "is_oem": False,
                "quality_rating": 4.2,
                "compatibility_notes": "Volkswagen/Audi 1.8T, 2.0 motorokhoz",
                "labor_hours": 0.5,
            }
        }


class RepairCostEstimate(BaseModel):
    """
    Javítási költség becslés.

    Comprehensive repair cost estimate including parts and labor.
    """

    dtc_code: str = Field(..., min_length=1, max_length=10, description="DTC hibakód")
    repair_name: str = Field(
        ..., min_length=1, max_length=200, description="Javítás megnevezése magyarul"
    )

    # Alkatrész költségek
    parts_cost_min: int = Field(..., ge=0, description="Alkatrészek minimális költsége (HUF)")
    parts_cost_max: int = Field(..., ge=0, description="Alkatrészek maximális költsége (HUF)")

    # Munkadíj
    labor_cost_min: int = Field(..., ge=0, description="Munkadíj minimum (HUF)")
    labor_cost_max: int = Field(..., ge=0, description="Munkadíj maximum (HUF)")

    # Összesen
    total_cost_min: int = Field(..., ge=0, description="Teljes költség minimum (HUF)")
    total_cost_max: int = Field(..., ge=0, description="Teljes költség maximum (HUF)")
    currency: str = Field("HUF", max_length=3, description="Pénznem")

    # Időbecslés
    estimated_hours: float = Field(..., ge=0, le=100, description="Becsült javítási idő órában")

    # Nehézség
    difficulty: LaborDifficulty = Field(..., description="Javítás nehézségi szintje")

    # Megbízhatóság
    confidence: float = Field(..., ge=0, le=1, description="Becslés megbízhatósága (0-1)")

    # Részletező információk
    parts: List[PartInfo] = Field(default_factory=list, description="Szükséges alkatrészek listája")
    notes: Optional[str] = Field(None, max_length=1000, description="További megjegyzések")
    disclaimer: str = Field(
        "A költségbecslés tájékoztató jellegű. A tényleges árak a szerviz és az alkatrész minőségétől függően változhatnak.",
        max_length=500,
        description="Jogi nyilatkozat",
    )

    @field_validator("parts_cost_max")
    @classmethod
    def validate_parts_cost(cls, v: int, info) -> int:
        """Ensure max parts cost is not less than min."""
        if "parts_cost_min" in info.data and v < info.data["parts_cost_min"]:
            return info.data["parts_cost_min"]
        return v

    @field_validator("labor_cost_max")
    @classmethod
    def validate_labor_cost(cls, v: int, info) -> int:
        """Ensure max labor cost is not less than min."""
        if "labor_cost_min" in info.data and v < info.data["labor_cost_min"]:
            return info.data["labor_cost_min"]
        return v

    @field_validator("total_cost_max")
    @classmethod
    def validate_total_cost(cls, v: int, info) -> int:
        """Ensure max total cost is not less than min."""
        if "total_cost_min" in info.data and v < info.data["total_cost_min"]:
            return info.data["total_cost_min"]
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Clamp confidence to valid range."""
        return max(0.0, min(1.0, v))

    class Config:
        json_schema_extra = {
            "example": {
                "dtc_code": "P0171",
                "repair_name": "Sovány keverék javítása - MAF szenzor csere",
                "parts_cost_min": 15000,
                "parts_cost_max": 85000,
                "labor_cost_min": 8000,
                "labor_cost_max": 20000,
                "total_cost_min": 23000,
                "total_cost_max": 105000,
                "currency": "HUF",
                "estimated_hours": 1.0,
                "difficulty": "medium",
                "confidence": 0.75,
                "parts": [],
                "notes": "A MAF szenzor cseréje általában megoldja a P0171 hibakódot. Ellenőrizze a légszűrőt is.",
                "disclaimer": "A költségbecslés tájékoztató jellegű. A tényleges árak a szerviz és az alkatrész minőségétől függően változhatnak.",
            }
        }


class PartsSearchRequest(BaseModel):
    """
    Alkatrész keresési kérés.

    Request for searching parts compatible with a specific vehicle.
    """

    vehicle_make: str = Field(..., min_length=1, max_length=100, description="Gyártó")
    vehicle_model: str = Field(..., min_length=1, max_length=100, description="Modell")
    vehicle_year: int = Field(..., ge=1900, le=2030, description="Évjárat")
    dtc_code: Optional[str] = Field(None, max_length=10, description="DTC hibakód (opcionális)")
    part_category: Optional[PartCategory] = Field(None, description="Kategória szűrő")
    search_term: Optional[str] = Field(None, max_length=200, description="Keresési kifejezés")

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_make": "Volkswagen",
                "vehicle_model": "Golf",
                "vehicle_year": 2018,
                "dtc_code": "P0171",
                "part_category": "sensors",
                "search_term": "MAF",
            }
        }


class PartsSearchResponse(BaseModel):
    """
    Alkatrész keresési válasz.

    Response containing matching parts and repair cost estimate.
    """

    parts: List[PartInfo] = Field(..., description="Találatok listája")
    total_count: int = Field(..., ge=0, description="Összes találat száma")
    cost_estimate: Optional[RepairCostEstimate] = Field(
        None, description="Javítási költség becslés ha DTC kód megadva"
    )

    class Config:
        json_schema_extra = {"example": {"parts": [], "total_count": 5, "cost_estimate": None}}
