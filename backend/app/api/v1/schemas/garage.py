"""
Pydantic schemas for the Garage API (vehicles, reminders, maintenance costs).
"""

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class FuelType(str, Enum):
    PETROL = "petrol"
    DIESEL = "diesel"
    ELECTRIC = "electric"
    HYBRID = "hybrid"
    LPG = "lpg"
    OTHER = "other"


class ReminderType(str, Enum):
    OIL_CHANGE = "oil_change"
    TIRE_ROTATION = "tire_rotation"
    MUESZAKI_VIZSGA = "mueszaki_vizsga"
    KOTELEZO_BIZTOSITAS = "kotelezo_biztositas"
    COOLANT = "coolant"
    BRAKE_FLUID = "brake_fluid"
    TIMING_BELT = "timing_belt"
    AIR_FILTER = "air_filter"
    BRAKE_PADS = "brake_pads"
    CUSTOM = "custom"


REMINDER_TYPE_LABELS = {
    "oil_change": "Olajcsere",
    "tire_rotation": "Gumicsere / Forgatás",
    "mueszaki_vizsga": "Műszaki vizsga",
    "kotelezo_biztositas": "Kötelező biztosítás megújítás",
    "coolant": "Hűtőfolyadék csere",
    "brake_fluid": "Fékfolyadék csere",
    "timing_belt": "Vezérszíj csere",
    "air_filter": "Légszűrő csere",
    "brake_pads": "Fékbetét csere",
    "custom": "Egyedi emlékeztető",
}


# ─── UserVehicle ─────────────────────────────────────────────────────────────


class UserVehicleCreate(BaseModel):
    nickname: Optional[str] = Field(
        None, max_length=100, description="Becenév (pl. 'Az én Golfom')"
    )
    make: str = Field(..., min_length=1, max_length=100, description="Gyártó")
    model: str = Field(..., min_length=1, max_length=100, description="Modell")
    year: int = Field(..., ge=1960, le=2030, description="Évjárat")
    vin: Optional[str] = Field(
        None, min_length=17, max_length=17, description="VIN szám (17 karakter)"
    )
    license_plate: Optional[str] = Field(None, max_length=20, description="Rendszám")
    mileage_km: Optional[int] = Field(None, ge=0, le=2000000, description="Kilométeróra állás")
    fuel_type: Optional[FuelType] = Field(None, description="Üzemanyag típus")
    color: Optional[str] = Field(None, max_length=50, description="Szín")
    notes: Optional[str] = Field(None, max_length=1000, description="Megjegyzések")

    @field_validator("vin")
    @classmethod
    def validate_vin(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 17:
            raise ValueError("A VIN szám pontosan 17 karakter hosszú kell legyen")
        return v

    @field_validator("license_plate")
    @classmethod
    def normalize_plate(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return v.upper().strip()


class UserVehicleUpdate(BaseModel):
    nickname: Optional[str] = Field(None, max_length=100)
    make: Optional[str] = Field(None, min_length=1, max_length=100)
    model: Optional[str] = Field(None, min_length=1, max_length=100)
    year: Optional[int] = Field(None, ge=1960, le=2030)
    vin: Optional[str] = Field(None, min_length=17, max_length=17)
    license_plate: Optional[str] = Field(None, max_length=20)
    mileage_km: Optional[int] = Field(None, ge=0, le=2000000)
    fuel_type: Optional[FuelType] = None
    color: Optional[str] = Field(None, max_length=50)
    notes: Optional[str] = Field(None, max_length=1000)


class UserVehicleResponse(BaseModel):
    id: str
    user_id: str
    nickname: Optional[str]
    make: str
    model: str
    year: int
    vin: Optional[str]
    license_plate: Optional[str]
    mileage_km: Optional[int]
    fuel_type: Optional[str]
    color: Optional[str]
    notes: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    health_score: Optional[int] = Field(
        None, ge=0, le=100, description="Jármű egészségi pontszám (0-100)"
    )
    upcoming_reminders_count: Optional[int] = Field(
        None, ge=0, description="Következő 30 napon belüli emlékeztetők száma"
    )

    model_config = {"from_attributes": True}


class UserVehicleListResponse(BaseModel):
    vehicles: List[UserVehicleResponse]
    total: int


# ─── MaintenanceReminder ──────────────────────────────────────────────────────


class MaintenanceReminderCreate(BaseModel):
    vehicle_id: str = Field(..., description="Jármű azonosítója")
    reminder_type: ReminderType = Field(..., description="Emlékeztető típusa")
    title: str = Field(..., min_length=1, max_length=200, description="Emlékeztető neve")
    due_date: Optional[date] = Field(None, description="Esedékesség dátuma")
    due_mileage_km: Optional[int] = Field(
        None, ge=0, le=2000000, description="Esedékesség kilométer"
    )
    notes: Optional[str] = Field(None, max_length=1000, description="Megjegyzés")


class MaintenanceReminderUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    due_date: Optional[date] = None
    due_mileage_km: Optional[int] = Field(None, ge=0, le=2000000)
    notes: Optional[str] = Field(None, max_length=1000)


class MaintenanceReminderResponse(BaseModel):
    id: str
    vehicle_id: str
    user_id: str
    reminder_type: str
    reminder_type_label: str = ""
    title: str
    due_date: Optional[date]
    due_mileage_km: Optional[int]
    notes: Optional[str]
    is_completed: bool
    completed_at: Optional[datetime]
    email_sent_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    days_until_due: Optional[int] = Field(
        None, description="Hány nap múlva esedékes (negatív ha lejárt)"
    )
    urgency: Optional[str] = Field(None, description="overdue/urgent/upcoming/ok")

    model_config = {"from_attributes": True}


class MaintenanceReminderListResponse(BaseModel):
    reminders: List[MaintenanceReminderResponse]
    total: int
    overdue_count: int = 0
    urgent_count: int = 0


# ─── MaintenanceCost ──────────────────────────────────────────────────────────


class MaintenanceCostCreate(BaseModel):
    vehicle_id: str = Field(..., description="Jármű azonosítója")
    service_type: str = Field(..., min_length=1, max_length=100, description="Szerviz típusa")
    cost_huf: int = Field(..., ge=0, description="Költség forintban")
    service_date: date = Field(..., description="Szerviz dátuma")
    mileage_km: Optional[int] = Field(None, ge=0, le=2000000)
    workshop_name: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=1000)
    diagnosis_session_id: Optional[str] = Field(None, description="Kapcsolódó diagnózis session ID")


class MaintenanceCostResponse(BaseModel):
    id: str
    vehicle_id: str
    user_id: str
    diagnosis_session_id: Optional[str]
    service_type: str
    cost_huf: int
    service_date: date
    mileage_km: Optional[int]
    workshop_name: Optional[str]
    notes: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class MaintenanceCostListResponse(BaseModel):
    costs: List[MaintenanceCostResponse]
    total: int
    total_cost_huf: int = 0


# ─── Health Score ─────────────────────────────────────────────────────────────


class VehicleHealthScore(BaseModel):
    vehicle_id: str
    score: int = Field(..., ge=0, le=100, description="Egészségi pontszám (0-100)")
    category: str = Field(..., description="kiváló/jó/figyelmet_igényel/kritikus")
    category_color: str = Field(..., description="green/yellow/orange/red")
    factors: List[dict] = Field(default_factory=list, description="Pontszámot befolyásoló tényezők")
