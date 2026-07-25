"""
Vehicle schemas.
"""

from enum import Enum
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


# Generic type for pagination
T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response schema."""

    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Maximum items per page")
    offset: int = Field(..., description="Number of items skipped")
    has_more: bool = Field(..., description="Whether more items are available")


class VehicleMake(BaseModel):
    """Schema for vehicle make (manufacturer)."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Make name")
    country: Optional[str] = Field(None, description="Country of origin")
    logo_url: Optional[str] = Field(None, description="Logo URL")

    class Config:
        from_attributes = True


class VehicleModel(BaseModel):
    """Schema for vehicle model."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Model name")
    make_id: str = Field(..., description="Reference to make")
    year_start: Optional[int] = Field(None, description="First production year")
    year_end: Optional[int] = Field(
        None, description="Last production year (null if still in production)"
    )
    body_types: List[str] = Field(default_factory=list, description="Available body types")

    class Config:
        from_attributes = True


class VehicleYearsResponse(BaseModel):
    """Schema for vehicle years response."""

    years: List[int] = Field(..., description="Available years in descending order")
    make: str = Field(..., description="Vehicle make")
    model: str = Field(..., description="Vehicle model")


class VINDecodeRequest(BaseModel):
    """Schema for VIN decode request."""

    vin: str = Field(..., min_length=17, max_length=17, description="17-character VIN")

    class Config:
        json_schema_extra = {"example": {"vin": "WVWZZZ3CZWE123456"}}


class VINDecodeResponse(BaseModel):
    """Schema for VIN decode response."""

    vin: str = Field(..., description="The decoded VIN")
    make: str = Field(..., description="Vehicle manufacturer")
    model: str = Field(..., description="Vehicle model")
    year: int = Field(..., description="Model year")
    trim: Optional[str] = Field(None, description="Trim level")
    engine: Optional[str] = Field(None, description="Engine specification")
    transmission: Optional[str] = Field(None, description="Transmission type")
    drive_type: Optional[str] = Field(None, description="Drive type (FWD, RWD, AWD, 4WD)")
    body_type: Optional[str] = Field(None, description="Body type")
    fuel_type: Optional[str] = Field(None, description="Fuel type")
    region: Optional[str] = Field(None, description="Region of manufacture")
    country_of_origin: Optional[str] = Field(None, description="Country of manufacture")

    class Config:
        from_attributes = True


class VehicleCreate(BaseModel):
    """Schema for creating a vehicle record."""

    make: str = Field(..., min_length=1, max_length=100)
    model: str = Field(..., min_length=1, max_length=100)
    year: int = Field(..., ge=1900, le=2030)
    vin: Optional[str] = Field(None, min_length=17, max_length=17)
    engine_code: Optional[str] = Field(None, max_length=50)
    mileage_km: Optional[int] = Field(None, ge=0)


class VehicleResponse(BaseModel):
    """Schema for vehicle response."""

    id: str
    make: str
    model: str
    year: int
    vin: Optional[str] = None
    engine_code: Optional[str] = None
    mileage_km: Optional[int] = None

    class Config:
        from_attributes = True


class VehicleDetailResponse(BaseModel):
    """Schema for detailed vehicle response including Neo4j data."""

    id: str = Field(..., description="Vehicle UID")
    make: str = Field(..., description="Vehicle manufacturer")
    model: str = Field(..., description="Vehicle model")
    year_start: Optional[int] = Field(None, description="First production year")
    year_end: Optional[int] = Field(None, description="Last production year")
    platform: Optional[str] = Field(None, description="Vehicle platform code")
    engine_codes: List[str] = Field(default_factory=list, description="Available engine codes")
    body_types: List[str] = Field(default_factory=list, description="Available body types")

    class Config:
        from_attributes = True


class VehicleCommonIssue(BaseModel):
    """Schema for common vehicle issue/DTC."""

    code: str = Field(..., description="DTC code")
    description_en: Optional[str] = Field(None, description="English description")
    description_hu: Optional[str] = Field(None, description="Hungarian description")
    severity: Optional[str] = Field(None, description="Issue severity")
    frequency: Optional[str] = Field(None, description="How common the issue is")
    occurrence_count: Optional[int] = Field(None, description="Number of reported occurrences")


class VehicleComplaintComponent(BaseModel):
    """A vehicle component ranked by NHTSA consumer-complaint frequency.

    Sourced from the imported NHTSA complaint corpus in PostgreSQL. Unlike the
    DTC-based `VehicleCommonIssue` list - which depends on a complaint narrative
    literally quoting a fault code, something consumers almost never do - the
    component field is present on every complaint row, so this list actually
    carries data.
    """

    component: str = Field(
        ..., description="Raw NHTSA component label (uppercase, e.g. 'ELECTRICAL SYSTEM')"
    )
    component_hu: Optional[str] = Field(
        None,
        description=(
            "Hungarian label, or null when no verified translation exists "
            "(clients should fall back to `component`)"
        ),
    )
    complaint_count: int = Field(..., description="Complaints filed against this component")
    share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of this vehicle's total complaints (0..1)",
    )
    crash_count: int = Field(..., description="Complaints that reported a crash")
    fire_count: int = Field(..., description="Complaints that reported a fire")
    injury_count: int = Field(..., description="Total injuries reported")
    death_count: int = Field(..., description="Total deaths reported")


class DataSourceStatus(str, Enum):
    """Whether the datastore behind a response section actually answered.

    ``unavailable`` means the section's list is empty for lack of data ACCESS,
    not for lack of data. A client must never present such an empty list as a
    factual absence ("this vehicle has no complaints", "this model was never
    sold in the US") - that would be a fabricated claim.
    """

    OK = "ok"
    UNAVAILABLE = "unavailable"


class CommonIssuesSources(BaseModel):
    """Per-source load status for :class:`VehicleCommonIssuesResponse`.

    The response is assembled from two INDEPENDENT datastores, each of which
    degrades to an empty list on its own outage (see the service docstrings).
    Without this block the two failure modes - "nothing to report" and "could
    not reach the source" - are indistinguishable on the wire, and the UI is
    forced to guess. It guessed wrong: a PostgreSQL blip made the panel state,
    as fact, that the user's car was never sold in the US.

    Future-proofing: every source gets its OWN REQUIRED field here. There is no
    ``ok`` default and no free-form dict to forget a key in, so wiring a third
    source in without reporting its status fails at response construction
    (pydantic ``ValidationError`` on every request) instead of silently
    inheriting "ok" - which is exactly how this defect would otherwise recur.
    """

    components: DataSourceStatus = Field(
        ...,
        description=(
            "NHTSA complaint corpus (PostgreSQL) backing `components` and `total_complaints`"
        ),
    )
    issues: DataSourceStatus = Field(
        ..., description="Complaint->DTC graph (Neo4j) backing `issues`"
    )


class VehicleCommonIssuesResponse(BaseModel):
    """Schema for vehicle common issues response."""

    make: str = Field(..., description="Vehicle make")
    model: str = Field(..., description="Vehicle model")
    year: Optional[int] = Field(None, description="Optional year filter")
    issues: List[VehicleCommonIssue] = Field(default_factory=list, description="Common issues")
    components: List[VehicleComplaintComponent] = Field(
        default_factory=list,
        description="Components ranked by NHTSA complaint frequency (descending)",
    )
    total_complaints: int = Field(
        0,
        ge=0,
        description=(
            "Total NHTSA complaints stored for this vehicle across ALL components "
            "(the denominator for `share`; 0 means no complaint data)"
        ),
    )
    sources: CommonIssuesSources = Field(
        ...,
        description=(
            "Per-source load status. Tells an empty list caused by a datastore "
            "outage apart from a genuinely empty one - clients must only present "
            "the latter as an absence of data"
        ),
    )
