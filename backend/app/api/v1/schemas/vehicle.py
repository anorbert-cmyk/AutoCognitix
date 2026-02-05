"""
Vehicle schemas.
"""


from pydantic import BaseModel, Field


class VehicleMake(BaseModel):
    """Schema for vehicle make (manufacturer)."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Make name")
    country: str | None = Field(None, description="Country of origin")
    logo_url: str | None = Field(None, description="Logo URL")

    class Config:
        from_attributes = True


class VehicleModel(BaseModel):
    """Schema for vehicle model."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Model name")
    make_id: str = Field(..., description="Reference to make")
    year_start: int = Field(..., description="First production year")
    year_end: int | None = Field(None, description="Last production year (null if still in production)")
    body_types: list[str] = Field(default_factory=list, description="Available body types")

    class Config:
        from_attributes = True


class VINDecodeRequest(BaseModel):
    """Schema for VIN decode request."""

    vin: str = Field(..., min_length=17, max_length=17, description="17-character VIN")

    class Config:
        json_schema_extra = {
            "example": {
                "vin": "WVWZZZ3CZWE123456"
            }
        }


class VINDecodeResponse(BaseModel):
    """Schema for VIN decode response."""

    vin: str = Field(..., description="The decoded VIN")
    make: str = Field(..., description="Vehicle manufacturer")
    model: str = Field(..., description="Vehicle model")
    year: int = Field(..., description="Model year")
    trim: str | None = Field(None, description="Trim level")
    engine: str | None = Field(None, description="Engine specification")
    transmission: str | None = Field(None, description="Transmission type")
    drive_type: str | None = Field(None, description="Drive type (FWD, RWD, AWD, 4WD)")
    body_type: str | None = Field(None, description="Body type")
    fuel_type: str | None = Field(None, description="Fuel type")
    region: str | None = Field(None, description="Region of manufacture")
    country_of_origin: str | None = Field(None, description="Country of manufacture")

    class Config:
        from_attributes = True


class VehicleCreate(BaseModel):
    """Schema for creating a vehicle record."""

    make: str = Field(..., min_length=1, max_length=100)
    model: str = Field(..., min_length=1, max_length=100)
    year: int = Field(..., ge=1900, le=2030)
    vin: str | None = Field(None, min_length=17, max_length=17)
    engine_code: str | None = Field(None, max_length=50)
    mileage_km: int | None = Field(None, ge=0)


class VehicleResponse(BaseModel):
    """Schema for vehicle response."""

    id: str
    make: str
    model: str
    year: int
    vin: str | None = None
    engine_code: str | None = None
    mileage_km: int | None = None

    class Config:
        from_attributes = True
