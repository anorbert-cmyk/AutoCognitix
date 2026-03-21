"""
Service shop schemas - models for the Szerviz Összehasonlítás feature.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ServiceShop(BaseModel):
    """Schema for a service shop."""

    id: str = Field(..., description="Unique shop identifier")
    name: str = Field(..., description="Shop name")
    address: str = Field(..., description="Full street address")
    city: str = Field(..., description="City name")
    region: str = Field(..., description="Region/county name")
    lat: float = Field(..., description="Latitude")
    lng: float = Field(..., description="Longitude")
    phone: Optional[str] = Field(None, description="Phone number")
    website: Optional[str] = Field(None, description="Website URL")
    rating: float = Field(..., ge=0.0, le=5.0, description="Average rating (0-5)")
    review_count: int = Field(..., ge=0, description="Number of reviews")
    specializations: List[str] = Field(
        default_factory=list,
        description="Specializations (general, german, japanese, electric, diagnosis, bodywork)",
    )
    accepted_makes: List[str] = Field(default_factory=list, description="Accepted vehicle makes")
    price_level: int = Field(
        ..., ge=1, le=3, description="Price level (1=budget, 2=mid, 3=premium)"
    )
    services: List[str] = Field(default_factory=list, description="Available service types")
    opening_hours: Optional[str] = Field(None, description="Opening hours")
    has_inspection: bool = Field(False, description="Offers vehicle inspection")
    has_courtesy_car: bool = Field(False, description="Offers courtesy car")
    distance_km: Optional[float] = Field(
        None, description="Distance from user in km (if location provided)"
    )


class Region(BaseModel):
    """Schema for a Hungarian region."""

    id: str = Field(..., description="Region identifier")
    name: str = Field(..., description="Region display name")
    county: str = Field(..., description="County (megye) name")
    lat: float = Field(..., description="Center latitude")
    lng: float = Field(..., description="Center longitude")
    shop_count: int = Field(0, ge=0, description="Number of shops in region")


class ServiceSearchParams(BaseModel):
    """Query parameters for service shop search."""

    region: Optional[str] = Field(None, description="Filter by region ID")
    vehicle_make: Optional[str] = Field(None, description="Filter by accepted vehicle make")
    service_type: Optional[str] = Field(None, description="Filter by service type")
    sort_by: Optional[str] = Field(
        None,
        description="Sort by: rating, price, distance",
        pattern="^(rating|price|distance)$",
    )
    lat: Optional[float] = Field(None, ge=45.0, le=49.0, description="User latitude")
    lng: Optional[float] = Field(None, ge=16.0, le=23.0, description="User longitude")
    limit: int = Field(20, ge=1, le=100, description="Results per page")
    offset: int = Field(0, ge=0, description="Results offset")


class ServiceSearchResponse(BaseModel):
    """Response for service shop search."""

    shops: List[ServiceShop] = Field(default_factory=list, description="Matching shops")
    total: int = Field(0, ge=0, description="Total matching shops")
    limit: int = Field(20, ge=1, le=100, description="Results per page")
    offset: int = Field(0, ge=0, description="Results offset")
    has_more: bool = Field(False, description="Whether more results available")
    regions: List[Region] = Field(default_factory=list, description="Available regions")
