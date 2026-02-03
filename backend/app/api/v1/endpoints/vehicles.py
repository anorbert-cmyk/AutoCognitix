"""
Vehicle endpoints - vehicle information and VIN decoding.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.api.v1.schemas.vehicle import (
    VehicleMake,
    VehicleModel,
    VINDecodeRequest,
    VINDecodeResponse,
)

router = APIRouter()


@router.get("/makes", response_model=List[VehicleMake])
async def get_vehicle_makes(
    search: Optional[str] = Query(None, min_length=1, description="Search term for filtering makes"),
):
    """
    Get list of vehicle makes (manufacturers).

    Args:
        search: Optional search term to filter makes

    Returns:
        List of vehicle makes
    """
    # TODO: Implement with database
    # Placeholder data for common European makes
    makes = [
        VehicleMake(id="audi", name="Audi", country="Germany"),
        VehicleMake(id="bmw", name="BMW", country="Germany"),
        VehicleMake(id="citroen", name="Citroën", country="France"),
        VehicleMake(id="fiat", name="Fiat", country="Italy"),
        VehicleMake(id="ford", name="Ford", country="USA"),
        VehicleMake(id="honda", name="Honda", country="Japan"),
        VehicleMake(id="hyundai", name="Hyundai", country="South Korea"),
        VehicleMake(id="kia", name="Kia", country="South Korea"),
        VehicleMake(id="mazda", name="Mazda", country="Japan"),
        VehicleMake(id="mercedes", name="Mercedes-Benz", country="Germany"),
        VehicleMake(id="nissan", name="Nissan", country="Japan"),
        VehicleMake(id="opel", name="Opel", country="Germany"),
        VehicleMake(id="peugeot", name="Peugeot", country="France"),
        VehicleMake(id="renault", name="Renault", country="France"),
        VehicleMake(id="seat", name="SEAT", country="Spain"),
        VehicleMake(id="skoda", name="Škoda", country="Czech Republic"),
        VehicleMake(id="suzuki", name="Suzuki", country="Japan"),
        VehicleMake(id="toyota", name="Toyota", country="Japan"),
        VehicleMake(id="volkswagen", name="Volkswagen", country="Germany"),
        VehicleMake(id="volvo", name="Volvo", country="Sweden"),
    ]

    if search:
        search_lower = search.lower()
        makes = [m for m in makes if search_lower in m.name.lower()]

    return makes


@router.get("/models/{make_id}", response_model=List[VehicleModel])
async def get_vehicle_models(
    make_id: str,
    year: Optional[int] = Query(None, ge=1900, le=2030, description="Filter by year"),
):
    """
    Get list of models for a specific make.

    Args:
        make_id: ID of the vehicle make
        year: Optional year to filter models

    Returns:
        List of vehicle models
    """
    # TODO: Implement with database
    # Placeholder data for Volkswagen models
    if make_id == "volkswagen":
        models = [
            VehicleModel(id="golf", name="Golf", make_id="volkswagen", year_start=1974, year_end=None),
            VehicleModel(id="passat", name="Passat", make_id="volkswagen", year_start=1973, year_end=None),
            VehicleModel(id="polo", name="Polo", make_id="volkswagen", year_start=1975, year_end=None),
            VehicleModel(id="tiguan", name="Tiguan", make_id="volkswagen", year_start=2007, year_end=None),
            VehicleModel(id="touran", name="Touran", make_id="volkswagen", year_start=2003, year_end=None),
            VehicleModel(id="arteon", name="Arteon", make_id="volkswagen", year_start=2017, year_end=None),
            VehicleModel(id="id3", name="ID.3", make_id="volkswagen", year_start=2020, year_end=None),
            VehicleModel(id="id4", name="ID.4", make_id="volkswagen", year_start=2021, year_end=None),
        ]

        if year:
            models = [
                m for m in models
                if m.year_start <= year and (m.year_end is None or m.year_end >= year)
            ]

        return models

    # Return empty list for unknown makes (will be populated from database)
    return []


@router.post("/decode-vin", response_model=VINDecodeResponse)
async def decode_vin(request: VINDecodeRequest):
    """
    Decode a VIN (Vehicle Identification Number) to get vehicle details.

    Uses NHTSA vPIC API for US vehicles and other sources for European vehicles.

    Args:
        request: VIN decode request

    Returns:
        Decoded vehicle information
    """
    vin = request.vin.upper().strip()

    # Validate VIN format (basic validation)
    if len(vin) != 17:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="VIN must be exactly 17 characters",
        )

    # Check for invalid characters
    invalid_chars = set("IOQ")
    if any(c in invalid_chars for c in vin):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="VIN contains invalid characters (I, O, Q are not allowed)",
        )

    # TODO: Implement NHTSA API call
    # For now, return placeholder based on VIN structure

    # Determine region from first character
    first_char = vin[0]
    if first_char in "12345":
        region = "North America"
    elif first_char in "SJKLMNPR":
        region = "Asia"
    elif first_char in "SALFGHJKLMNPRSTUVWXYZ"[0:12]:
        region = "Europe"
    else:
        region = "Unknown"

    return VINDecodeResponse(
        vin=vin,
        make="Volkswagen",  # Placeholder
        model="Golf",  # Placeholder
        year=2018,  # Placeholder
        trim="GTI",  # Placeholder
        engine="2.0L TSI",  # Placeholder
        transmission="DSG",  # Placeholder
        drive_type="FWD",  # Placeholder
        body_type="Hatchback",  # Placeholder
        fuel_type="Gasoline",  # Placeholder
        region=region,
        country_of_origin="Germany",  # Placeholder
    )


@router.get("/years")
async def get_available_years():
    """
    Get list of available vehicle years.

    Returns:
        List of years from 1980 to current year + 1
    """
    from datetime import datetime

    current_year = datetime.now().year
    years = list(range(current_year + 1, 1979, -1))

    return {"years": years}
