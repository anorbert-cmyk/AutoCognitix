"""
Vehicle endpoints - vehicle information and VIN decoding.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.v1.schemas.vehicle import (
    VehicleMake,
    VehicleModel,
    VINDecodeRequest,
    VINDecodeResponse,
)
from app.services.nhtsa_service import (
    Complaint,
    NHTSAError,
    NHTSAService,
    Recall,
    get_nhtsa_service,
)

router = APIRouter()


# =============================================================================
# Response Models for new endpoints
# =============================================================================


class RecallSummary(Recall):
    """Recall response model (re-exports from nhtsa_service)."""
    pass


class ComplaintSummary(Complaint):
    """Complaint response model (re-exports from nhtsa_service)."""
    pass


# =============================================================================
# VIN Decoding
# =============================================================================


@router.post("/decode-vin", response_model=VINDecodeResponse)
async def decode_vin(
    request: VINDecodeRequest,
    nhtsa_service: NHTSAService = Depends(get_nhtsa_service),
):
    """
    Decode a VIN (Vehicle Identification Number) to get vehicle details.

    Uses NHTSA vPIC API for decoding.

    Args:
        request: VIN decode request

    Returns:
        Decoded vehicle information

    Raises:
        HTTPException: If VIN is invalid or decoding fails
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

    try:
        # Use NHTSA service to decode VIN
        result = await nhtsa_service.decode_vin(vin)

        # Check if decoding was successful
        if not result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid VIN: {result.error_text or 'Unknown error'}",
            )

        # Build engine description from available data
        engine_parts = []
        if result.engine_displacement_l:
            engine_parts.append(f"{result.engine_displacement_l}L")
        if result.engine_cylinders:
            engine_parts.append(f"{result.engine_cylinders}-cyl")
        if result.fuel_type_primary:
            engine_parts.append(result.fuel_type_primary)
        engine = " ".join(engine_parts) if engine_parts else None

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
            make=result.make or "Unknown",
            model=result.model or "Unknown",
            year=result.model_year or 0,
            trim=result.body_class,
            engine=engine,
            transmission=result.transmission_style,
            drive_type=result.drive_type,
            body_type=result.body_class,
            fuel_type=result.fuel_type_primary,
            region=region,
            country_of_origin=result.plant_country,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except NHTSAError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"NHTSA API error: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to decode VIN: {str(e)}",
        )


# =============================================================================
# Vehicle Makes
# =============================================================================


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
    # Static list as fallback - can be extended to use NHTSA API later
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


# =============================================================================
# Vehicle Models
# =============================================================================


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
    # TODO: Implement with database or NHTSA API
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


# =============================================================================
# Vehicle Years
# =============================================================================


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


# =============================================================================
# Recalls
# =============================================================================


@router.get("/{make}/{model}/{year}/recalls", response_model=List[Recall])
async def get_vehicle_recalls(
    make: str = Path(..., description="Vehicle make (e.g., Toyota)"),
    model: str = Path(..., description="Vehicle model (e.g., Camry)"),
    year: int = Path(..., ge=1900, le=2030, description="Model year"),
    nhtsa_service: NHTSAService = Depends(get_nhtsa_service),
):
    """
    Get recall information for a specific vehicle.

    Fetches recall data from NHTSA (National Highway Traffic Safety Administration).

    Args:
        make: Vehicle manufacturer name
        model: Vehicle model name
        year: Model year

    Returns:
        List of recalls for the specified vehicle

    Raises:
        HTTPException: If NHTSA API request fails
    """
    try:
        recalls = await nhtsa_service.get_recalls(make, model, year)
        return recalls

    except NHTSAError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"NHTSA API error: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch recalls: {str(e)}",
        )


# =============================================================================
# Complaints
# =============================================================================


@router.get("/{make}/{model}/{year}/complaints", response_model=List[Complaint])
async def get_vehicle_complaints(
    make: str = Path(..., description="Vehicle make (e.g., Toyota)"),
    model: str = Path(..., description="Vehicle model (e.g., Camry)"),
    year: int = Path(..., ge=1900, le=2030, description="Model year"),
    nhtsa_service: NHTSAService = Depends(get_nhtsa_service),
):
    """
    Get complaint information for a specific vehicle.

    Fetches complaint data from NHTSA (National Highway Traffic Safety Administration).

    Args:
        make: Vehicle manufacturer name
        model: Vehicle model name
        year: Model year

    Returns:
        List of complaints for the specified vehicle (includes summary and count)

    Raises:
        HTTPException: If NHTSA API request fails
    """
    try:
        complaints = await nhtsa_service.get_complaints(make, model, year)
        return complaints

    except NHTSAError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"NHTSA API error: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch complaints: {str(e)}",
        )
