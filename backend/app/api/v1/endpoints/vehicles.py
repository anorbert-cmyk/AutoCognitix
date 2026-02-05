"""
Vehicle endpoints - vehicle information and VIN decoding.
"""

from __future__ import annotations

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
# OpenAPI Response Examples
# =============================================================================

VIN_DECODE_RESPONSES = {
    200: {
        "description": "VIN successfully decoded",
        "content": {
            "application/json": {
                "example": {
                    "vin": "WVWZZZ3CZWE123456",
                    "make": "Volkswagen",
                    "model": "Golf",
                    "year": 2018,
                    "trim": "Hatchback",
                    "engine": "2.0L 4-cyl Gasoline",
                    "transmission": "Automatic",
                    "drive_type": "FWD",
                    "body_type": "Hatchback",
                    "fuel_type": "Gasoline",
                    "region": "Europe",
                    "country_of_origin": "Germany"
                }
            }
        }
    },
    400: {
        "description": "Invalid VIN",
        "content": {
            "application/json": {
                "examples": {
                    "invalid_length": {
                        "summary": "VIN wrong length",
                        "value": {"detail": "VIN must be exactly 17 characters"}
                    },
                    "invalid_chars": {
                        "summary": "Invalid characters",
                        "value": {"detail": "VIN contains invalid characters (I, O, Q are not allowed)"}
                    }
                }
            }
        }
    },
    502: {
        "description": "NHTSA API error",
        "content": {
            "application/json": {
                "example": {"detail": "NHTSA API error: Service temporarily unavailable"}
            }
        }
    }
}

MAKES_RESPONSES = {
    200: {
        "description": "List of vehicle makes",
        "content": {
            "application/json": {
                "example": [
                    {"id": "volkswagen", "name": "Volkswagen", "country": "Germany", "logo_url": None},
                    {"id": "audi", "name": "Audi", "country": "Germany", "logo_url": None},
                    {"id": "bmw", "name": "BMW", "country": "Germany", "logo_url": None}
                ]
            }
        }
    }
}

MODELS_RESPONSES = {
    200: {
        "description": "List of vehicle models for the specified make",
        "content": {
            "application/json": {
                "example": [
                    {
                        "id": "golf",
                        "name": "Golf",
                        "make_id": "volkswagen",
                        "year_start": 1974,
                        "year_end": None,
                        "body_types": []
                    },
                    {
                        "id": "passat",
                        "name": "Passat",
                        "make_id": "volkswagen",
                        "year_start": 1973,
                        "year_end": None,
                        "body_types": []
                    }
                ]
            }
        }
    }
}

YEARS_RESPONSES = {
    200: {
        "description": "Available vehicle years",
        "content": {
            "application/json": {
                "example": {"years": [2027, 2026, 2025, 2024, 2023, 2022, 2021, 2020]}
            }
        }
    }
}

RECALLS_RESPONSES = {
    200: {
        "description": "List of recalls for the specified vehicle",
        "content": {
            "application/json": {
                "example": [
                    {
                        "campaign_number": "24V123000",
                        "manufacturer": "Volkswagen",
                        "subject": "Fuel Pump May Fail",
                        "summary": "The fuel pump may fail causing the engine to stall without warning.",
                        "consequence": "An engine stall while driving increases the risk of a crash.",
                        "remedy": "Dealers will replace the fuel pump free of charge.",
                        "report_received_date": "2024-01-15",
                        "component": "FUEL SYSTEM"
                    }
                ]
            }
        }
    },
    502: {
        "description": "NHTSA API error",
        "content": {
            "application/json": {
                "example": {"detail": "NHTSA API error: Failed to fetch recalls"}
            }
        }
    }
}

COMPLAINTS_RESPONSES = {
    200: {
        "description": "List of complaints for the specified vehicle",
        "content": {
            "application/json": {
                "example": [
                    {
                        "odi_number": "12345678",
                        "manufacturer": "Volkswagen",
                        "crash": False,
                        "fire": False,
                        "number_of_injuries": 0,
                        "number_of_deaths": 0,
                        "date_of_incident": "2023-06-15",
                        "date_complaint_filed": "2023-06-20",
                        "summary": "The vehicle experienced sudden power loss while driving on the highway.",
                        "components": "ENGINE"
                    }
                ]
            }
        }
    },
    502: {
        "description": "NHTSA API error",
        "content": {
            "application/json": {
                "example": {"detail": "NHTSA API error: Failed to fetch complaints"}
            }
        }
    }
}


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


@router.post(
    "/decode-vin",
    response_model=VINDecodeResponse,
    responses=VIN_DECODE_RESPONSES,
    summary="Decode VIN",
    description="""
**Decode a Vehicle Identification Number (VIN)** to retrieve vehicle details.

Uses the NHTSA vPIC API for decoding.

**VIN Format:**
- Exactly 17 characters
- Characters I, O, Q are not allowed
- Contains manufacturer, vehicle attributes, and serial number

**Example VINs:**
- `WVWZZZ3CZWE123456` - European Volkswagen
- `1HGBH41JXMN109186` - North American Honda
    """,
)
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
            detail=f"Failed to decode VIN: {e!s}",
        )


# =============================================================================
# Vehicle Makes
# =============================================================================


@router.get(
    "/makes",
    response_model=list[VehicleMake],
    responses=MAKES_RESPONSES,
    summary="Get vehicle makes",
    description="""
**Get list of vehicle manufacturers** (makes).

Optionally filter by search term to find specific makes.

**Examples:**
- `/api/v1/vehicles/makes` - All makes
- `/api/v1/vehicles/makes?search=volk` - Makes containing "volk"
    """,
)
async def get_vehicle_makes(
    search: str | None = Query(None, min_length=1, description="Search term for filtering makes"),
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


@router.get(
    "/models/{make_id}",
    response_model=list[VehicleModel],
    responses=MODELS_RESPONSES,
    summary="Get vehicle models",
    description="""
**Get list of vehicle models** for a specific manufacturer.

Optionally filter by year to get models available for that year.

**Examples:**
- `/api/v1/vehicles/models/volkswagen` - All VW models
- `/api/v1/vehicles/models/volkswagen?year=2020` - VW models from 2020
    """,
)
async def get_vehicle_models(
    make_id: str,
    year: int | None = Query(None, ge=1900, le=2030, description="Filter by year"),
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


@router.get(
    "/years",
    responses=YEARS_RESPONSES,
    summary="Get available years",
    description="""
**Get list of available vehicle years** for selection.

Returns years from 1980 to current year + 1 (for next model year vehicles).
    """,
)
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


@router.get(
    "/{make}/{model}/{year}/recalls",
    response_model=list[Recall],
    responses=RECALLS_RESPONSES,
    summary="Get vehicle recalls",
    description="""
**Get recall information** for a specific vehicle from NHTSA.

Recalls are safety-related defects reported to the National Highway Traffic Safety Administration.

**Example:**
`/api/v1/vehicles/Volkswagen/Golf/2018/recalls`
    """,
)
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
            detail=f"Failed to fetch recalls: {e!s}",
        )


# =============================================================================
# Complaints
# =============================================================================


@router.get(
    "/{make}/{model}/{year}/complaints",
    response_model=list[Complaint],
    responses=COMPLAINTS_RESPONSES,
    summary="Get vehicle complaints",
    description="""
**Get consumer complaints** for a specific vehicle from NHTSA.

Complaints are consumer-reported issues submitted to NHTSA.
Includes crash and injury information when reported.

**Example:**
`/api/v1/vehicles/Volkswagen/Golf/2018/complaints`
    """,
)
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
            detail=f"Failed to fetch complaints: {e!s}",
        )
