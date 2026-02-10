"""
Vehicle endpoints - vehicle information, VIN decoding, recalls, and complaints.

Provides endpoints to:
- List vehicle makes from Neo4j database
- List vehicle models for a make
- Get available years for a make/model
- Decode VIN using NHTSA API
- Get recalls and complaints from NHTSA
"""

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.v1.schemas.vehicle import (
    PaginatedResponse,
    VehicleCommonIssue,
    VehicleCommonIssuesResponse,
    VehicleMake,
    VehicleModel,
    VehicleYearsResponse,
    VINDecodeRequest,
    VINDecodeResponse,
)
from app.core.logging import get_logger
from app.services.nhtsa_service import (
    Complaint,
    NHTSAError,
    NHTSAService,
    Recall,
    get_nhtsa_service,
)
from app.services.vehicle_service import VehicleService, get_vehicle_service

router = APIRouter()
logger = get_logger(__name__)


# =============================================================================
# OpenAPI Response Examples
# =============================================================================

VIN_DECODE_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
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
                    "country_of_origin": "Germany",
                }
            }
        },
    },
    400: {
        "description": "Invalid VIN",
        "content": {
            "application/json": {
                "examples": {
                    "invalid_length": {
                        "summary": "VIN wrong length",
                        "value": {"detail": "VIN must be exactly 17 characters"},
                    },
                    "invalid_chars": {
                        "summary": "Invalid characters",
                        "value": {
                            "detail": "VIN contains invalid characters (I, O, Q are not allowed)"
                        },
                    },
                }
            }
        },
    },
    502: {
        "description": "NHTSA API error",
        "content": {
            "application/json": {
                "example": {"detail": "NHTSA API error: Service temporarily unavailable"}
            }
        },
    },
}

MAKES_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "List of vehicle makes with pagination",
        "content": {
            "application/json": {
                "example": {
                    "items": [
                        {"id": "audi", "name": "Audi", "country": "Germany", "logo_url": None},
                        {"id": "bmw", "name": "BMW", "country": "Germany", "logo_url": None},
                        {
                            "id": "volkswagen",
                            "name": "Volkswagen",
                            "country": "Germany",
                            "logo_url": None,
                        },
                    ],
                    "total": 150,
                    "limit": 20,
                    "offset": 0,
                    "has_more": True,
                }
            }
        },
    }
}

MODELS_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "List of vehicle models for the specified make",
        "content": {
            "application/json": {
                "example": {
                    "items": [
                        {
                            "id": "golf",
                            "name": "Golf",
                            "make_id": "volkswagen",
                            "year_start": 1974,
                            "year_end": None,
                            "body_types": ["Hatchback", "Wagon"],
                        },
                        {
                            "id": "passat",
                            "name": "Passat",
                            "make_id": "volkswagen",
                            "year_start": 1973,
                            "year_end": None,
                            "body_types": ["Sedan", "Wagon"],
                        },
                    ],
                    "total": 45,
                    "limit": 20,
                    "offset": 0,
                    "has_more": True,
                }
            }
        },
    },
    404: {
        "description": "Make not found",
        "content": {
            "application/json": {"example": {"detail": "No models found for make: UnknownMake"}}
        },
    },
}

YEARS_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "Available vehicle years",
        "content": {
            "application/json": {
                "example": {
                    "years": [2026, 2025, 2024, 2023, 2022, 2021, 2020],
                    "make": "Volkswagen",
                    "model": "Golf",
                }
            }
        },
    },
    404: {
        "description": "Vehicle not found",
        "content": {
            "application/json": {
                "example": {"detail": "No vehicle found for Volkswagen UnknownModel"}
            }
        },
    },
}

RECALLS_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "List of recalls for the specified vehicle",
        "content": {
            "application/json": {
                "example": [
                    {
                        "campaign_number": "24V123000",
                        "manufacturer": "Volkswagen",
                        "make": "Volkswagen",
                        "model": "Golf",
                        "model_year": 2018,
                        "recall_date": "2024-01-15",
                        "component": "FUEL SYSTEM",
                        "summary": "The fuel pump may fail causing the engine to stall without warning.",
                        "consequence": "An engine stall while driving increases the risk of a crash.",
                        "remedy": "Dealers will replace the fuel pump free of charge.",
                        "notes": None,
                        "nhtsa_id": "24V123",
                    }
                ]
            }
        },
    },
    502: {
        "description": "NHTSA API error",
        "content": {
            "application/json": {"example": {"detail": "NHTSA API error: Failed to fetch recalls"}}
        },
    },
}

COMPLAINTS_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "List of complaints for the specified vehicle",
        "content": {
            "application/json": {
                "example": [
                    {
                        "odinumber": "12345678",
                        "manufacturer": "Volkswagen",
                        "make": "Volkswagen",
                        "model": "Golf",
                        "model_year": 2018,
                        "crash": False,
                        "fire": False,
                        "injuries": 0,
                        "deaths": 0,
                        "complaint_date": "2023-06-20",
                        "date_of_incident": "2023-06-15",
                        "components": "ENGINE",
                        "summary": "The vehicle experienced sudden power loss while driving on the highway.",
                    }
                ]
            }
        },
    },
    502: {
        "description": "NHTSA API error",
        "content": {
            "application/json": {
                "example": {"detail": "NHTSA API error: Failed to fetch complaints"}
            }
        },
    },
}

COMMON_ISSUES_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "Common issues for the specified vehicle",
        "content": {
            "application/json": {
                "example": {
                    "make": "Volkswagen",
                    "model": "Golf",
                    "year": 2018,
                    "issues": [
                        {
                            "code": "P0420",
                            "description_en": "Catalyst System Efficiency Below Threshold",
                            "description_hu": "Katalizator hatekonysaga az also hatar alatt",
                            "severity": "medium",
                            "frequency": "common",
                            "occurrence_count": 150,
                        }
                    ],
                }
            }
        },
    }
}


# =============================================================================
# Vehicle Makes
# =============================================================================


@router.get(
    "/makes",
    response_model=PaginatedResponse[VehicleMake],
    responses=MAKES_RESPONSES,
    summary="Get vehicle makes",
    description="""
**Get list of vehicle manufacturers** (makes) from the database.

Queries Neo4j for all unique makes across 8,145+ vehicles.

**Features:**
- Pagination support with limit/offset
- Search filter for finding specific makes
- Country of origin included when available

**Examples:**
- `/api/v1/vehicles/makes` - All makes (paginated)
- `/api/v1/vehicles/makes?search=volk` - Makes containing "volk"
- `/api/v1/vehicles/makes?limit=50&offset=0` - First 50 makes
    """,
)
async def get_vehicle_makes(
    search: Optional[str] = Query(
        None, min_length=1, description="Search term for filtering makes"
    ),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    vehicle_service: VehicleService = Depends(get_vehicle_service),
) -> Dict[str, Any]:
    """
    Get list of vehicle makes (manufacturers).

    Args:
        search: Optional search term to filter makes
        limit: Maximum number of results
        offset: Number of results to skip
        vehicle_service: Vehicle service instance

    Returns:
        Paginated list of vehicle makes
    """
    try:
        makes_data, total = await vehicle_service.get_all_makes(
            search=search,
            limit=limit,
            offset=offset,
        )

        items = [
            VehicleMake(
                id=make["id"],
                name=make["name"],
                country=make.get("country"),
                logo_url=None,
            )
            for make in makes_data
        ]

        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        }

    except Exception as e:
        logger.error(f"Error fetching makes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch vehicle makes: {e!s}",
        )


# =============================================================================
# Vehicle Models
# =============================================================================


@router.get(
    "/models",
    response_model=PaginatedResponse[VehicleModel],
    responses=MODELS_RESPONSES,
    summary="Get vehicle models",
    description="""
**Get list of vehicle models** for a specific manufacturer.

Queries Neo4j for all models matching the specified make.

**Features:**
- Pagination support with limit/offset
- Optional search filter within models
- Year range and body types included

**Examples:**
- `/api/v1/vehicles/models?make=Toyota` - All Toyota models
- `/api/v1/vehicles/models?make=Volkswagen&search=golf` - VW Golf models
    """,
)
async def get_vehicle_models(
    make: str = Query(..., min_length=1, description="Vehicle make (manufacturer)"),
    search: Optional[str] = Query(
        None, min_length=1, description="Search term for filtering models"
    ),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    vehicle_service: VehicleService = Depends(get_vehicle_service),
) -> Dict[str, Any]:
    """
    Get list of models for a specific make.

    Args:
        make: Vehicle make (manufacturer)
        search: Optional search term to filter models
        limit: Maximum number of results
        offset: Number of results to skip
        vehicle_service: Vehicle service instance

    Returns:
        Paginated list of vehicle models
    """
    try:
        models_data, total = await vehicle_service.get_models_for_make(
            make=make,
            search=search,
            limit=limit,
            offset=offset,
        )

        if not models_data and offset == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No models found for make: {make}",
            )

        items = [
            VehicleModel(
                id=model["id"],
                name=model["name"],
                make_id=model["make_id"],
                year_start=model.get("year_start"),
                year_end=model.get("year_end"),
                body_types=model.get("body_types", []),
            )
            for model in models_data
        ]

        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching models for {make}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch vehicle models: {e!s}",
        )


# =============================================================================
# Vehicle Years
# =============================================================================


@router.get(
    "/years",
    response_model=VehicleYearsResponse,
    responses=YEARS_RESPONSES,
    summary="Get available years",
    description="""
**Get list of available vehicle years** for a specific make and model.

Returns all years when the vehicle was produced.

**Example:**
`/api/v1/vehicles/years?make=Toyota&model=Camry`
    """,
)
async def get_available_years(
    make: str = Query(..., min_length=1, description="Vehicle make"),
    model: str = Query(..., min_length=1, description="Vehicle model"),
    vehicle_service: VehicleService = Depends(get_vehicle_service),
) -> VehicleYearsResponse:
    """
    Get list of available vehicle years for a specific make and model.

    Args:
        make: Vehicle make
        model: Vehicle model
        vehicle_service: Vehicle service instance

    Returns:
        List of years in descending order
    """
    try:
        years = await vehicle_service.get_years_for_vehicle(make=make, model=model)

        if not years:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No vehicle found for {make} {model}",
            )

        return VehicleYearsResponse(
            years=years,
            make=make,
            model=model,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching years for {make} {model}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch vehicle years: {e!s}",
        )


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
) -> VINDecodeResponse:
    """
    Decode a VIN (Vehicle Identification Number) to get vehicle details.

    Uses NHTSA vPIC API for decoding.

    Args:
        request: VIN decode request
        nhtsa_service: NHTSA service instance

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
        logger.error(f"Error decoding VIN {vin}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to decode VIN: {e!s}",
        )


# =============================================================================
# Recalls
# =============================================================================


@router.get(
    "/{make}/{model}/{year}/recalls",
    response_model=List[Recall],
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
) -> List[Recall]:
    """
    Get recall information for a specific vehicle.

    Fetches recall data from NHTSA (National Highway Traffic Safety Administration).

    Args:
        make: Vehicle manufacturer name
        model: Vehicle model name
        year: Model year
        nhtsa_service: NHTSA service instance

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
        logger.error(f"Error fetching recalls for {make} {model} {year}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch recalls: {e!s}",
        )


# =============================================================================
# Complaints
# =============================================================================


@router.get(
    "/{make}/{model}/{year}/complaints",
    response_model=List[Complaint],
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
) -> List[Complaint]:
    """
    Get complaint information for a specific vehicle.

    Fetches complaint data from NHTSA (National Highway Traffic Safety Administration).

    Args:
        make: Vehicle manufacturer name
        model: Vehicle model name
        year: Model year
        nhtsa_service: NHTSA service instance

    Returns:
        List of complaints for the specified vehicle

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
        logger.error(f"Error fetching complaints for {make} {model} {year}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch complaints: {e!s}",
        )


# =============================================================================
# Common Issues (from Neo4j)
# =============================================================================


@router.get(
    "/{make}/{model}/common-issues",
    response_model=VehicleCommonIssuesResponse,
    responses=COMMON_ISSUES_RESPONSES,
    summary="Get common vehicle issues",
    description="""
**Get common DTC codes and issues** for a specific vehicle from the Neo4j knowledge graph.

Returns issues that are commonly reported for this make/model combination,
including frequency and occurrence data.

**Example:**
`/api/v1/vehicles/Volkswagen/Golf/common-issues?year=2018`
    """,
)
async def get_vehicle_common_issues(
    make: str = Path(..., description="Vehicle make (e.g., Volkswagen)"),
    model: str = Path(..., description="Vehicle model (e.g., Golf)"),
    year: Optional[int] = Query(None, ge=1900, le=2030, description="Optional year filter"),
    vehicle_service: VehicleService = Depends(get_vehicle_service),
) -> VehicleCommonIssuesResponse:
    """
    Get common issues for a specific vehicle from the knowledge graph.

    Args:
        make: Vehicle manufacturer name
        model: Vehicle model name
        year: Optional year filter
        vehicle_service: Vehicle service instance

    Returns:
        Common issues response with list of DTC codes and their frequency
    """
    try:
        issues_data = await vehicle_service.get_vehicle_common_issues(
            make=make,
            model=model,
            year=year,
        )

        issues = [
            VehicleCommonIssue(
                code=issue["code"],
                description_en=issue.get("description_en"),
                description_hu=issue.get("description_hu"),
                severity=issue.get("severity"),
                frequency=issue.get("frequency"),
                occurrence_count=issue.get("occurrence_count"),
            )
            for issue in issues_data
        ]

        return VehicleCommonIssuesResponse(
            make=make,
            model=model,
            year=year,
            issues=issues,
        )

    except Exception as e:
        logger.error(f"Error fetching common issues for {make} {model}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch common issues: {e!s}",
        )
