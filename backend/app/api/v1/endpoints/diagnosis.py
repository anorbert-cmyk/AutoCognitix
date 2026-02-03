"""
Diagnosis endpoints - core functionality for vehicle diagnostics.
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.diagnosis import (
    DiagnosisRequest,
    DiagnosisResponse,
    DiagnosisHistoryItem,
    ProbableCause,
    RepairRecommendation,
    Source,
)
from app.db.postgres.session import get_db
from app.services.diagnosis_service import (
    DiagnosisService,
    DiagnosisServiceError,
    DTCValidationError,
    VINDecodeError,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=DiagnosisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_vehicle(
    request: DiagnosisRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze vehicle based on DTC codes and symptoms.

    This is the main diagnostic endpoint that:
    1. Processes DTC codes and Hungarian symptom text
    2. Searches vector database for similar issues
    3. Queries knowledge graph for diagnostic paths
    4. Generates AI-powered diagnosis with confidence scores

    Args:
        request: Diagnosis request with vehicle info, DTCs, and symptoms
        db: Database session

    Returns:
        Comprehensive diagnosis with probable causes and repair recommendations

    Raises:
        400: Invalid DTC codes or VIN
        500: Internal service error
    """
    try:
        async with DiagnosisService(db) as service:
            result = await service.analyze_vehicle(request)
            return result

    except DTCValidationError as e:
        logger.warning(f"DTC validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except VINDecodeError as e:
        logger.warning(f"VIN decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid VIN: {e}",
        )

    except DiagnosisServiceError as e:
        logger.error(f"Diagnosis service error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during diagnosis. Please try again.",
        )

    except Exception as e:
        logger.exception(f"Unexpected error in analyze_vehicle: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        )


@router.get("/{diagnosis_id}", response_model=DiagnosisResponse)
async def get_diagnosis(
    diagnosis_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific diagnosis by ID.

    Args:
        diagnosis_id: UUID of the diagnosis
        db: Database session

    Returns:
        The diagnosis result

    Raises:
        404: Diagnosis not found
    """
    try:
        async with DiagnosisService(db) as service:
            result = await service.get_diagnosis_by_id(diagnosis_id)

            if result is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Diagnosis {diagnosis_id} not found",
                )

            return result

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error retrieving diagnosis {diagnosis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the diagnosis.",
        )


@router.get("/history/list", response_model=List[DiagnosisHistoryItem])
async def get_diagnosis_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    # TODO: Add authentication dependency
    # current_user: User = Depends(get_current_user),
):
    """
    Get diagnosis history for the current user.

    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of previous diagnoses
    """
    try:
        async with DiagnosisService(db) as service:
            # TODO: Get user_id from authenticated user
            # For now, return empty list as placeholder
            # history = await service.get_user_history(current_user.id, skip, limit)
            return []

    except Exception as e:
        logger.exception(f"Error retrieving diagnosis history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving diagnosis history.",
        )


@router.post("/quick-analyze")
async def quick_analyze(
    dtc_codes: List[str] = Query(..., min_length=1, max_length=10),
    db: AsyncSession = Depends(get_db),
):
    """
    Quick analysis endpoint for single DTC code lookup.

    This is a simplified endpoint that doesn't require vehicle info
    and returns basic DTC information without full RAG analysis.

    Args:
        dtc_codes: List of DTC codes to analyze
        db: Database session

    Returns:
        Basic DTC information and common causes
    """
    try:
        async with DiagnosisService(db) as service:
            # Validate DTC codes
            for code in dtc_codes:
                code = code.upper().strip()
                if not (len(code) == 5 and code[0] in "PBCU" and code[1:].isdigit()):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid DTC code format: {code}",
                    )

            # Get DTC details from repository
            dtc_details = []
            for code in dtc_codes:
                details = await service.dtc_repository.get_by_code(code)
                if details:
                    dtc_details.append({
                        "code": details.code,
                        "description": details.description_hu or details.description_en,
                        "severity": details.severity,
                        "symptoms": details.symptoms,
                        "possible_causes": details.possible_causes,
                    })
                else:
                    dtc_details.append({
                        "code": code,
                        "description": "Hibakód nem található az adatbázisban",
                        "severity": "unknown",
                        "symptoms": [],
                        "possible_causes": [],
                    })

            return {
                "dtc_codes": dtc_details,
                "message": "Részletes diagnózishoz használja a /analyze végpontot járműadatokkal.",
            }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Error in quick analyze: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during quick analysis.",
        )
