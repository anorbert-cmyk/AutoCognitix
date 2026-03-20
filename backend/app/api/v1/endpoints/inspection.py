"""
Műszaki Vizsga (Technical Inspection) endpoints.

Provides vehicle technical inspection evaluation based on DTC codes,
mapping them to Hungarian MOT (Műszaki Vizsga) categories with
risk assessment and repair cost estimates.

Author: AutoCognitix Team
"""

from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.endpoints.auth import get_optional_current_user
from app.api.v1.schemas.inspection import (
    InspectionRequest,
    InspectionResponse,
)
from app.core.logging import get_logger
from app.db.postgres.models import User
from app.services.inspection_service import (
    InspectionService,
    InspectionServiceError,
    get_inspection_service,
)

router = APIRouter()
logger = get_logger(__name__)

# OpenAPI response examples
EVALUATE_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "Műszaki vizsga értékelés sikeres",
        "content": {
            "application/json": {
                "example": {
                    "overall_risk": "high",
                    "risk_score": 0.6,
                    "failing_items": [
                        {
                            "category": "emissions",
                            "category_hu": "Kipufogó emisszió",
                            "issue": "Katalizátor hatásfok alatt (Bank 1)",
                            "related_dtc": "P0420",
                            "severity": "fail",
                            "fix_recommendation": (
                                "Katalizátor hatásfok "
                                "alatt (Bank 1) - javasolt "
                                "alkatrészek: Katalizátor, "
                                "Lambda szonda"
                            ),
                            "estimated_cost_min": 92000,
                            "estimated_cost_max": 535000,
                        }
                    ],
                    "passing_categories": [
                        "body_chassis",
                        "braking",
                        "electrical",
                        "engine_drivetrain",
                        "lighting",
                        "safety_systems",
                        "suspension_steering",
                        "tires_wheels",
                        "visibility",
                    ],
                    "recommendations": [
                        "1 kritikus hiba - a műszaki vizsga "
                        "NEM lenne sikeres a jelenlegi "
                        "állapotban.",
                        "Emisszió-szabályozás javítása "
                        "szükséges a szonda/katalizátor/"
                        "EGR rendszerben.",
                    ],
                    "estimated_total_fix_cost_min": 92000,
                    "estimated_total_fix_cost_max": 535000,
                    "vehicle_info": "Volkswagen Golf 2018 1.4 TSI, 98 420 km",
                    "dtc_count": 1,
                }
            }
        },
    },
    422: {
        "description": "Érvénytelen kérés (validációs hiba)",
    },
    500: {
        "description": "Szerverhiba az értékelés során",
    },
}


@router.post(
    "/evaluate",
    response_model=InspectionResponse,
    status_code=status.HTTP_200_OK,
    responses=EVALUATE_RESPONSES,
    summary="Műszaki vizsga értékelés",
    description=(
        "DTC hibakódok alapján értékeli a jármű műszaki vizsga "
        "esélyeit. Visszaadja a kockázati szintet, a bukási "
        "tételeket és a becsült javítási költségeket."
    ),
)
async def evaluate_inspection(
    request: InspectionRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
) -> InspectionResponse:
    """Evaluate vehicle DTC codes for technical inspection readiness.

    Args:
        request: Vehicle info and DTC codes to evaluate.
        current_user: Authenticated user (optional).

    Returns:
        InspectionResponse with risk assessment.

    Raises:
        HTTPException: On validation or internal errors.
    """
    user_info = f"user_id={current_user.id}" if current_user else "anonymous"
    logger.info(
        "Műszaki vizsga kérés érkezett",
        extra={
            "user": user_info,
            "vehicle_make": request.vehicle_make,
            "vehicle_model": request.vehicle_model,
            "vehicle_year": request.vehicle_year,
            "dtc_count": len(request.dtc_codes),
        },
    )

    try:
        service: InspectionService = get_inspection_service()
        response = await service.evaluate(request)

        logger.info(
            "Műszaki vizsga értékelés sikeres",
            extra={
                "user": user_info,
                "risk_level": response.overall_risk.value,
                "risk_score": response.risk_score,
                "dtc_count": response.dtc_count,
            },
        )
        return response

    except InspectionServiceError as exc:
        logger.error(
            f"Műszaki vizsga szolgáltatás hiba: {exc.message}",
            extra={"details": exc.details},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": ("Hiba történt a műszaki vizsga értékelés során."),
                "error": exc.message,
            },
        ) from exc

    except Exception as exc:
        logger.error(
            "Váratlan hiba a műszaki vizsga értékelésben",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": ("Váratlan szerverhiba történt. Kérjük próbálja újra később."),
            },
        ) from exc
