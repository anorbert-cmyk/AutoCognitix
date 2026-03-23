"""
Megéri Megjavítani? (Worth Repairing?) calculator endpoints.

Provides vehicle valuation vs. repair cost analysis to help
owners decide whether to repair, sell, or scrap their vehicle.
Uses Hungarian market data for accurate estimates.

Author: AutoCognitix Team
"""

from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.endpoints.auth import get_optional_current_user
from app.api.v1.schemas.calculator import (
    CalculatorRequest,
    CalculatorResponse,
)
from app.core.log_sanitizer import sanitize_log
from app.core.logging import get_logger
from app.db.postgres.models import User
from app.services.calculator_service import get_calculator_service

router = APIRouter()
logger = get_logger(__name__)

# OpenAPI response examples
EVALUATE_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "description": "Kalkuláció sikeres",
        "content": {
            "application/json": {
                "example": {
                    "vehicle_value_min": 3200000,
                    "vehicle_value_max": 4100000,
                    "vehicle_value_avg": 3650000,
                    "repair_cost_min": 150000,
                    "repair_cost_max": 220000,
                    "ratio": 0.05,
                    "recommendation": "repair",
                    "recommendation_text": (
                        "Javítás ajánlott. A javítási költség mindössze 5%-a a jármű értékének."
                    ),
                    "breakdown": {
                        "parts_cost": 120000,
                        "labor_cost": 65000,
                        "additional_costs": 0,
                    },
                    "factors": [
                        {
                            "name": "Jármű kora",
                            "impact": "negative",
                            "description": ("8 éves jármű, átlagon felüli értékcsökkenést mutat."),
                        }
                    ],
                    "alternative_scenarios": [
                        {
                            "scenario": "Eladás jelenlegi állapotban",
                            "description": ("A jármű jelenlegi állapotában történő értékesítése."),
                            "estimated_value": 3000000,
                        }
                    ],
                    "confidence_score": 0.75,
                    "currency": "HUF",
                    "ai_disclaimer": (
                        "Ez a becslés tájékoztató jellegű, statisztikai adatokon alapul."
                    ),
                }
            }
        },
    },
    422: {
        "description": "Érvénytelen kérés (validációs hiba)",
    },
    500: {
        "description": "Szerverhiba a kalkuláció során",
    },
}


@router.post(
    "/evaluate",
    response_model=CalculatorResponse,
    status_code=status.HTTP_200_OK,
    responses=EVALUATE_RESPONSES,
    summary="Megéri megjavítani? kalkulátor",
    description=(
        "Jármű értékbecslés és javítási költség elemzés alapján "
        "meghatározza, hogy megéri-e a javítás. Visszaadja az "
        "ajánlást, költségbontást és alternatív forgatókönyveket."
    ),
)
async def evaluate_repair_worthiness(
    request: CalculatorRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
) -> CalculatorResponse:
    """Evaluate whether repairing a vehicle is worth the cost.

    Args:
        request: Vehicle info and optional repair cost.
        current_user: Authenticated user (optional).

    Returns:
        CalculatorResponse with valuation and recommendation.

    Raises:
        HTTPException: On validation or internal errors.
    """
    user_info = f"user_id={current_user.id}" if current_user else "anonymous"
    logger.info(
        "Kalkulátor kérés érkezett",
        extra={
            "user": user_info,
            "vehicle_make": sanitize_log(request.vehicle_make),
            "vehicle_model": sanitize_log(request.vehicle_model),
            "vehicle_year": request.vehicle_year,
            "mileage_km": request.mileage_km,
            "condition": request.condition.value,
        },
    )

    try:
        service = get_calculator_service()
        result: Dict[str, Any] = await service.calculate(
            vehicle_make=request.vehicle_make,
            vehicle_model=request.vehicle_model,
            vehicle_year=request.vehicle_year,
            mileage_km=request.mileage_km,
            condition=request.condition.value,
            repair_cost_huf=request.repair_cost_huf,
            diagnosis_id=request.diagnosis_id,
            fuel_type=request.fuel_type,
        )

        response = CalculatorResponse(**result)

        logger.info(
            "Kalkuláció sikeres",
            extra={
                "user": user_info,
                "recommendation": response.recommendation.value,
                "ratio": response.ratio,
                "confidence": response.confidence_score,
            },
        )
        return response

    except ValueError as exc:
        logger.warning(
            f"Érvénytelen kalkulátor bemenet: {exc}",
            extra={"user": user_info},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": ("Érvénytelen bemenet a kalkulációhoz."),
                "error": str(exc),
            },
        ) from exc

    except Exception as exc:
        logger.error(
            "Váratlan hiba a kalkulátor értékelésben",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": ("Váratlan szerverhiba történt. Kérjük próbálja újra később."),
            },
        ) from exc
